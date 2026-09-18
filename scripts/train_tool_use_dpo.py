"""Run QLoRA DPO on preference pairs collected from an SFT tool-use model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.env import resolve_hf_token


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(key), str) and row[key] for key in ("prompt", "chosen", "rejected")):
                raise ValueError(f"{path}:{line_number} requires non-empty prompt, chosen and rejected strings")
            rows.append({key: row[key] for key in ("prompt", "chosen", "rejected")})
    if not rows:
        raise ValueError(f"{path} contains no preference pairs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA DPO for multi-turn structured tool decisions")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/tool_use/dpo_train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/tool_use/dpo_validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tool_use_dpo_adapter"))
    parser.add_argument("--run-name", default="tool-use-dpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps")
    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires an NVIDIA GPU")
    if not args.sft_adapter.exists():
        raise FileNotFoundError(f"SFT adapter not found: {args.sft_adapter}")

    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_pairs = load_pairs(args.train)
    validation_pairs = load_pairs(args.validation)
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=token, device_map="auto", quantization_config=quantization
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    model.enable_input_require_grads()

    dpo_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        use_cache=False,
        max_length=args.max_length,
        truncation_mode="keep_end",
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        run_name=args.run_name,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=Dataset.from_list(train_pairs),
        eval_dataset=Dataset.from_list(validation_pairs),
        processing_class=tokenizer,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    validation_metrics = next(
        (entry for entry in reversed(trainer.state.log_history) if "eval_loss" in entry), {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "scope": "DPO on model-generated tool-use errors; reference adapter is copied from the SFT adapter",
        "base_model": args.model_id,
        "sft_adapter": str(args.sft_adapter),
        "train_preference_pairs": len(train_pairs),
        "validation_preference_pairs": len(validation_pairs),
        "training_args": vars(args),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
