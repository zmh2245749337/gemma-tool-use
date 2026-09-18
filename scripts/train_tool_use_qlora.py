"""QLoRA fine-tuning for context-aware structured tool decisions."""

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
from gemma_eval.tool_use_data import load_jsonl
from gemma_eval.traces import render_sft_pair


def encode_example(tokenizer, row: dict, max_length: int) -> dict[str, list[int]]:
    prompt, answer = render_sft_pair(row)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    if hasattr(prompt_ids, "get") and "input_ids" in prompt_ids:
        prompt_ids = prompt_ids["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    all_ids = prompt_ids + answer_ids + eos
    labels = [-100] * len(prompt_ids) + answer_ids + eos
    if len(all_ids) > max_length:
        # Preserve the complete target whenever possible; truncate oldest prompt tokens first.
        answer_length = len(answer_ids) + len(eos)
        if answer_length >= max_length:
            raise ValueError(
                f"target alone has {answer_length} tokens; increase --max-length above {max_length}"
            )
        keep_prompt = max_length - answer_length
        all_ids = prompt_ids[-keep_prompt:] + answer_ids + eos
        labels = [-100] * keep_prompt + answer_ids + eos
    return {"input_ids": all_ids, "attention_mask": [1] * len(all_ids), "labels": labels}


class ToolUseDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rows: list[dict], max_length: int):
        self.items = [encode_example(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for multi-turn tool decisions")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--train", type=Path, default=Path("data/tool_use/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/tool_use/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tool_use_qlora_adapter"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=None, help="small local smoke run")
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps")

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires an NVIDIA GPU; inference-only tests can run on CPU.")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_validation_samples:
        validation_rows = validation_rows[: args.max_validation_samples]
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=token, device_map="auto", quantization_config=quantization
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()
    train_dataset = ToolUseDataset(tokenizer, train_rows, args.max_length)
    validation_dataset = ToolUseDataset(tokenizer, validation_rows, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
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
        "scope": "Agent Trace SFT for multi-turn state tracking and guarded tool decisions",
        "trace_schema_version": "1.0",
        "base_model": args.model_id,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "trainable_parameter_note": "QLoRA adapters only; base weights remain frozen",
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
