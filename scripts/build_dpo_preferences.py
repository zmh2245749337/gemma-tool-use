"""Collect training-only model errors and turn them into DPO preference pairs.

The script deliberately accepts an arbitrary labelled split, but its normal
training invocation uses ``data/tool_use/train.jsonl`` only.  Validation pairs
are generated separately for early stopping; test and challenge files are not
inputs to the DPO training pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.preference_data import classify_model_error
from gemma_eval.tool_use_data import canonical_json, load_jsonl, tool_use_prompt


def render_chat_prompt(tokenizer, row: dict) -> str:
    prompt = tool_use_prompt(row["history"], row["current_user"], row["previous_state"])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from a tool-use adapter's training errors")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/tool_use/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tool_use/dpo_train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--precision", choices=["fp16", "4bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "false_tool_call",
            "decision_mismatch",
            "tool_mismatch",
            "argument_mismatch",
            "state_mismatch",
            "schema_invalid",
        ],
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel

    from gemma_eval.modeling import build_chat_inputs, load_model_bundle

    if not args.adapter.exists():
        raise FileNotFoundError(f"adapter not found: {args.adapter}")
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    bundle = load_model_bundle(args.model_id, args.precision)
    model = PeftModel.from_pretrained(bundle.model, args.adapter)
    model.eval()

    accepted = set(args.categories)
    pairs: list[dict] = []
    categories: Counter[str] = Counter()
    try:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="collecting DPO pairs", unit="sample")
    except ImportError:
        iterator = rows

    for row in iterator:
        rendered_prompt = render_chat_prompt(bundle.tokenizer, row)
        inputs = build_chat_inputs(bundle.tokenizer, model, tool_use_prompt(row["history"], row["current_user"], row["previous_state"]))
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        raw = bundle.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        category, comparison = classify_model_error(raw, row["output"])
        if category is None or category not in accepted:
            continue
        categories[category] += 1
        pairs.append(
            {
                "prompt": rendered_prompt,
                "chosen": canonical_json(row["output"]),
                "rejected": raw,
                "sample_id": row.get("sample_id"),
                "error_category": category,
                "source": row.get("source"),
                "comparison": comparison,
            }
        )

    if not pairs:
        raise RuntimeError(
            "no preference pairs were collected; inspect the adapter, generation settings, "
            "or requested error categories"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in pairs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "scope": "model-generated preference pairs; source rows are never relabelled",
        "input": str(args.input),
        "adapter": str(args.adapter),
        "model_id": args.model_id,
        "precision": args.precision,
        "input_samples": len(rows),
        "preference_pairs": len(pairs),
        "error_categories": dict(categories),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
