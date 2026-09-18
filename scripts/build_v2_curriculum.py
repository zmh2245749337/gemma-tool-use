"""Build a deterministic V2 curriculum from training-only examples.

The script never changes labels and never copies validation/test/challenge rows
into training.  It increases the sampling frequency of training examples that
match the error slices observed after V1: no-tool boundaries, corrections,
parameter-dense calls and long state tracking.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(ROOT / "src")
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from gemma_eval.tool_use_data import load_jsonl, state_item_count


CORRECTION_MARKERS = ("不是", "改成", "换成", "更正", "纠正", "说错", "刚才忘", "刚才没有")


def argument_item_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(argument_item_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(argument_item_count(child) for child in value)
    return 1


def is_parameter_dense(row: dict[str, Any], minimum_items: int) -> bool:
    call = row["output"].get("tool_call")
    return bool(call and argument_item_count(call["arguments"]) >= minimum_items)


def category_rows(
    rows: list[dict[str, Any]], minimum_argument_items: int, minimum_state_items: int
) -> dict[str, list[dict[str, Any]]]:
    return {
        "no_tool_boundary": [row for row in rows if row["output"]["decision"] == "no_tool"],
        "condition_correction": [
            row for row in rows if any(marker in row["current_user"] for marker in CORRECTION_MARKERS)
        ],
        "parameter_dense": [
            row for row in rows if is_parameter_dense(row, minimum_argument_items)
        ],
        "long_state": [
            row
            for row in rows
            if state_item_count(row["output"]["belief_state"]) >= minimum_state_items
        ],
    }


def duplicate(row: dict[str, Any], category: str, copy_index: int) -> dict[str, Any]:
    item = copy.deepcopy(row)
    original_id = str(row["sample_id"])
    item["sample_id"] = f"{original_id}__v2_{category}_{copy_index}"
    source = dict(item.get("source") or {})
    source["augmentation"] = {
        "version": "v2",
        "strategy": category,
        "derived_from": original_id,
        "labels_modified": False,
    }
    item["source"] = source
    return item


def sample_with_cap(
    rows: list[dict[str, Any]], copies: int, cap: int | None, rng: random.Random
) -> list[tuple[dict[str, Any], int]]:
    selected = list(rows)
    rng.shuffle(selected)
    if cap is not None:
        selected = selected[:cap]
    return [(row, copy_index) for row in selected for copy_index in range(1, copies + 1)]


def assert_no_heldout_leakage(
    train_rows: list[dict[str, Any]], heldout_paths: list[Path]
) -> None:
    train_ids = {str(row["sample_id"]) for row in train_rows}
    for path in heldout_paths:
        heldout_ids = {str(row["sample_id"]) for row in load_jsonl(path)}
        overlap = train_ids & heldout_ids
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"training/held-out sample_id overlap in {path}: {examples}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training-only V2 hard-example curriculum")
    parser.add_argument("--train", type=Path, default=Path("data/tool_use/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tool_use/train_v2.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/tool_use/train_v2_manifest.json")
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        nargs="*",
        default=[
            Path("data/tool_use/validation.jsonl"),
            Path("data/tool_use/test.jsonl"),
            Path("data/tool_use/challenge.jsonl"),
        ],
    )
    parser.add_argument("--no-tool-copies", type=int, default=1)
    parser.add_argument("--correction-copies", type=int, default=3)
    parser.add_argument("--parameter-dense-copies", type=int, default=1)
    parser.add_argument("--parameter-dense-cap", type=int, default=300)
    parser.add_argument("--long-state-copies", type=int, default=1)
    parser.add_argument("--long-state-cap", type=int, default=200)
    parser.add_argument("--minimum-argument-items", type=int, default=4)
    parser.add_argument("--minimum-state-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = load_jsonl(args.train)
    assert_no_heldout_leakage(train_rows, args.heldout)
    buckets = category_rows(
        train_rows, args.minimum_argument_items, args.minimum_state_items
    )
    settings = {
        "no_tool_boundary": (args.no_tool_copies, None),
        "condition_correction": (args.correction_copies, None),
        "parameter_dense": (args.parameter_dense_copies, args.parameter_dense_cap),
        "long_state": (args.long_state_copies, args.long_state_cap),
    }
    augmented: list[dict[str, Any]] = []
    added_counts: Counter[str] = Counter()
    for category, (copies, cap) in settings.items():
        if copies < 0:
            raise ValueError(f"copies must be non-negative: {category}={copies}")
        for row, copy_index in sample_with_cap(buckets[category], copies, cap, rng):
            augmented.append(duplicate(row, category, copy_index))
            added_counts[category] += 1

    combined = [copy.deepcopy(row) for row in train_rows] + augmented
    rng.shuffle(combined)
    write_jsonl(args.output, combined)
    manifest = {
        "scope": "training-only V2 curriculum; labels are unchanged",
        "seed": args.seed,
        "input": str(args.train),
        "output": str(args.output),
        "heldout_checked": [str(path) for path in args.heldout],
        "base_samples": len(train_rows),
        "v2_samples": len(combined),
        "eligible_examples": {name: len(rows) for name, rows in buckets.items()},
        "added_examples": dict(added_counts),
        "settings": {
            "copies_and_caps": settings,
            "minimum_argument_items": args.minimum_argument_items,
            "minimum_state_items": args.minimum_state_items,
        },
        "decision_distribution": dict(
            Counter(row["output"]["decision"] for row in combined)
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
