"""Collect V2 training and evaluation outputs into JSON and Markdown summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "strict_json_valid_rate",
    "schema_valid_rate",
    "state_slot_f1",
    "decision_accuracy",
    "tool_accuracy",
    "arguments_exact_rate",
    "argument_slot_f1",
    "no_tool_decision_accuracy",
    "no_tool_false_positive_rate",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed V2 run")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/tool_use/train_v2_manifest.json")
    )
    parser.add_argument(
        "--training", type=Path, default=Path("artifacts/tool_use_qlora_v2_adapter/training_metrics.json")
    )
    parser.add_argument(
        "--validation", type=Path, default=Path("reports/tool_use_qlora_v2_validation_4bit.json")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("reports/tool_use_qlora_v2_4bit.json")
    )
    parser.add_argument(
        "--challenge", type=Path, default=Path("reports/tool_use_qlora_v2_challenge_4bit.json")
    )
    parser.add_argument("--json-output", type=Path, default=Path("reports/v2_run_summary.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/V2_RUN_SUMMARY.md"))
    args = parser.parse_args()

    paths = {
        "manifest": args.manifest,
        "training": args.training,
        "validation": args.validation,
        "test": args.test,
        "challenge": args.challenge,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing V2 outputs: " + ", ".join(missing))

    reports = {name: read_json(paths[name]) for name in ("validation", "test", "challenge")}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": read_json(args.manifest),
        "training": read_json(args.training),
        "evaluations": {
            name: {
                "dataset": report.get("dataset"),
                "adapter": report.get("adapter"),
                "metrics": report["metrics"],
                "report_path": str(paths[name]),
            }
            for name, report in reports.items()
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# V2 Run Summary",
        "",
        f"- Generated: {summary['created_at_utc']}",
        f"- Base training samples: {summary['dataset']['base_samples']}",
        f"- V2 training samples: {summary['dataset']['v2_samples']}",
        "",
        "| Metric | Validation | Test | Challenge |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(percent(reports[split]["metrics"].get(metric)) for split in reports)
            + " |"
        )
    lines.extend(
        [
            "",
            "Full per-sample records remain in the three evaluation JSON files.",
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"JSON summary: {args.json_output}")
    print(f"Markdown summary: {args.markdown_output}")


if __name__ == "__main__":
    main()
