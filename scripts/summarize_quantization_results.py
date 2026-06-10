#!/usr/bin/env python3
"""Create a comparison table across separate HimalayaGPT quantization runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Path prefix to write .json and .md comparison files.",
    )
    return parser.parse_args()


def quantization_for_model(model: str) -> str:
    suffix = model.rsplit("-", 1)[-1]
    return suffix if suffix in {"bf16", "q8", "q4"} else "unknown"


def load_model_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for summary_path in sorted((run_dir / "models").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_dir"] = str(run_dir)
        summary["summary_path"] = str(summary_path)
        summary["samples_path"] = str(summary_path.with_name("samples.jsonl"))
        summary["quantization"] = quantization_for_model(summary["model"])
        summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No model summaries found under {run_dir}")
    return summaries


def metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HimalayaGPT Quantization Comparison",
        "",
        "| Model | Quantization | Prompt Strict | Inst Strict | Prompt Loose | Inst Loose | Errors | Samples |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item["model"]):
        lines.append(
            "| {model} | {quantization} | {prompt_strict} | {inst_strict} | {prompt_loose} | {inst_loose} | {errors} | {samples} |".format(
                model=row["model"],
                quantization=row["quantization"],
                prompt_strict=metric(row.get("prompt_level_strict_acc")),
                inst_strict=metric(row.get("inst_level_strict_acc")),
                prompt_loose=metric(row.get("prompt_level_loose_acc")),
                inst_loose=metric(row.get("inst_level_loose_acc")),
                errors=row.get("errored_examples"),
                samples=row.get("examples"),
            )
        )
    lines.append("")
    lines.append("Each row is computed from that model's separate raw `samples.jsonl` benchmark artifact.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = []
    for run_dir in args.run_dirs:
        rows.extend(load_model_summaries(run_dir))

    output_json = args.output_prefix.with_suffix(".json")
    output_md = args.output_prefix.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_md, rows)
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
