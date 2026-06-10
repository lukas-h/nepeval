#!/usr/bin/env python3
"""Download only the Nepali IndicIFEval splits and write JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


DATASET_ID = "ai4bharat/IndicIFEval"
DATASET_URL = "https://huggingface.co/datasets/ai4bharat/IndicIFEval"
LANGUAGE_SPLIT = "ne"
TRACKS = ("indicifeval-ground", "indicifeval-trans")
EXPECTED_COUNTS = {
    "indicifeval-ground": 341,
    "indicifeval-trans": 490,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the Nepali-only split from both IndicIFEval tracks. "
            "The language split is intentionally fixed to 'ne'."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/indicifeval_ne"),
        help="Directory to write Nepali JSONL files into.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision to read.",
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Fail if current row counts differ from the counts documented when this repo was set up.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default))
            handle.write("\n")


def find_nepali_parquet_files(track: str, revision: str) -> list[str]:
    prefix = f"{track}/{LANGUAGE_SPLIT}-"
    files = [
        repo_file
        for repo_file in list_repo_files(DATASET_ID, repo_type="dataset", revision=revision)
        if repo_file.startswith(prefix) and repo_file.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(
            f"No Nepali parquet files found for {DATASET_ID}:{track} at revision {revision}"
        )
    return sorted(files)


def load_track(track: str, revision: str) -> tuple[list[dict[str, Any]], list[str]]:
    source_files = find_nepali_parquet_files(track, revision)
    rows = []
    for source_file in source_files:
        parquet_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=source_file,
            revision=revision,
        )
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            record = dict(row)
            record["source_dataset"] = DATASET_ID
            record["source_config"] = track
            record["source_file"] = source_file
            record["language_split"] = LANGUAGE_SPLIT
            rows.append(record)
    return rows, source_files


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "revision": args.revision,
        "language_split": LANGUAGE_SPLIT,
        "license": "cc-by-4.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracks": {},
    }
    all_rows: list[dict[str, Any]] = []

    for track in TRACKS:
        rows, source_files = load_track(track, args.revision)
        expected = EXPECTED_COUNTS[track]
        if len(rows) != expected:
            message = (
                f"{track}/{LANGUAGE_SPLIT} has {len(rows)} rows; "
                f"expected {expected} from the setup-time dataset card."
            )
            if args.strict_counts:
                print(f"error: {message}", file=sys.stderr)
                return 1
            print(f"warning: {message}", file=sys.stderr)

        track_path = args.out_dir / track / f"{LANGUAGE_SPLIT}.jsonl"
        write_jsonl(track_path, rows)
        all_rows.extend(rows)
        manifest["tracks"][track] = {
            "rows": len(rows),
            "source_files": source_files,
            "path": str(track_path),
        }
        print(f"wrote {len(rows):>4} rows -> {track_path}")

    combined_path = args.out_dir / "all.jsonl"
    write_jsonl(combined_path, all_rows)
    manifest["combined"] = {
        "rows": len(all_rows),
        "path": str(combined_path),
    }

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(all_rows):>4} rows -> {combined_path}")
    print(f"wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
