#!/usr/bin/env python3
"""Run Nepali IndicIFEval against every model exposed by an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_BASE_URL = "https://himalayagpt.api.scalabs.ai/v1"
DEFAULT_DATA_DIR = Path("data/indicifeval_ne")
DEFAULT_RESULTS_ROOT = Path("results/himalayagpt")
TASKS = (
    ("indicifeval-ground", Path("data/indicifeval_ne/indicifeval-ground/ne.jsonl"), Path("lm_eval_tasks/indicifeval-ground/utils.py")),
    ("indicifeval-trans", Path("data/indicifeval_ne/indicifeval-trans/ne.jsonl"), Path("lm_eval_tasks/indicifeval-trans/utils.py")),
)


@dataclass(frozen=True)
class EvalDoc:
    track: str
    index: int
    row: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return f"{self.track}:{self.row.get('key', self.index)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-token-env", default="NEPEVAL_API_TOKEN")
    parser.add_argument("--fallback-token-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", help="Optional model IDs. Defaults to every /models entry.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-track smoke-test limit.")
    parser.add_argument("--store-raw-api-response", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def require_token(args: argparse.Namespace) -> str:
    token = os.getenv(args.api_token_env) or os.getenv(args.fallback_token_env)
    if not token:
        raise SystemExit(
            f"Set {args.api_token_env} or {args.fallback_token_env}; the bearer token is intentionally not stored in this repo."
        )
    return token


def api_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def request_json(
    method: str,
    url: str,
    token: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], float]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
            latency = time.perf_counter() - started
            response.raise_for_status()
            return response.json(), latency
        except Exception as exc:  # noqa: BLE001 - persisted in result artifacts.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(str(last_error))


def list_models(base_url: str, token: str, timeout: float, retries: int) -> list[dict[str, Any]]:
    payload, _ = request_json("GET", api_url(base_url, "models"), token, timeout=timeout, retries=retries)
    models = payload.get("data", [])
    if not isinstance(models, list):
        raise ValueError(f"Unexpected /models response: {payload}")
    return models


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_docs(limit: int | None) -> list[EvalDoc]:
    docs: list[EvalDoc] = []
    for track, path, _utils_path in TASKS:
        rows = load_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        docs.extend(EvalDoc(track=track, index=index, row=row) for index, row in enumerate(rows))
    return docs


def load_task_utils() -> dict[str, Any]:
    modules = {}
    for track, _path, utils_path in TASKS:
        module_name = f"_nepeval_runner_{track.replace('-', '_')}_utils"
        spec = importlib.util.spec_from_file_location(module_name, utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {utils_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[track] = module
    return modules


def extract_message_content(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices") or []
    if not choices:
        return "", None
    first = choices[0]
    message = first.get("message") or {}
    return message.get("content") or "", first.get("finish_reason")


def evaluate_one(
    *,
    doc: EvalDoc,
    model: str,
    base_url: str,
    token: str,
    args: argparse.Namespace,
    task_utils: dict[str, Any],
) -> dict[str, Any]:
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": doc.row["prompt"]}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    result: dict[str, Any] = {
        "sample_id": doc.sample_id,
        "track": doc.track,
        "dataset_index": doc.index,
        "key": doc.row.get("key"),
        "prompt": doc.row.get("prompt"),
        "instruction_id_list": doc.row.get("instruction_id_list"),
        "kwargs": doc.row.get("kwargs"),
        "tags": doc.row.get("tags"),
        "resp_lang": doc.row.get("resp_lang"),
        "source_file": doc.row.get("source_file"),
        "model": model,
        "request": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }

    try:
        payload, latency = request_json(
            "POST",
            api_url(base_url, "chat/completions"),
            token,
            json_body=request_body,
            timeout=args.timeout,
            retries=args.retries,
        )
        response_text, finish_reason = extract_message_content(payload)
        result.update(
            {
                "ok": True,
                "latency_sec": latency,
                "response": response_text,
                "finish_reason": finish_reason,
                "usage": payload.get("usage"),
            }
        )
        if args.store_raw_api_response:
            result["api_response"] = payload
    except Exception as exc:  # noqa: BLE001 - errors are benchmark artifacts.
        response_text = ""
        result.update({"ok": False, "latency_sec": None, "response": "", "error": str(exc)})

    scores = task_utils[doc.track].process_results_ne(doc.row, [response_text])
    result["metrics"] = {
        "prompt_level_strict_acc": bool(scores["prompt_level_strict_acc"]),
        "inst_level_strict_acc": [bool(value) for value in scores["inst_level_strict_acc"]],
        "prompt_level_loose_acc": bool(scores["prompt_level_loose_acc"]),
        "inst_level_loose_acc": [bool(value) for value in scores["inst_level_loose_acc"]],
    }
    return result


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_track.setdefault(sample["track"], []).append(sample)

    summary = {
        "examples": len(samples),
        "errored_examples": sum(1 for sample in samples if not sample["ok"]),
        "tracks": {},
    }
    for track, track_samples in sorted(by_track.items()):
        strict_inst = [
            value
            for sample in track_samples
            for value in sample["metrics"]["inst_level_strict_acc"]
        ]
        loose_inst = [
            value
            for sample in track_samples
            for value in sample["metrics"]["inst_level_loose_acc"]
        ]
        latencies = [sample["latency_sec"] for sample in track_samples if sample["latency_sec"] is not None]
        finish_reasons: dict[str, int] = {}
        for sample in track_samples:
            finish_reason = sample.get("finish_reason") or "none"
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1

        summary["tracks"][track] = {
            "examples": len(track_samples),
            "errored_examples": sum(1 for sample in track_samples if not sample["ok"]),
            "prompt_level_strict_acc": mean_bool(
                [sample["metrics"]["prompt_level_strict_acc"] for sample in track_samples]
            ),
            "inst_level_strict_acc": mean_bool(strict_inst),
            "prompt_level_loose_acc": mean_bool(
                [sample["metrics"]["prompt_level_loose_acc"] for sample in track_samples]
            ),
            "inst_level_loose_acc": mean_bool(loose_inst),
            "avg_latency_sec": statistics.fmean(latencies) if latencies else None,
            "total_latency_sec": sum(latencies),
            "avg_response_chars": statistics.fmean(len(sample["response"]) for sample in track_samples),
            "finish_reason_counts": finish_reasons,
        }

    all_strict_inst = [
        value for sample in samples for value in sample["metrics"]["inst_level_strict_acc"]
    ]
    all_loose_inst = [
        value for sample in samples for value in sample["metrics"]["inst_level_loose_acc"]
    ]
    all_latencies = [sample["latency_sec"] for sample in samples if sample["latency_sec"] is not None]
    summary.update(
        {
            "prompt_level_strict_acc": mean_bool(
                [sample["metrics"]["prompt_level_strict_acc"] for sample in samples]
            ),
            "inst_level_strict_acc": mean_bool(all_strict_inst),
            "prompt_level_loose_acc": mean_bool(
                [sample["metrics"]["prompt_level_loose_acc"] for sample in samples]
            ),
            "inst_level_loose_acc": mean_bool(all_loose_inst),
            "avg_latency_sec": statistics.fmean(all_latencies) if all_latencies else None,
            "total_latency_sec": sum(all_latencies),
        }
    )
    return summary


def run_model(
    *,
    model: str,
    docs: list[EvalDoc],
    base_url: str,
    token: str,
    args: argparse.Namespace,
    output_dir: Path,
    task_utils: dict[str, Any],
) -> dict[str, Any]:
    model_dir = output_dir / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    samples_path = model_dir / "samples.jsonl"
    progress_path = model_dir / "progress.json"
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"running {model} on {len(docs)} Nepali examples", flush=True)

    samples: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                doc=doc,
                model=model,
                base_url=base_url,
                token=token,
                args=args,
                task_utils=task_utils,
            )
            for doc in docs
        ]
        with samples_path.open("w", encoding="utf-8") as samples_handle:
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                sample = future.result()
                samples.append(sample)
                append_jsonl(samples_handle, sample)
                if completed % 25 == 0 or completed == len(futures):
                    progress = {
                        "model": model,
                        "completed_examples": completed,
                        "total_examples": len(futures),
                        "errored_examples": sum(1 for item in samples if not item["ok"]),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    write_json(progress_path, progress)
                    print(f"  {model}: {completed}/{len(futures)}", flush=True)

    samples.sort(key=lambda sample: (sample["track"], sample["dataset_index"]))
    summary = summarize_samples(samples)
    summary.update(
        {
            "model": model,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "generation": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
        }
    )
    write_jsonl(samples_path, samples)
    write_json(model_dir / "summary.json", summary)
    return summary


def write_markdown_summary(path: Path, run_summary: dict[str, Any]) -> None:
    lines = [
        "# HimalayaGPT Nepali IndicIFEval Results",
        "",
        f"- Run ID: `{run_summary['run_id']}`",
        f"- API base URL: `{run_summary['api_base_url']}`",
        f"- Examples per model: `{run_summary['examples_per_model']}`",
        f"- Generated at: `{run_summary['completed_at']}`",
        "",
        "| Model | Prompt Strict | Inst Strict | Prompt Loose | Inst Loose | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in run_summary["models"]:
        lines.append(
            "| {model} | {ps:.4f} | {is_:.4f} | {pl:.4f} | {il:.4f} | {errors} |".format(
                model=model["model"],
                ps=model["prompt_level_strict_acc"] or 0.0,
                is_=model["inst_level_strict_acc"] or 0.0,
                pl=model["prompt_level_loose_acc"] or 0.0,
                il=model["inst_level_loose_acc"] or 0.0,
                errors=model["errored_examples"],
            )
        )
    lines.append("")
    lines.append("The bearer token is not stored in this repository; reruns require `NEPEVAL_API_TOKEN` or `OPENAI_API_KEY`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    token = require_token(args)
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    models_payload = list_models(args.api_base_url, token, args.timeout, args.retries)
    discovered_model_ids = sorted(str(model["id"]) for model in models_payload if model.get("id"))
    model_ids = args.models or discovered_model_ids
    if not model_ids:
        raise SystemExit("No models found.")

    docs = load_docs(args.limit)
    task_utils = load_task_utils()

    run_config = {
        "run_id": output_dir.name,
        "api_base_url": args.api_base_url,
        "models": model_ids,
        "discovered_models": models_payload,
        "tasks": [track for track, _path, _utils_path in TASKS],
        "examples_per_model": len(docs),
        "limit_per_track": args.limit,
        "generation": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "run_config.json", run_config)

    summaries = []
    for model in model_ids:
        summaries.append(
            run_model(
                model=model,
                docs=docs,
                base_url=args.api_base_url,
                token=token,
                args=args,
                output_dir=output_dir,
                task_utils=task_utils,
            )
        )

    run_summary = {
        **run_config,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "models": summaries,
    }
    write_json(output_dir / "summary.json", run_summary)
    write_markdown_summary(output_dir / "summary.md", run_summary)
    print(f"wrote benchmark artifacts -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
