# nepeval

Nepali-only setup for running [AI4Bharat IndicIFEval](https://huggingface.co/datasets/ai4bharat/IndicIFEval) against a local model.

The project goal is to maintain reproducible Nepali AI evaluations: Nepali-only benchmark data, executable evaluation code, raw model outputs, and aggregate benchmark stats are all kept in this repo.

This repo is intentionally scoped to Nepali (`ne`) only. The upstream dataset card lists two relevant tracks:

- `indicifeval-ground`, split `ne`: 341 rows
- `indicifeval-trans`, split `ne`: 490 rows

No Git LFS is configured because the upstream dataset and the Nepali JSONL materialization are both only a few MB.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[eval]'
```

If you want to run with vLLM, also install the vLLM extra supported by your platform:

```bash
python -m pip install 'lm-eval[vllm]'
```

## Materialize Nepali Data

```bash
python scripts/materialize_nepali_indicifeval.py
```

This writes:

- `data/indicifeval_ne/indicifeval-ground/ne.jsonl`
- `data/indicifeval_ne/indicifeval-trans/ne.jsonl`
- `data/indicifeval_ne/all.jsonl`
- `data/indicifeval_ne/manifest.json`

The script has no language flag on purpose; it always downloads only split `ne`.

## Run A Local Model

Using vLLM through `lm-evaluation-harness`:

```bash
./scripts/run_nepali_indicifeval_lm_eval.sh /path/to/local/model
```

Equivalent direct command:

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=/path/to/local/model,max_model_len=8192,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks indicifeval_ground_ne,indicifeval_trans_ne \
  --gen_kwargs "temperature=0,do_sample=false,max_gen_toks=1280" \
  --batch_size auto \
  --output_path results/my-model \
  --log_samples \
  --num_fewshot 0 \
  --apply_chat_template \
  --confirm_run_unsafe_code
```

The only task YAMLs present in `lm_eval_tasks` are the Nepali tasks, so the wrapper cannot accidentally select Hindi, Bengali, Marathi, or other Indic language splits unless new task files are added later.

## Run HimalayaGPT API Models

The OpenAI-compatible runner discovers every model from `https://himalayagpt.api.scalabs.ai/v1/models`, runs both Nepali benchmark tracks against each model, and stores raw samples plus aggregate stats under `results/himalayagpt/<run-id>/`.

Keep the bearer token in the environment:

```bash
export NEPEVAL_API_TOKEN='...'
python scripts/run_openai_compatible_api_benchmarks.py
```

The runner defaults to 8 concurrent requests. Override with `--concurrency N` if the API needs a lower or higher request rate.

For the current hosted API, the model discovery endpoint returns:

- `himalaya-bf16`
- `himalaya-q8`
- `himalaya-q4`

Useful smoke test:

```bash
NEPEVAL_API_TOKEN='...' python scripts/run_openai_compatible_api_benchmarks.py --limit 2 --output-dir results/himalayagpt/smoke
```

Each run writes:

- `run_config.json`: model list, generation settings, data/task metadata.
- `summary.json`: aggregate stats for every model.
- `summary.md`: compact Markdown table for review.
- `models/<model>/summary.json`: per-model aggregate stats.
- `models/<model>/samples.jsonl`: every prompt, response, score, latency, finish reason, and raw API response. This file is streamed while the run is active.
- `models/<model>/progress.json`: live per-model completion count while the run is active.

The persisted metrics are:

- `prompt_level_strict_acc`
- `inst_level_strict_acc`
- `prompt_level_loose_acc`
- `inst_level_loose_acc`

The runner uses the same vendored Nepali instruction checkers as the lm-eval task configs.

## Attribution

This project uses and adapts the Nepali (`ne`) splits from:

- Original dataset: [ai4bharat/IndicIFEval](https://huggingface.co/datasets/ai4bharat/IndicIFEval)
- Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Upstream benchmark code: [AI4Bharat/IndicIFEval](https://github.com/AI4Bharat/IndicIFEval)
- Paper: [IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages](https://arxiv.org/abs/2602.22125)

Original authors: Thanmay Jayakumar, Mohammed Safi Ur Rahman Khan, Raj Dabre, Ratish Puduppully, and Anoop Kunchukuttan.

Local changes in this repo:

- Filters/materializes only the Nepali (`ne`) split from `indicifeval-ground` and `indicifeval-trans`.
- Converts the Nepali parquet rows to JSONL for local use.
- Vendors only the Nepali `lm-evaluation-harness` task configs and instruction checkers.

Please cite the original work when using these evaluation data:

```bibtex
@article{jayakumar2026indicifeval,
  title={IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages},
  author={Thanmay Jayakumar and Mohammed Safi Ur Rahman Khan and Raj Dabre and Ratish Puduppully and Anoop Kunchukuttan},
  year={2026},
  eprint={2602.22125},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.22125},
}
```

Vendored checker code keeps the upstream Apache 2.0 headers inherited from the original IFEval evaluation code.
