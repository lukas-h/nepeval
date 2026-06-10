#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/local/model [extra lm_eval args...]" >&2
  exit 2
fi

MODEL_PATH="$1"
shift

MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=${MODEL_PATH},max_model_len=8192,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True}"
TASKS="${TASKS:-indicifeval_ground_ne,indicifeval_trans_ne}"
OUTPUT_PATH="${OUTPUT_PATH:-results/$(basename "${MODEL_PATH}")}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0,do_sample=false,max_gen_toks=1280}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

lm_eval \
  --model "${MODEL_BACKEND}" \
  --model_args "${MODEL_ARGS}" \
  --include_path lm_eval_tasks \
  --tasks "${TASKS}" \
  --gen_kwargs "${GEN_KWARGS}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  --num_fewshot 0 \
  --apply_chat_template \
  --confirm_run_unsafe_code \
  "$@"

