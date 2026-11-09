#!/usr/bin/env bash
# Measure step time for a list of pipeline layouts on one node (HF Jobs).
#
# LAYOUTS is a ";"-separated list. Each entry is a Megatron layout string, or
# "default" for no --pipeline-model-parallel-layout (Megatron's own split). A
# "+prof" suffix adds --timing-log-level 2 --timing-log-option all; those runs feed
# the cost model and are never used as KPI runs (level-2 timers synchronize).
#
# Recipe: the MoE handbook's 7B-A1B flag groups (Megatron d6316eb45) with
#   PAR  : PP=$PP, EP=$EP, TP=1 (proxy: PP4, EP1 on 4x A100 80GB)
#   MOE  : --num-experts $NUM_EXPERTS (proxy 32: same per-token compute as the
#          handbook's 64, top-8, but fits a pipeline stage without EP)
#   OPT  : --micro-batch-size $MBS --global-batch-size $GBS (proxy 1 and 32)
#   MTP  : --mtp-num-layers $MTP
# Everything else is the handbook's groups unchanged. No --save: no checkpoint.
#
# Results (train.log, meta.json per layout, plan.json) go to the private HF
# dataset HF_REPO under runs/$RUN_TAG after each layout and on exit.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
MEGATRON_PATH=${MEGATRON_PATH:-/work/Megatron-LM}
DATA_ROOT=${DATA_ROOT:-/data}
RUN_ROOT=${RUN_ROOT:-/runs}
HF_REPO=${HF_REPO:-antonellof/moe-layout-runs}
DATA_REPO=${DATA_REPO:-antonellof/moe-copilot-day1}
TOKENIZER_REPO=${TOKENIZER_REPO:-unsloth/Llama-3.2-1B}
NUM_DOCS=${NUM_DOCS:-5000}
IMAGE=${IMAGE:-unknown}
PP=${PP:-4}; EP=${EP:-1}; NUM_EXPERTS=${NUM_EXPERTS:-32}; MTP=${MTP:-1}
MBS=${MBS:-1}; GBS=${GBS:-32}; ITERS=${ITERS:-20}; WARMUP=${WARMUP:-5}
NPROC=${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}
CONFIG=${CONFIG:-$REPO_DIR/configs/proxy_a100x4_pp4_mtp1.json}
RUN_TAG=${RUN_TAG:-layouts-$(date -u +%Y%m%dT%H%M%SZ)}
LAYOUTS=${LAYOUTS:?set LAYOUTS, e.g. "default;Et*4|t*5|t*5|t*2mL"}

mkdir -p "$DATA_ROOT" "$RUN_ROOT"
export PYTHONPATH="$MEGATRON_PATH:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0

upload_runs() {
  [[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set, skipping upload" >&2; return 0; }
  hf upload "$HF_REPO" "$RUN_ROOT" "runs/$RUN_TAG" --repo-type dataset --private \
    --commit-message "${1:-runs $RUN_TAG}" >/dev/null \
    && echo "uploaded $RUN_ROOT to datasets/$HF_REPO/runs/$RUN_TAG" || echo "upload failed" >&2
}
trap 'upload_runs "runs $RUN_TAG (exit)"' EXIT

# ------------------------------------------------------------------ tokenizer + data
TOKENIZER_DIR="$DATA_ROOT/tokenizer"
[[ -f "$TOKENIZER_DIR/tokenizer.json" ]] || hf download "$TOKENIZER_REPO" \
  --include 'tokenizer*' --include 'special_tokens_map.json' --include 'config.json' --local-dir "$TOKENIZER_DIR"
PREFIX_NAME="fineweb-edu_${NUM_DOCS}"
if [[ ! -f "$DATA_ROOT/${PREFIX_NAME}_text_document.bin" ]]; then
  hf download "$DATA_REPO" --repo-type dataset \
    --include "data/${PREFIX_NAME}_text_document.bin" --include "data/${PREFIX_NAME}_text_document.idx" \
    --local-dir "$DATA_ROOT/hub" >/dev/null
  mv "$DATA_ROOT/hub/data/${PREFIX_NAME}_text_document."{bin,idx} "$DATA_ROOT/"
fi
DATA_PATH="$DATA_ROOT/${PREFIX_NAME}_text_document"

# ------------------------------------------------------------------ flag groups
PAR=(--tensor-model-parallel-size 1 --pipeline-model-parallel-size "$PP"
     --context-parallel-size 1 --expert-model-parallel-size "$EP" --expert-tensor-parallel-size 1)
MOE=(--num-experts "$NUM_EXPERTS" --moe-ffn-hidden-size 1024 --moe-router-topk 8
     --moe-token-dispatcher-type alltoall --moe-grouped-gemm
     --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3
     --moe-router-dtype fp32 --moe-per-layer-logging)
MODEL=(--use-mcore-models --use-flash-attn --disable-bias-linear
       --transformer-impl transformer_engine
       --position-embedding-type rope --rotary-percent 1.0 --rotary-base 500000
       --rotary-seq-len-interpolation-factor 1 --normalization RMSNorm --swiglu --norm-epsilon 1e-5
       --num-layers 16 --hidden-size 2048 --ffn-hidden-size 8192
       --num-attention-heads 32 --group-query-attention --num-query-groups 4
       --kv-channels 64 --qk-layernorm --seq-length 4096 --max-position-embeddings 4096
       --attention-dropout 0 --hidden-dropout 0 --init-method-std 0.02)
MTPF=(--mtp-num-layers "$MTP")
DATA=(--tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_DIR"
      --data-path "$DATA_PATH" --data-cache-path "$DATA_ROOT/cache"
      --split 100,0,0 --no-mmap-bin-files --num-workers 6 --dataloader-type cyclic
      --reset-attention-mask --reset-position-ids)
OPT=(--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
     --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-samples 27732145
     --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1 --weight-decay 0.1
     --lr 5e-4 --min-lr 0 --lr-warmup-samples 2000 --lr-decay-samples 27732145
     --lr-decay-style WSD --lr-wsd-decay-style linear --lr-wsd-decay-samples 732145
     --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
     --cross-entropy-loss-fusion --cross-entropy-fusion-impl native --enable-experimental)
LOG=(--log-interval 1 --log-throughput --log-num-zeros-in-grad --log-memory-interval 5
     --eval-interval 100000 --eval-iters 0 --manual-gc --manual-gc-interval 5
     --distributed-timeout-minutes 60 --seed 1234 --bf16 --exit-interval "$ITERS")

MEGATRON_COMMIT=$(git -C "$MEGATRON_PATH" rev-parse HEAD 2>/dev/null || echo unknown)
GPUS=$(nvidia-smi -L 2>/dev/null | tr '\n' ';' || echo unknown)

# ------------------------------------------------------------------ layouts
# Entries run in order. A "plan:N" entry is deferred: after the profiling runs are
# fitted, the planner's top N layouts are measured (closed loop in one job).
RANKS_PER_STAGE=${RANKS_PER_STAGE:-$(( NPROC / PP ))}   # ranks pp*k .. pp*k+k-1 form stage pp
PLAN_ARGS=${PLAN_ARGS:---exclude-standalone-mtp}
idx=0; prof_args=(); plan_n=0

run_layout() {  # $1 layout ("default" or a layout string), $2 prof (0/1), $3 name suffix
  local lay=$1 prof=$2 suffix=${3:-}
  idx=$((idx + 1))
  local name
  name=$(printf '%02d_%s%s%s' "$idx" "$(echo "$lay" | tr '|*' '-x')" "$([[ $prof == 1 ]] && echo _prof)" "$suffix")
  local RUN_DIR="$RUN_ROOT/$name"; mkdir -p "$RUN_DIR"
  [[ -f "$RUN_DIR/done" ]] && { echo "skip $name"; return 0; }
  local EXTRA=()
  [[ "$lay" != default ]] && EXTRA+=(--pipeline-model-parallel-layout "$lay")
  [[ $prof == 1 ]] && EXTRA+=(--timing-log-level 2 --timing-log-option all)
  local CMD=(python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC"
       "$MEGATRON_PATH/pretrain_gpt.py"
       "${PAR[@]}" "${MODEL[@]}" "${MTPF[@]}" "${DATA[@]}" "${OPT[@]}" "${MOE[@]}" "${LOG[@]}" "${EXTRA[@]}")
  echo "=== $name ($RUN_TAG) layout=$lay prof=$prof ==="
  local s0; s0=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log" || echo "layout $name FAILED" | tee "$RUN_DIR/failed"
  python - "$RUN_DIR/meta.json" "$lay" "$prof" "$RUN_TAG" "$IMAGE" "$MEGATRON_COMMIT" "$GPUS" \
    "$(( $(date +%s) - s0 ))" "$suffix" "${CMD[@]}" <<'PY'
import json, sys
out, lay, prof, tag, image, commit, gpus, wall, suffix, *cmd = sys.argv[1:]
json.dump({"layout": lay, "profiling": prof == "1", "run_tag": tag, "image": image,
           "megatron_commit": commit, "gpus": gpus, "wall_s": int(wall),
           "source": suffix.lstrip("_") or "listed", "cmd": cmd}, open(out, "w"), indent=2)
PY
  touch "$RUN_DIR/done"
  if [[ $prof == 1 && ! -f "$RUN_DIR/failed" ]]; then prof_args+=(--run "$lay=$RUN_DIR/train.log"); fi
  python -m moe_layout.report "$RUN_ROOT" --warmup "$WARMUP" > "$RUN_ROOT/summary.json" || true
  upload_runs "runs $RUN_TAG: $name done"
}

fit_and_plan() {  # fit the profiling runs so far, write costs.json and plan.json
  (( ${#prof_args[@]} >= 2 )) || { echo "fit skipped: need 2 profiling runs" >&2; return 0; }
  python -m moe_layout fit "$CONFIG" "${prof_args[@]}" --warmup "$WARMUP" \
    --ranks-per-stage "$RANKS_PER_STAGE" -o "$RUN_ROOT/costs.json" > "$RUN_ROOT/fit.json" || return 0
  # shellcheck disable=SC2086
  python -m moe_layout plan "$CONFIG" --pp "$PP" --costs "$RUN_ROOT/costs.json" \
    --hbm-gb "${HBM_GB:-76}" --top 8 $PLAN_ARGS > "$RUN_ROOT/plan.json" || true
  upload_runs "runs $RUN_TAG: fit and plan"
}

baseline_args=()   # timer-free listed runs, used to recalibrate the overhead at the end
IFS=';' read -r -a ENTRIES <<< "$LAYOUTS"
for entry in "${ENTRIES[@]}"; do
  entry=$(echo "$entry" | xargs)
  [[ -n "$entry" ]] || continue
  if [[ "$entry" == plan:* ]]; then
    # closed loop, in place: fit what has been profiled so far, then measure the top N
    fit_and_plan
    if [[ -s "$RUN_ROOT/plan.json" ]]; then
      mapfile -t PICKS < <(python -c 'import json, sys
top = json.load(open(sys.argv[1]))["top"][: int(sys.argv[2])]
print("\n".join(c["layout"] for c in top))' "$RUN_ROOT/plan.json" "${entry#plan:}")
      for k in "${!PICKS[@]}"; do run_layout "${PICKS[$k]}" 0 "_plan$((k + 1))"; done
    fi
    continue
  fi
  lay=${entry%+prof}; prof=0; [[ "$entry" == *+prof ]] && prof=1
  run_layout "$lay" "$prof"
  if [[ $prof == 0 ]]; then
    last=$(ls -d "$RUN_ROOT"/[0-9][0-9]_* | tail -1)
    [[ -f "$last/failed" ]] || baseline_args+=(--run "$lay=$last/train.log")
  fi
done
[[ -f "$RUN_ROOT/costs.json" ]] || fit_and_plan

# ------------------------------------------------------------------ report
# Overhead recalibrated on the timer-free listed baselines only, so the planner's picks
# stay out-of-sample; predictions in summary.json use costs_calibrated.json.
REPORT=(python -m moe_layout.report "$RUN_ROOT" --warmup "$WARMUP")
if [[ -f "$RUN_ROOT/costs.json" ]]; then
  costs_for_report="$RUN_ROOT/costs.json"
  if (( ${#baseline_args[@]} >= 2 )); then
    python -m moe_layout calibrate "$CONFIG" --pp "$PP" --costs "$RUN_ROOT/costs.json" \
      "${baseline_args[@]}" --warmup "$WARMUP" -o "$RUN_ROOT/costs_calibrated.json" \
      > "$RUN_ROOT/calibrate.json" && costs_for_report="$RUN_ROOT/costs_calibrated.json"
  fi
  REPORT+=(--config "$CONFIG" --costs "$costs_for_report" --pp "$PP")
fi
"${REPORT[@]}" | tee "$RUN_ROOT/summary.json"
