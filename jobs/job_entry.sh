#!/usr/bin/env bash
# HF Jobs entry on the stock NGC image (nvcr.io/nvidia/pytorch:26.06-py3):
# pip extras, Megatron-LM at the pin, this repo, then jobs/run_layouts.sh.
#   hf jobs run -d --flavor a100x4 --timeout 50m --secrets HF_TOKEN \
#     -e LAYOUTS="default;Et*4|t*5|t*5|t*2mL" -e IMAGE=nvcr.io/nvidia/pytorch:26.06-py3 \
#     nvcr.io/nvidia/pytorch:26.06-py3 bash -c \
#     'git clone -q https://github.com/antonellof/moe-layout /work/moe-layout && bash /work/moe-layout/jobs/job_entry.sh'
set -euo pipefail
CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MEGATRON_PATH=${MEGATRON_PATH:-/work/Megatron-LM}
PIN=d6316eb45bf1bfbb6f1f848c0b56e55166564d80
echo "host: $(hostname)  nproc: $(nproc)  date: $(date -u +%FT%TZ)  code: $(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi -L 2>/dev/null || echo "no GPUs visible"
PKGS=("huggingface_hub>=1.0" transformers sentencepiece tiktoken omegaconf tensorboard)
pip install --no-cache-dir -q "${PKGS[@]}" || PIP_CONSTRAINT= pip install --no-cache-dir -q "${PKGS[@]}"
if [[ ! -d "$MEGATRON_PATH/.git" ]]; then
  mkdir -p "$MEGATRON_PATH"
  git -C "$MEGATRON_PATH" init -q
  git -C "$MEGATRON_PATH" remote add origin https://github.com/NVIDIA/Megatron-LM.git
  git -C "$MEGATRON_PATH" fetch -q --depth 1 origin "$PIN"
  git -C "$MEGATRON_PATH" checkout -q FETCH_HEAD
fi
echo "megatron: $(git -C "$MEGATRON_PATH" rev-parse HEAD)"
exec bash "$CODE_DIR/jobs/run_layouts.sh"
