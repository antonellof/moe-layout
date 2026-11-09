# moe-layout

A measured pipeline-layout planner for Megatron-LM MoE models with MTP. It reads
per-stage timings from two short profiling runs, fits a per-unit cost model, simulates
Megatron's 1F1B schedule for every layout Megatron accepts, and prints the
`--pipeline-model-parallel-layout` string with the lowest predicted step time,
together with per-stage memory. It also works as a sizer: parameters, FLOPs and memory
per unit from a model config or a Megatron command line.

## Why

With pipeline parallelism, the slowest stage sets the pace. MoE models with
multi-token prediction (MTP) make stages very uneven:

- Megatron puts every MTP layer, the output layer and the loss on the last stage
  unless you write a layout string by hand.
- The MoE Training Handbook (Hugging Face, Nov 2026) saw early steps about 37 percent
  slower with one MTP layer, and an OOM with 53 layers on 4 stages.
- Ling 2.0 (arXiv 2510.22115, section 5.2) measured an MTP layer at about 1.7x a MoE
  layer and got a 40 percent end-to-end gain by refining the split by hand.
- Published recipes hand-write the string, for example DeepSeek-V3's
  `Et*3|(tt|)*29m|L`. NVIDIA's GLM-5.2 tracker picks layouts with A/B runs.

At the handbook's width (hidden 2048, vocab 128k) the output layer alone costs about
2.9 decoder layers, and an MTP layer with its own logits about 4. On PP4 with 16 layers
Megatron's default last stage carries about 10.9 decoder layers of work against 4 on the
others.

Prior art checked on 2026-11-09: hand layouts, count-based auto layouts (Megatron
Lite's `_auto_layout`, Megatron-Bridge), profile-guided partitioners for dense models
(PipeDream, AdaPipe) and runtime MoE rebalancing (DynMo) exist. None of them reads
Megatron's own timings and emits the full `E/t/m/L` string. Details in
[moe-copilot PLAN.md](https://github.com/antonellof/moe-copilot/blob/main/PLAN.md).

## Status

| Stage | What | State |
| --- | --- | --- |
| 0 | Planner, cost model, 1F1B simulator, sizer, Megatron log parsers, tests (CPU) | done |
| A | 4x A100: does MTP with PP train at the pin; default vs count-based; 2 profiling runs (about $8) | next, needs approval |
| B | 4x A100: planner's picks measured, prediction error, best-of-N (about $8) | after A |
| C | 8x H200: handbook scale with expert parallelism (about $13) | optional |

KPI (stage B): measured step time of the planner's layout against Megatron's default
split and the count-based rule. Pass: at least 10 percent faster than count-based,
within 2 percent of the best measured layout, prediction error at most 5 percent.

What the FLOP-only prior predicts for the PP4 proxy, before any measurement (summary
of the JSON printed by
`python -m moe_layout plan configs/proxy_a100x4_pp4_mtp1.json --pp 4 --tflops 150 --hbm-gb 80`):

```
best              Et*4|t*6|t*6|mL        3.53 s
megatron_default  Et*4|t*4|t*4|t*4mL     5.38 s   (best predicted 1.52x faster)
count_based       Et*4|t*5|t*5|t*2mL     4.45 s   (best predicted 1.26x faster)
```

These are predictions from FLOPs, not results. Stages A and B measure them.

## Use

```bash
python -m moe_layout size configs/handbook_7b_a1b.json          # 6.86B total, 1.22B active
python -m moe_layout size configs/proxy_a100x4_pp4_mtp1.json --pp 4 --layout "Et*4|t*4|t*4|t*4mL"
python -m moe_layout baselines CONFIG --pp 4                     # default and count-based strings
python -m moe_layout profile-layouts CONFIG --pp 4               # two layouts that identify all costs
python -m moe_layout fit CONFIG --run "LAYOUT1=run1/train.log" --run "LAYOUT2=run2/train.log" -o costs.json
python -m moe_layout plan CONFIG --pp 4 --costs costs.json --hbm-gb 76
python -m moe_layout score CONFIG --pp 4 --costs costs.json --layout "Et*4|t*6|t*6|mL" --measured-ms 3600
python -m moe_layout.report RUN_ROOT --config CONFIG --costs costs.json
```

`CONFIG` is a JSON `ModelConfig` (see `configs/`), or a file holding a Megatron command
line with `--from-args`. Profiling runs need `--timing-log-level 2 --timing-log-option all`.
Stdlib only, Python 3.10+.

## Run the stages on Hugging Face Jobs

`jobs/job_entry.sh` runs on the stock `nvcr.io/nvidia/pytorch:26.06-py3`, clones
Megatron-LM at `d6316eb45` and this repo, then `jobs/run_layouts.sh` runs each layout in
`LAYOUTS` (";"-separated, `default` = no layout flag, `+prof` = with level-2 timers)
and uploads logs to a private HF dataset after every layout.

```bash
IMG=nvcr.io/nvidia/pytorch:26.06-py3
BOOT='git clone -q https://github.com/antonellof/moe-layout /work/moe-layout && bash /work/moe-layout/jobs/job_entry.sh'
# stage A: about $8 on a100x4 ($10/h); HF timeouts count from job creation, keep margin
hf jobs run -d --flavor a100x4 --timeout 50m --secrets HF_TOKEN -e IMAGE=$IMG \
  -e LAYOUTS="default;Et*4|t*5|t*5|t*2mL;Et*4|t*4|t*4|t*4mL+prof;Et*7|t*5|t*4m|L+prof" \
  $IMG bash -c "$BOOT"
```

The proxy recipe is the handbook's 7B-A1B flag groups with PP4, EP1, 32 experts (the
same per-token compute as the handbook's 64 at top-8, so a stage fits in 80 GB without
expert parallelism), micro-batch 1, global batch 32 and one MTP layer.

## How it works

- `layout.py` parses and emits Megatron's layout grammar and mirrors its validator at
  the pin: E first, L last, MTP after all decoders, all MTP layers in one virtual stage,
  last virtual stage of its rank, not on PP rank 0. It enumerates every valid
  contiguous split (816 for 16 layers, 1 MTP, PP4).
- `costmodel.py` fits per-micro-batch forward and backward costs for E, t, m, L and a
  per-stage constant by least squares over all stages of the profiling runs, with a
  weak ridge prior toward FLOP ratios for anything the runs cannot identify.
- `simulate.py` replays Megatron's non-interleaved 1F1B order (warmup, steady 1F1B,
  cooldown) with p2p latency.
- `sizer.py` gives parameters, FLOPs and per-stage memory (bf16 weights, fp32 grads,
  Adam state sharded by the distributed optimizer, recompute activations, logits).

Limitations: VPP > 1 is validated but not simulated yet. Routing imbalance under
expert parallelism is not modeled until stage C. The memory model is an estimate to be
calibrated against Megatron's `max allocated` print.

## License

MIT, see [LICENSE](LICENSE).
