# moe-layout: technical report

A measured pipeline-layout planner for Megatron-LM MoE models with multi-token
prediction. November 2026. Code: this repository. Raw logs: private HF dataset
`antonellof/moe-layout-runs` (available on request); every number below is
reproducible from the files in [results/](../results) and the commands at the end.

## 1. Summary

Megatron-LM splits a model across pipeline stages with
`--pipeline-model-parallel-layout`, a string you write by hand. For MoE models with an
MTP layer, Megatron's default split leaves the last stage much heavier than the rest. The
output layer and the MTP layer both sit there, and the slowest stage sets the speed of
every step.

moe-layout fits the cost of each unit (embedding, decoder layer, MTP layer, loss) from
Megatron's own per-GPU timers on two short profiling runs. It then simulates Megatron's
1F1B schedule for every layout Megatron's validator accepts and prints the fastest one
that fits in memory.

| Hardware | Model | Planner layout | vs Megatron default | vs count-based rule | Prediction error (pick) |
| --- | --- | --- | --- | --- | --- |
| 8x H200 | handbook 7B-A1B, 64 experts, EP2, PP4, MTP1, micro-batch 4 | `Et*5\|t*6\|t*5\|mL` | **1.39x** | **1.12x** | 2.7% |
| 4x A100 80GB | same layers, 32 experts, EP1, PP4, MTP1, micro-batch 1 | `Et*5\|t*6\|t*5\|mL` | **1.54x** | **1.26x** | 2.7% |

Total GPU cost of the study: about $16 on Hugging Face Jobs.

## 2. Background

- **Pipeline parallelism.** Consecutive layers form stages on different GPUs.
  Micro-batches flow through them under the 1F1B schedule. With equal stages a step takes
  about `(micro-batches + stages - 1) x stage time`. With unequal stages, everything waits
  for the slowest.
- **Layout string** ([Megatron docs](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/docs/user-guide/features/pipeline_parallel_layout.md)).
  `E` embedding, `t` decoder layer, `m` MTP layer, `L` loss including the output layer,
  `|` between stages. Example: DeepSeek-V3 on PP16 with 2 virtual stages is
  `Et*3|(tt|)*29,m|L`.
- **MTP** ([DeepSeek-V3](https://arxiv.org/abs/2412.19437),
  [Megatron docs](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/docs/user-guide/features/multi_token_prediction.md)).
  An extra module that predicts a further token, with its own transformer block and a
  pass through the shared output layer. Without a custom layout Megatron places all MTP
  layers on the last pipeline stage.
- **The MoE Training Handbook** ([Hugging Face, Nov 2026](https://huggingface.co/spaces/huggingface/moe-blogpost)).
  It reports early steps about 37 percent slower with one MTP layer at 256 nodes, and an
  OOM with 53 layers on 4 stages. Its 7B-A1B recipe is the model used here, with Megatron
  pinned at `d6316eb45`.
- **Ling 2.0** ([arXiv 2510.22115](https://arxiv.org/abs/2510.22115), section 5.2). An MTP
  layer costs about 1.7x a MoE layer at its scale. Refining the split by hand gave a
  40 percent end-to-end gain.

## 3. Method

### 3.1 Units and layouts

A model is the ordered sequence `[E, t*N, m*K, L]`. A layout is a split of that sequence
into `PP x VPP` contiguous stages. The validator at the pin requires:
- `E` first and `L` last, once each;
- all `m` after every `t`, in one virtual stage;
- MTP in the last virtual stage of its rank, and not on rank 0.

`moe_layout.layout` mirrors these rules exactly. It enumerates 816 valid layouts for 16
layers, 1 MTP layer and PP4.

### 3.2 Cost model

Per micro-batch, stage `s` costs `fwd_s = sum_u n_su * cf_u + cf_0` and
`bwd_s = sum_u n_su * cb_u + cb_0`. Here `n_su` counts units of type `u` in the stage,
and `cf_0`, `cb_0` are per-stage constants. The observations are Megatron's per-rank
`forward-compute` and `backward-compute` timers (`--timing-log-level 2
--timing-log-option all`), divided by the number of micro-batches. They are fitted by
least squares over every stage of the profiling runs. A weak ridge prior toward FLOP
ratios keeps any unit the data cannot identify near its FLOP-based value.

The two profiling layouts are chosen so that every unit is identifiable:
`Et*4|t*4|t*4|t*4mL` (Megatron's default) and `Et*7|t*5|t*4m|L`. The second varies the
decoder counts and puts MTP on its own stage, which separates `m` from `L`.

### 3.3 Schedule simulator and search

`moe_layout.simulate` replays Megatron's non-interleaved 1F1B order for each stage:
- warmup of `PP - s - 1` forwards;
- steady F, B pairs;
- cooldown backwards.

Each forward waits for the previous stage's forward plus a p2p hop, and each backward
for the next stage's backward. Predicted step time is the makespan plus a per-step
overhead (optimizer, gradient sync). Every valid layout is scored, layouts over the HBM
limit are dropped, and ties are broken by:
1. fewer near-bottleneck stages;
2. lower peak memory;
3. evenness.

### 3.4 Overhead calibration

Level-2 timers synchronize the GPU and add about 4 percent to step time, so the overhead
measured on profiling runs is inflated. From stage C on, the overhead is recalibrated on
timer-free baseline runs, and the planner's own picks are never used for this, so their
predictions stay out of sample. Stages A and B used the profiling overhead, which is
why their predictions lean slightly high.

### 3.5 Memory model

Per stage: bf16 weights, fp32 gradients and Adam state sharded by the distributed
optimizer, plus logits for each output head on the stage (`L` and `m`). The
logit term (10.3 bytes per logit per head) and a 0.13 GB constant were calibrated
against Megatron's `max allocated` on 40 measured stages (stages A and B). The mean
error is 2.8 percent and the max 13.7 percent.

## 4. Setup

| | Stages A and B (proxy) | Stage C (handbook scale) |
| --- | --- | --- |
| Hardware | HF Jobs `a100x4`, 4x A100 80GB | HF Jobs `h200x8`, 8x H200 141GB |
| Parallelism | PP4, EP1, DP1, TP1 | PP4, EP2, DP2, TP1 |
| Experts | 32 (same per-token compute as 64 at top-8; a stage fits without EP) | 64 |
| Batch | micro-batch 1, global 32 (32 micro-batches) | micro-batch 4, global 128 (16 micro-batches) |
| Iterations | 20, median of 6 to 20 | 15, median of 5 to 15 |
| Config | `configs/proxy_a100x4_pp4_mtp1.json` | `configs/handbook_h200x8_pp4_ep2_mtp1.json` |

Both setups use:
- the handbook's 7B-A1B flag groups otherwise unchanged: 16 layers, hidden 2048, top-8,
  full recompute, BF16, aux loss 1e-3;
- one MTP layer, 4096-token sequences, the Llama 3.2 tokenizer and FineWeb-Edu data;
- image `nvcr.io/nvidia/pytorch:26.06-py3` (torch 2.13.0a0, CUDA 13.3, TE 2.16.0);
- Megatron-LM `d6316eb45bf1bfbb6f1f848c0b56e55166564d80`.

Every run's full command line is in `results/*/meta.json`.

## 5. Results

### 5.1 Stage A: profile and fit (4x A100, `stageA-20261109T111955Z`)

| Layout | Measured step | Predicted | Error |
| --- | --- | --- | --- |
| Megatron default `Et*4\|t*4\|t*4\|t*4mL` | 6.41 s | 6.53 s | +1.7% |
| count-based `Et*4\|t*5\|t*5\|t*2mL` (not in the fit) | 5.25 s | 5.29 s | +0.6% |
| profiling 1 (default, with timers) | 6.55 s | 6.53 s | -0.3% |
| profiling 2 `Et*7\|t*5\|t*4m\|L` (with timers) | 5.32 s | 5.20 s | -2.2% |

On the default split, per-stage time per micro-batch was 93, 84, 81 and 191 ms. The last
stage carried 2.3x the work of the middle ones, and the per-stage fit matched every stage
within about 2 ms.

### 5.2 Stage B: measure the picks (4x A100, `stageB-20261109T113743Z`)

| Layout | Role | Measured | Predicted | Error |
| --- | --- | --- | --- | --- |
| `Et*5\|t*6\|t*5\|mL` | planner's pick | **4.26 s** | 4.37 s | +2.7% |
| `Et*5\|t*5\|t*6\|mL` | planner's #2, fastest measured | 4.23 s | 4.39 s | +3.7% |
| `Et*6\|t*5\|t*5\|mL` | neighbour | 4.58 s | 4.64 s | +1.3% |
| `Et*4\|t*6\|t*6\|mL` | planner's #3, two equal bottlenecks | 4.70 s | 4.39 s | -6.4% |
| `Et*5\|t*6\|t*5m\|L` | standalone MTP | 5.28 s | 4.63 s | -12.3% |
| default | repeat on the same node | 6.54 s | 6.53 s | -0.2% |

The pick is **1.54x** faster than the default on the same node, and **1.26x** faster than
count-based. That uses count-based from stage A, adjusted by the 2 percent node
difference that the default repeat measured; the raw figure is 1.24x. The pick is
within 0.5 percent of the best layout measured.

### 5.3 Stage C: closed loop at handbook scale (8x H200, `stageC-20261109T120517Z`)

One job ran everything in this order:
1. the two profiling layouts;
2. the planner, on its own fit (`plan:1`);
3. its pick;
4. both baselines, on the same node.

| Layout | Role | Measured | Predicted | Error |
| --- | --- | --- | --- | --- |
| `Et*5\|t*6\|t*5\|mL` | planner's pick | **3.80 s** | 3.70 s | -2.7% |
| `Et*4\|t*5\|t*5\|t*2mL` | count-based | 4.27 s | 4.28 s | +0.3% |
| default | Megatron's split | 5.28 s | 5.27 s | -0.2% |
| profiling 1, with timers | | 5.37 s | 5.27 s | -1.8% |
| profiling 2, with timers | | 4.28 s | 4.18 s | -2.4% |

The pick is **1.39x** faster than the default and **1.12x** faster than count-based, on
the same node.

### 5.4 Fitted unit costs

| Unit | A100, micro-batch 1 (ms, fwd+bwd) | in decoder layers | H200, micro-batch 4, EP2 (ms) | in decoder layers | FLOP estimate |
| --- | --- | --- | --- | --- | --- |
| decoder layer `t` | 20.0 | 1.0 | 32.9 | 1.0 | 1.0 |
| loss with output layer `L` | 79.8 | 4.0 | 119.1 | 3.6 | 2.9 |
| MTP layer `m` | 28.8 | 1.44 | 47.2 | 1.43 | 4.0 |
| embedding `E` | 10.4 | 0.52 | 8.6 | 0.26 | 0 |

The ratios are close across the two GPU types. Against FLOPs, MTP measures much cheaper
(1.43 vs 4.0 decoder layers) and the output layer heavier (3.6 to 4.0 vs 2.9). One
possible reason is that MTP's pass through the output layer is computed together with
the main loss, which would move that cost from `m` to `L`. This is not verified; the
planner only relies on the measured split.

### 5.5 FLOP-only prior vs measured

Planning from FLOPs alone (at an assumed 150 TFLOP/s) would have picked
`Et*4|t*6|t*6|mL`, which measured 4.70 s in stage B. The profiled pick ran 4.26 s, 10
percent faster.

## 6. Where the model misses

- **MTP on its own stage, 12 percent slower than predicted (stage B).** The per-stage
  compute of a standalone MTP stage matched the fit in the timed profiling run, so the
  gap lies outside stage compute. Plausible causes are extra communication or
  synchronization specific to standalone MTP. This is not diagnosed yet. Such layouts are
  flagged (`mtp_standalone`) and excluded by default in the closed loop.
- **Two equally slow stages, 6 percent slower than predicted (stage B).** The simulator
  uses fixed stage times, while real stages jitter, and two critical stages compound the
  waiting. Reported as `critical_stages`, and used as the first tie-breaker.
- **Memory of the `mL` stage at micro-batch 4 (stage C).** The prediction was 52 GB
  against 38 GB measured, which errs on the safe side. The logit term was calibrated at
  micro-batch 1. Decoder stages were within about 10 percent.

## 7. Limitations

- **One node, short runs, one seed.** 15 to 20 iterations per layout, early training.
  Routing imbalance late in training differs from early training.
- **One model.** The 16-layer, hidden-2048 recipe has a heavy output layer relative to its
  decoder layers. At hidden 7168 (DeepSeek-V3 class) the output layer is relatively
  lighter, so the gain should shrink, though the method is the same.
- **No interleaved schedule.** VPP above 1 is validated, but its interleaved schedule is
  not simulated.
- **No routing variance.** Expert-parallel routing variance is not modeled, beyond what
  the profiling runs average in.
- **Smaller batch at stage C.** Global batch 128 instead of the handbook's 512, to fit
  the budget; fewer micro-batches mean a larger pipeline bubble.
- **Thin best-of-N on H200.** In stage C the pick was the fastest of 3 measured
  layouts; the wider check (6 layouts) was on A100.

## 8. Reproduce

```bash
IMG=nvcr.io/nvidia/pytorch:26.06-py3
BOOT='git clone -q https://github.com/antonellof/moe-layout /work/moe-layout && bash /work/moe-layout/jobs/job_entry.sh'
# Stage C as one closed-loop job (8x H200, about $10)
hf jobs run -d --flavor h200x8 --timeout 32m --secrets HF_TOKEN -e IMAGE=$IMG \
  -e CONFIG=/work/moe-layout/configs/handbook_h200x8_pp4_ep2_mtp1.json \
  -e PP=4 -e EP=2 -e NUM_EXPERTS=64 -e MBS=4 -e GBS=128 -e ITERS=15 -e WARMUP=4 -e HBM_GB=130 \
  -e LAYOUTS="Et*4|t*4|t*4|t*4mL+prof;Et*7|t*5|t*4m|L+prof;plan:1;default;Et*4|t*5|t*5|t*2mL" \
  $IMG bash -c "$BOOT"
# Stage A then B on 4x A100 (about $6): LAYOUTS as listed in results/stage*/meta.json
```

The runner needs:
- `HF_TOKEN`, with pre-paid HF Jobs credits;
- a dataset repo with `data/fineweb-edu_5000_text_document.{bin,idx}`, set as `DATA_REPO`
  (produced by the data prep in [moe-copilot](https://github.com/antonellof/moe-copilot)).

Logs go to `HF_REPO`. HF Jobs timeouts count from job creation, so keep a few minutes of
margin.

## 9. Cost

| Stage | Flavor | Running time | Cost |
| --- | --- | --- | --- |
| A | a100x4 ($10/h) | 14.5 min | about $2.50 |
| B | a100x4 | 19.5 min | about $3.30 |
| C | h200x8 ($40/h) | 14.4 min | about $9.60 |

## 10. Prior art

Checked on 2026-11-09:
- **Hand-written layouts.** DeepSeek-V3 and GLM-5.2 recipes, and Ling 2.0's manual
  refinement.
- **Count-based auto layouts.** Megatron Lite `_auto_layout` and Megatron-Bridge.
- **Profile-guided partitioners for dense models.** PipeDream and AdaPipe.
- **Runtime MoE layer migration.** DynMo.
- **An internal schedule simulator.** NVIDIA's GLM-5.2 tracker (Megatron-LM issue #6392)
  mentions one; it is not published.

None of these turns Megatron's own timings into the full `E/t/m/L` string. Details are in
[moe-copilot PLAN.md](https://github.com/antonellof/moe-copilot/blob/main/PLAN.md).

## 11. Next

- Simulate interleaved (VPP) schedules.
- Diagnose the standalone-MTP gap with a timed run.
- Model stage jitter.
- Test multi-node and a DeepSeek-V3-shaped model (61 layers).
- Offer the planner upstream, as a Megatron-LM or Megatron-Bridge helper.

## Lineage

This tool came out of [moe-copilot](https://github.com/antonellof/moe-copilot), which
validated other ideas on the same recipe. Its Day 1 attribution experiment passed its
pre-registered test (fragile). Its Day 2 controller pilot did not beat quantile balancing.
