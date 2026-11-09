# moe-layout

**Pick a balanced pipeline split for a Megatron-LM MoE model with multi-token
prediction (MTP), from two short profiling runs instead of trial and error.**

When a model is trained with pipeline parallelism, its layers are split across GPUs,
and the slowest GPU sets the speed of every step. Megatron-LM lets you control that
split with one flag, `--pipeline-model-parallel-layout`, but you have to write the
string by hand. For MoE models with MTP the default split is badly uneven. moe-layout
measures what each part of the model really costs, simulates Megatron's pipeline
schedule for every split Megatron accepts, and prints the layout string with the
lowest predicted step time, plus the memory each GPU will need.

**Results.** The planner picks its layout from two 20-iteration profiling runs on a
4-stage pipeline with one MTP layer. Its layout ran:
- on 8x H200 at the handbook's scale (64 experts, expert parallelism 2): **1.39x faster
  than Megatron's default split** and **1.12x faster than the count-based auto layout**,
  with step time predicted within 2.7 percent;
- on a 4x A100 proxy: 1.54x and 1.26x, within 0.5 percent of the best of 6 measured
  layouts.

About $16 of GPU time in total. Details and caveats are in [Status](#status); the full
write-up (method, setup, all numbers, limitations, how to reproduce) is in
[docs/REPORT.md](docs/REPORT.md). Blog post:
[One Stage Sets the Pace](https://www.fratepietro.com/2026/moe-layout-megatron-pipeline-planner-mtp/).
Proposed upstream in [NVIDIA/Megatron-LM#9197](https://github.com/NVIDIA/Megatron-LM/issues/9197).

It also works as a **sizer**: parameters, FLOPs and memory per layer type, from a
model config or a Megatron command line.

## Background

New to some of this? Here is what you need to know, with links.

### The MoE Training Handbook

[The Ultimate MoE Training Handbook](https://huggingface.co/spaces/huggingface/moe-blogpost)
(Hugging Face, November 2026) is a long, hands-on guide to training Mixture-of-Experts
models with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). It covers hardware
checks, a first MoE run, routing and load balancing, experts, stability and systems
tuning. Its scripts are in a
[companion gist](https://gist.github.com/nmntz/5de952d686f66d7bce5bce8cba9d7139), and
everything is pinned to Megatron commit `d6316eb45`, which this repo uses too.

Its reference model is a "7B-A1B" MoE: 16 layers, hidden size 2048, 64 experts with 8
active per token, about 6.86B total and 1.22B active parameters. Three of its
observations motivate this tool:

- **MTP slows training at scale.** With one MTP layer, early steps ran about 37 percent
  slower in the authors' 256-node runs, and spreading over 8 pipeline stages did not
  close the gap.
- **Megatron puts all MTP layers on the last pipeline stage** unless you pass a
  custom layout.
- **Memory fails on the busiest stage.** They hit an out-of-memory error with 53
  layers on 4 pipeline stages.

This repo grew out of [moe-copilot](https://github.com/antonellof/moe-copilot), a
validation of other tooling ideas on the same recipe.

### Pipeline parallelism in one paragraph

Pipeline parallelism (PP) cuts the model into consecutive **stages**, one per GPU (or
group of GPUs). A batch is split into **micro-batches** that flow through the stages
like an assembly line. Megatron's default schedule, **1F1B** (one forward, one
backward), interleaves forward and backward passes so each stage stays busy. With
equal stages a step takes about `(micro-batches + stages - 1) x stage time`. With
unequal stages, every stage waits for the slowest one. A stage that does 2x the work
makes the whole step about 2x slower, and the other GPUs sit idle part of the time.

### The layout flag

`--pipeline-model-parallel-layout` describes which layers each stage holds
([Megatron docs](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/docs/user-guide/features/pipeline_parallel_layout.md)):

| Symbol | Meaning |
| --- | --- |
| `E` | embedding layer |
| `t` | transformer decoder layer |
| `m` | MTP layer |
| `L` | loss (includes the output layer that turns hidden states into logits) |
| `\|` | boundary between stages |
| `t*4`, `(tt\|)*29` | repetition |

The Megatron docs' DeepSeek-V3 example, 61 decoder layers plus one MTP layer on
PP16 with 2 virtual stages per GPU, is `Et*3|(tt|)*29,m|L`.

### Multi-token prediction (MTP)

MTP, popularized by [DeepSeek-V3](https://arxiv.org/abs/2412.19437), trains the model
to also predict tokens further ahead. Each MTP module has a projection, a full
transformer block and its own pass through the shared output head, so it is much
heavier than one decoder layer. Megatron enables it with `--mtp-num-layers`
([Megatron docs](https://github.com/NVIDIA/Megatron-LM/blob/d6316eb45bf1bfbb6f1f848c0b56e55166564d80/docs/user-guide/features/multi_token_prediction.md)).
The MTP loss is weighted 0.1 by default.

### Why the default split is lopsided

The output layer multiplies every token by a `hidden x vocabulary` matrix (2048 x
128,256 in the handbook recipe). That is far more work than one decoder layer, and
MTP adds a full layer plus a second pass through the output layer. In units of "one
decoder layer of work", counted from FLOPs with full recompute (`python -m moe_layout
size`):

| Unit | Cost |
| --- | --- |
| decoder layer `t` | 1.0 |
| embedding `E` | about 0 |
| loss with output layer `L` | about 2.9 |
| MTP layer `m` | about 4.0 |

With 16 layers on 4 stages, Megatron's default split looks like this:

```
stage 0  E t t t t        ####                 4.0
stage 1  t t t t          ####                 4.0
stage 2  t t t t          ####                 4.0
stage 3  t t t t m L      ###########         10.9   <- every step waits for this GPU
```

Stages 0 to 2 are idle most of each step. Moving decoder layers off the last stage
fixes that, but the best split depends on real kernel speeds, the hardware and the
memory limits, which is why moe-layout measures rather than guesses. Ling 2.0
([arXiv 2510.22115](https://arxiv.org/abs/2510.22115), section 5.2) found an MTP layer
costs about 1.7x a MoE layer at its scale, and got a 40 percent end-to-end gain by
refining its split by hand.

## What moe-layout does

1. **Size.** Parameters, FLOPs and memory per unit (`E`, `t`, `m`, `L`) from a config.
   It reproduces the handbook's 6.86B total and 1.22B active.
2. **Profile.** Run two short layouts with Megatron's per-GPU timers
   (`--timing-log-level 2 --timing-log-option all`). `profile-layouts` picks two that
   make every unit's cost measurable.
3. **Fit.** Solve for each unit's forward and backward time per micro-batch from the
   per-stage timings.
4. **Search.** Enumerate every split Megatron's validator accepts (816 for 16 layers,
   1 MTP layer, PP4), simulate the 1F1B schedule for each, and drop the ones that
   don't fit in GPU memory.
5. **Emit.** Print the best layout strings, with predicted step time and memory per
   stage, next to Megatron's default and the count-based rule.

It's for anyone training with pipeline parallelism in Megatron-LM, especially MoE
models with MTP and large vocabularies. Today the usual approach is to try layouts in
A/B runs on the full cluster.

## Related work

Checked on 2026-11-09. Details and sources are in
[moe-copilot PLAN.md](https://github.com/antonellof/moe-copilot/blob/main/PLAN.md).

| Approach | Examples | Gap |
| --- | --- | --- |
| hand-written layouts | DeepSeek-V3 and GLM-5.2 recipes, Ling 2.0 | manual, needs A/B runs |
| count-based auto layout | Megatron Lite `_auto_layout`, Megatron-Bridge | treats `E`, `m`, `L` as one layer each |
| profile-guided partitioners | PipeDream, AdaPipe | dense models, no MTP, no Megatron layout string |
| runtime MoE rebalancing | DynMo | moves layers during training, no layout string, no MTP |

## Status

| Stage | What | State |
| --- | --- | --- |
| 0 | Planner, cost model, 1F1B simulator, sizer, Megatron log parsers, tests (CPU) | done |
| A | 4x A100: does MTP with PP train at the pin; default vs count-based; 2 profiling runs | **passed** ($2.50) |
| B | 4x A100: planner's picks measured, prediction error, best of N | **passed on speed, partial on prediction** ($3.30) |
| C | 8x H200: handbook scale with expert parallelism, closed loop in one job | **passed** (about $10) |

KPI (stage B): measured step time of the planner's layout against Megatron's default
split and the count-based rule. Pass: at least 10 percent faster than count-based,
within 2 percent of the best measured layout, prediction error at most 5 percent.

All runs: 4x A100 80GB (HF Jobs `a100x4`), proxy recipe, 20 iterations, median of
iterations 6 to 20, Megatron `d6316eb45`, image `nvcr.io/nvidia/pytorch:26.06-py3`.
Outputs, including every command line, are in [results/](results).

### Stage A: profile and fit (run tag `stageA-20261109T111955Z`)

| Layout | Measured step | Predicted (fit on the 2 profiling runs) | Error |
| --- | --- | --- | --- |
| Megatron default `Et*4\|t*4\|t*4\|t*4mL` | 6.41 s | 6.53 s | +1.7% |
| count-based `Et*4\|t*5\|t*5\|t*2mL` (not used in the fit) | 5.25 s | 5.29 s | +0.6% |
| profiling 1, same as default, with timers | 6.55 s | 6.53 s | -0.3% |
| profiling 2 `Et*7\|t*5\|t*4m\|L`, with timers | 5.32 s | 5.20 s | -2.2% |

Fitted cost per micro-batch (forward + backward): decoder layer 20.0 ms, loss with
output layer 79.8 ms (4.0 layers), MTP layer 28.8 ms (1.44 layers), embedding 10.4 ms.
Per stage, the fit matches the measured compute time within about 2 ms. FLOPs got the
direction right but not the size: on A100 at micro-batch 1 the output layer is
relatively heavier and MTP lighter than its FLOPs suggest. Level-2 timers add about 4
percent to step time, so KPI runs never use them.

### Stage B: measure the planner's picks (run tag `stageB-20261109T113743Z`)

| Layout | Role | Measured step | Predicted | Error |
| --- | --- | --- | --- | --- |
| `Et*5\|t*6\|t*5\|mL` | **planner's pick** | **4.26 s** | 4.37 s | +2.7% |
| `Et*5\|t*5\|t*6\|mL` | planner's #2, fastest measured | 4.23 s | 4.39 s | +3.7% |
| `Et*6\|t*5\|t*5\|mL` | neighbour | 4.58 s | 4.64 s | +1.3% |
| `Et*4\|t*6\|t*6\|mL` | planner's #3, two equally slow stages | 4.70 s | 4.39 s | -6.4% |
| `Et*5\|t*6\|t*5m\|L` | MTP on its own stage | 5.28 s | 4.63 s | -12.3% |
| default | repeat, same node | 6.54 s | 6.53 s | -0.2% |

| KPI | Target | Result |
| --- | --- | --- |
| speedup vs Megatron default (same job, same node) | none | **1.54x** |
| speedup vs count-based rule | at least 1.10x | **1.26x** (adjusted for the 2 percent slower node; 1.24x raw), pass |
| gap to the best measured layout | at most 2% | **0.5%**, pass |
| prediction error | at most 5% | pick 2.7%, mean 4.4%; 2 of 6 layouts miss (6.4%, 12.3%), partial |

What the misses say:
- **MTP on its own stage ran 12 percent slower than predicted.** Per-stage compute
  matched in the timed profiling run, so the extra cost is outside the stage compute.
  Plausible sources are extra communication or synchronization specific to standalone
  MTP; this is not yet diagnosed. The planner flags these layouts (`mtp_standalone`),
  and `--exclude-standalone-mtp` skips them.
- **Two equally slow stages ran 6 percent slower than predicted.** The simulator uses
  fixed stage times, while real stages jitter, and two critical stages compound the
  waiting. The planner now reports `critical_stages` and prefers layouts with one
  bottleneck when predictions tie.
- **Profiling pays.** The FLOP-only prior would have picked `Et*4|t*6|t*6|mL`, which
  measured 4.70 s. The fitted pick is 10 percent faster.

Memory: calibrated on all 40 measured stages, the sizer predicts Megatron's "max
allocated" with 2.8 percent mean error (max 13.7 percent, on the loss-only stage of
the standalone-MTP layouts).

Spend: stage A $2.50, stage B $3.30.

### Stage C: handbook scale on 8x H200 (run tag `stageC-20261109T120517Z`)

The handbook's 7B-A1B with 64 experts, expert parallelism 2, data parallel 2, PP4,
micro-batch 4, global batch 128, one MTP layer, 15 iterations (median of 5 to 15).
`configs/handbook_h200x8_pp4_ep2_mtp1.json`. One job ran everything in sequence:
1. the two profiling layouts;
2. the planner, on its own fit of those runs (`plan:1`);
3. its pick, then the two baselines, on the same node.

The fixed overhead was recalibrated on the two baselines only, so the pick's
prediction is out of sample.

| Layout | Role | Measured step | Predicted | Error |
| --- | --- | --- | --- | --- |
| `Et*5\|t*6\|t*5\|mL` | **planner's pick (closed loop)** | **3.80 s** | 3.70 s | -2.7% |
| `Et*4\|t*5\|t*5\|t*2mL` | count-based rule | 4.27 s | 4.28 s | +0.3% |
| default | Megatron's own split | 5.28 s | 5.27 s | -0.2% |
| `Et*4\|t*4\|t*4\|t*4mL` | profiling 1, with timers | 5.37 s | 5.27 s | -1.8% |
| `Et*7\|t*5\|t*4m\|L` | profiling 2, with timers | 4.28 s | 4.18 s | -2.4% |

| KPI | Target | Result |
| --- | --- | --- |
| speedup vs Megatron default (same node) | none | **1.39x** |
| speedup vs count-based rule (same node) | at least 1.10x | **1.12x**, pass |
| prediction error, KPI runs | at most 5% | **max 2.7%**, pass |
| gap to the best measured layout | at most 2% | the pick was the fastest of the 3 KPI layouts; only stage B measured a wider set |

Findings:
- **The same pick on both GPU types.** On H200 the planner chose the same layout it
  chose on A100. The fitted unit costs also transfer as ratios: loss 3.6x a decoder
  layer (4.0 on A100), MTP 1.43x (1.44), embedding 0.26x (0.52).
- **The gain over count-based is smaller here** (1.12x against 1.26x on A100): with
  micro-batch 4 and expert parallelism the decoder layers are relatively heavier.
- **Memory is conservative at micro-batch 4.** The sizer predicted 52 GB for the pick's
  `mL` stage against 38 GB measured. The logit-memory term was calibrated at
  micro-batch 1. Decoder stages were within about 10 percent.

Spend: about $10 (14.4 min of h200x8).

### Before measuring: the FLOP-only prior

For reference, what the planner predicted from FLOPs alone (summary of the JSON printed by
`python -m moe_layout plan configs/proxy_a100x4_pp4_mtp1.json --pp 4 --tflops 150 --hbm-gb 80`):

```
best              Et*4|t*6|t*6|mL        3.53 s   (measured 4.70 s)
megatron_default  Et*4|t*4|t*4|t*4mL     5.38 s   (measured 6.41 s)
count_based       Et*4|t*5|t*5|t*2mL     4.45 s   (measured 5.25 s)
```

## Use

```bash
pip install git+https://github.com/antonellof/moe-layout   # or run from a clone; stdlib only
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
line with `--from-args`. Python 3.10+, no dependencies.

## Run the stages on Hugging Face Jobs

`jobs/job_entry.sh` runs on the stock `nvcr.io/nvidia/pytorch:26.06-py3` image, clones
Megatron-LM at `d6316eb45` and this repo, then `jobs/run_layouts.sh` runs each layout
in `LAYOUTS`:
- entries are separated by `;`;
- `default` means no layout flag;
- a `+prof` suffix adds the level-2 timers.

Logs go to a private HF dataset after every layout.

```bash
IMG=nvcr.io/nvidia/pytorch:26.06-py3
BOOT='git clone -q https://github.com/antonellof/moe-layout /work/moe-layout && bash /work/moe-layout/jobs/job_entry.sh'
# stage A: about $8 on a100x4 ($10/h); HF timeouts count from job creation, keep margin
hf jobs run -d --flavor a100x4 --timeout 50m --secrets HF_TOKEN -e IMAGE=$IMG \
  -e LAYOUTS="default;Et*4|t*5|t*5|t*2mL;Et*4|t*4|t*4|t*4mL+prof;Et*7|t*5|t*4m|L+prof" \
  $IMG bash -c "$BOOT"
```

The proxy recipe uses the handbook's 7B-A1B flag groups with these changes:
- PP4 and EP1;
- 32 experts instead of 64. Per-token compute is the same at top-8, and a stage fits in 80 GB without expert parallelism;
- micro-batch 1 and global batch 32;
- one MTP layer.

## How it works

- `layout.py` parses and emits Megatron's layout grammar and mirrors its validator at
  the pin:
  - `E` first and `L` last;
  - MTP after all decoder layers, with all MTP layers in one stage;
  - MTP in the last virtual stage of its GPU, and not on the first GPU.

  It enumerates every valid contiguous split.
- `costmodel.py` fits per-micro-batch forward and backward costs for `E`, `t`, `m`, `L`
  and a per-stage constant by least squares over all stages of the profiling runs.
  A weak prior toward FLOP ratios covers anything the runs can't separate.
- `simulate.py` replays Megatron's non-interleaved 1F1B order (warmup, steady 1F1B,
  cooldown) with point-to-point transfer latency.
- `sizer.py` gives parameters, FLOPs and per-stage memory: bf16 weights, fp32 gradients,
  Adam state sharded by the distributed optimizer, recompute activations and logits.

Limitations:
- Interleaved schedules (more than one virtual stage per GPU) are validated but not simulated yet.
- Routing imbalance under expert parallelism isn't modeled until stage C.
- The memory model is an estimate, to be calibrated against Megatron's `max allocated` print.

## License

MIT, see [LICENSE](LICENSE).
