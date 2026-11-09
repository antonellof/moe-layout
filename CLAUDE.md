# moe-layout

Measured `--pipeline-model-parallel-layout` planner (plus sizer) for Megatron-LM MoE
models with MTP. Successor of the moe-copilot validation
(https://github.com/antonellof/moe-copilot, see its PLAN.md and RESULTS.md).
Megatron pinned at d6316eb45bf1bfbb6f1f848c0b56e55166564d80.

## Current phase: stages A, B, C done (about $16 of $30); results in README and results/

## Budget and rules (from the user)
- Total GPU budget for stages A, B, C: $30. Ask before each paid stage, with the cost.
- If a constraint forces a design change, propose it and get approval before running.
- Never touch HF Jobs this session did not launch.
- No em dash character anywhere. One change per run. Record image, Megatron commit
  and full command with every result (jobs write meta.json per layout).

## Verified against Megatron at the pin
- Layout grammar (pipeline_parallel_layer_layout.py): E, t, m, L; "|" between stages;
  "(...)*n" and "x*n"; stage s -> pp_rank s % PP, vpp_rank s // PP.
- Validator: E first, L last, once each; decoder and MTP counts match; no decoder after
  the first MTP; MTP only in the last virtual stage of its rank and all together; MTP
  not on PP rank 0 (checked only for non-final virtual stages); empty stages are not
  rejected (the planner does not search them).
- Without a layout flag, MTP goes on the last PP stage (multi_token_prediction.py
  mtp_on_this_rank); with a layout it can sit on any allowed rank (standalone MTP).
- Megatron Lite's _auto_layout (dev branch) splits [E, t*N, m*K, L] into equal unit
  counts, first `remainder` stages one larger: 16 layers, 1 MTP, PP4 -> Et*4|t*5|t*5|t*2mL.
- Timers with --timing-log-level 2 --timing-log-option all print after each iteration
  line: "times across ranks (ms):", then "  name:" and "     rank NN: X" (ms per
  iteration, summed over micro-batches; ranks with 0 are omitted).
- Memory: "[Rank N] (after K iterations) memory (MB) | ... | max allocated: X", from
  every DP-rank-0 (so every PP stage), early and every --log-memory-interval.

## Verified on 4x A100 (stage A stageA-20261109T111955Z, stage B stageB-20261109T113743Z)
- MTP with PP4 and custom layouts trains at the pin with the handbook flags, including
  standalone MTP (t*4m|L) and a loss-only last stage.
- Fitted per micro-batch (fwd+bwd): t 20.0 ms, L 79.8 ms, m 28.8 ms, E 10.4 ms, c0 3.5 ms;
  per-stage fit within about 2 ms. Level-2 timers add about 4 percent to step time.
- KPI: pick Et*5|t*6|t*5|mL 4.26 s vs default 6.54 s same node (1.54x) and count-based
  5.25 s (1.26x node-adjusted); within 0.5 percent of best measured (Et*5|t*5|t*6|mL).
- Prediction misses: standalone MTP -12.3 percent (outside stage compute, undiagnosed),
  two equal bottleneck stages -6.4 percent (jitter). Flags: mtp_standalone,
  critical_stages; --exclude-standalone-mtp.
- Node-to-node: the same default layout ran 2 percent slower on the stage B node.
- Memory: max allocated = weights + optimizer state + 10.3 bytes per logit per head
  (L, m) + 0.13 GB; 2.8 percent mean error over 40 stages.

## Verified on 8x H200 (stage C stageC-20261109T120517Z)
- Handbook scale (64 experts, EP2, DP2, PP4, MBS4, GBS128, MTP1) trains at the pin with
  custom layouts. Ranks 2k, 2k+1 form stage k (tp-cp-ep-dp-pp order).
- Closed loop (plan:1 in place) picked Et*5|t*6|t*5|mL: 3.80 s vs default 5.28 s
  (1.39x) and count-based 4.27 s (1.12x); prediction error max 2.7 percent on KPI runs.
- Unit cost ratios: L 3.62 t, m 1.43 t, E 0.26 t (A100: 4.0, 1.44, 0.52).
- Memory model over-predicts the mL stage at MBS4 (52 vs 38 GB); logit bytes were
  calibrated at MBS1.

## HF Jobs facts (from moe-copilot)
- Stock nvcr.io image pulls directly; jobs need pre-paid credits (402 otherwise).
- Timeouts count from job creation (scheduling plus pull, 3 to 5 min): keep margin.
- a100x4 4x A100 80GB $10/h; l40sx4 $8.30/h; h200x8 $40/h.
- Data: antonellof/moe-copilot-day1 (private) holds data/fineweb-edu_{5000,300000}_text_document.
