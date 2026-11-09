# moe-layout

Measured `--pipeline-model-parallel-layout` planner (plus sizer) for Megatron-LM MoE
models with MTP. Successor of the moe-copilot validation
(https://github.com/antonellof/moe-copilot, see its PLAN.md and RESULTS.md).
Megatron pinned at d6316eb45bf1bfbb6f1f848c0b56e55166564d80.

## Current phase: stage 0 done (CPU); stage A next, needs the user's OK

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

## Not yet verified (stage A)
- MTP with PP4 and a custom layout trains at the pin with the handbook flags
  (cross-entropy fusion, full recompute, tied embeddings).
- A loss-only last stage (Et*7|t*5|t*4m|L) builds and trains.
- Unit costs and the memory model against measured timings and max allocated.

## HF Jobs facts (from moe-copilot)
- Stock nvcr.io image pulls directly; jobs need pre-paid credits (402 otherwise).
- Timeouts count from job creation (scheduling plus pull, 3 to 5 min): keep margin.
- a100x4 4x A100 80GB $10/h; l40sx4 $8.30/h; h200x8 $40/h.
- Data: antonellof/moe-copilot-day1 (private) holds data/fineweb-edu_{5000,300000}_text_document.
