# Changelog

## 0.1.0 (2026-11-09)

First release.

- Layout grammar and validator mirroring Megatron-LM `d6316eb45`.
- Per-unit cost model fitted from Megatron's per-rank timers on two profiling layouts.
- 1F1B schedule simulator and exhaustive search with memory filter.
- Sizer (parameters, FLOPs, memory per unit), memory calibrated on 40 measured stages.
- HF Jobs runner with a closed loop (`plan:N`) and timer-free overhead calibration.
- Validated on 4x A100 (1.54x vs Megatron's default split, 1.26x vs count-based) and on
  8x H200 at the MoE handbook's scale (1.39x and 1.12x). See docs/REPORT.md.
