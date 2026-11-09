"""Discrete-event simulation of Megatron's non-interleaved 1F1B schedule.

Per pipeline stage s the op order is the one in
megatron/core/pipeline_parallel/schedules.py (forward_backward_pipelining_without_interleaving):
warmup = min(PP - s - 1, M) forwards, then M - warmup steady pairs (F, B),
then the remaining backwards. F(s, i) waits for F(s-1, i) plus one p2p hop;
B(s, i) waits for B(s+1, i) plus one hop (the last stage's B waits for its own F).

Inputs are per-micro-batch forward and backward times per stage (ms). Output
is the forward-backward makespan (ms), the quantity Megatron's
'forward-backward' timer measures, plus per-stage busy and idle time.
"""
from __future__ import annotations


def one_f_one_b(fwd: list[float], bwd: list[float], microbatches: int, p2p_ms: float = 0.0) -> dict:
    pp = len(fwd)
    if len(bwd) != pp or pp == 0:
        raise ValueError("fwd and bwd need one entry per stage")
    m = microbatches
    orders = []
    for s in range(pp):
        warm = min(pp - s - 1, m)
        ops = [("F", i) for i in range(warm)]
        for k in range(m - warm):
            ops.append(("F", warm + k))
            ops.append(("B", k))
        ops.extend(("B", i) for i in range(m - warm, m))
        orders.append(ops)

    done = {}               # (kind, stage, mb) -> end time
    free = [0.0] * pp       # when each stage becomes free
    pos = [0] * pp
    total_ops = sum(len(o) for o in orders)
    scheduled = 0
    while scheduled < total_ops:
        progress = False
        for s in range(pp):
            while pos[s] < len(orders[s]):
                kind, i = orders[s][pos[s]]
                if kind == "F":
                    dep = 0.0 if s == 0 else done.get(("F", s - 1, i))
                    if dep is None:
                        break
                    ready = dep + (p2p_ms if s else 0.0)
                    dur = fwd[s]
                else:
                    own = done.get(("F", s, i))
                    nxt = done.get(("B", s + 1, i)) if s < pp - 1 else own
                    if own is None or nxt is None:
                        break
                    ready = max(own, nxt + (p2p_ms if s < pp - 1 else 0.0))
                    dur = bwd[s]
                start = max(free[s], ready)
                end = start + dur
                done[(kind, s, i)] = end
                free[s] = end
                pos[s] += 1
                scheduled += 1
                progress = True
        if not progress:
            raise RuntimeError("schedule deadlock (bug)")
    makespan = max(free)
    busy = [m * (fwd[s] + bwd[s]) for s in range(pp)]
    return {"makespan_ms": makespan, "busy_ms": busy,
            "idle_frac": [1 - b / makespan for b in busy],
            "bottleneck_stage": max(range(pp), key=lambda s: fwd[s] + bwd[s])}


def bottleneck_bound(fwd, bwd, microbatches) -> float:
    """Strict lower bound on the makespan: the slowest stage's total work."""
    return microbatches * max(f + b for f, b in zip(fwd, bwd))
