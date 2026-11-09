"""Parse Megatron train.log output at d6316eb45.

- iteration lines: " [ts] iteration N/M | ... | elapsed time per iteration (ms): X | ..."
- per-rank timers (--timing-log-level 2 --timing-log-option all), printed after
  each iteration line:
      times across ranks (ms):
        forward-compute:
           rank  0: 1234.56
           rank  1: ...
  Values are ms per iteration (summed over micro-batches), divided by log interval.
- memory, printed by every pipeline rank early on and every --log-memory-interval:
      [Rank 3] (after 2 iterations) memory (MB) | allocated: ... | max allocated: X | ...
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

STEP_RE = re.compile(r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)")
LOSS_RE = re.compile(r"\blm loss:\s*([-+\d.eE]+)")
TIMER_HEAD_RE = re.compile(r"^\s{2}([a-z][a-z0-9-]*):\s*$")
TIMER_RANK_RE = re.compile(r"^\s+rank\s+(\d+):\s*([\d.]+)\s*$")
MEM_RE = re.compile(r"\[Rank (\d+)\].*?memory \(MB\).*?max allocated:\s*([\d.]+)")


def parse_log(path) -> dict:
    """Return {"steps": {it: ms}, "loss": {it: x}, "timers": {it: {name: {rank: ms}}},
    "max_alloc_mb": {rank: mb}}."""
    steps, loss, timers, mem = {}, {}, {}, {}
    cur_it, cur_timer, in_block = None, None, False
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if m:
            cur_it = int(m.group(1))
            steps[cur_it] = float(m.group(2))
            lm = LOSS_RE.search(line)
            if lm:
                loss[cur_it] = float(lm.group(1))
            in_block = False
            continue
        if "times across ranks (ms):" in line:
            in_block, cur_timer = True, None
            continue
        mm = MEM_RE.search(line)
        if mm:
            r, v = int(mm.group(1)), float(mm.group(2))
            mem[r] = max(mem.get(r, 0.0), v)
            continue
        if in_block and cur_it is not None:
            h = TIMER_HEAD_RE.match(line)
            if h:
                cur_timer = h.group(1)
                continue
            rr = TIMER_RANK_RE.match(line)
            if rr and cur_timer:
                timers.setdefault(cur_it, {}).setdefault(cur_timer, {})[int(rr.group(1))] = float(rr.group(2))
                continue
            if line.strip() and not rr and not h:
                in_block = False
    return {"steps": steps, "loss": loss, "timers": timers, "max_alloc_mb": mem}


def summarize(parsed: dict, warmup: int = 5, ranks: list[int] | None = None) -> dict:
    """Medians over iterations after warmup: step ms and per-rank timer ms."""
    its = [i for i in sorted(parsed["steps"]) if i > warmup]
    out = {"iters": len(its),
           "step_ms_median": statistics.median(parsed["steps"][i] for i in its) if its else None,
           "timers": {}, "max_alloc_mb": parsed["max_alloc_mb"]}
    names = sorted({n for i in its for n in parsed["timers"].get(i, {})})
    for n in names:
        per_rank = {}
        for i in its:
            for r, v in parsed["timers"].get(i, {}).get(n, {}).items():
                if ranks is None or r in ranks:
                    per_rank.setdefault(r, []).append(v)
        out["timers"][n] = {r: statistics.median(v) for r, v in sorted(per_rank.items())}
    return out
