"""Per-unit cost model fitted from Megatron per-rank timings.

Per micro-batch, stage s costs
    fwd_s = sum_u n_su * cf_u + cf_0        bwd_s = sum_u n_su * cb_u + cb_0
for units u in E, t, m, L (counts n_su from the layout) and a per-stage constant.
Measured per-rank 'forward-compute' and 'backward-compute' timers (ms per
iteration) divided by the micro-batch count are the observations.

Two profiling layouts on PP4 give 8 equations for 5 unknowns per direction.
E and L sit on fixed stages, so a profiling set should vary the decoder counts and
put MTP on its own stage once (see profiling_layouts) to separate m from L.
A weak ridge prior toward FLOP ratios (from the sizer) keeps any unit the data
cannot identify near its FLOP-based value instead of drifting.

Step time = simulated forward-backward makespan + overhead, with overhead
(optimizer, grad sync) = measured (elapsed - forward-backward) on the profiling
runs, adjusted by the optimizer time per parameter on the largest stage.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field

from .sizer import ModelConfig

UNITS = ("E", "t", "m", "L")


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-18:
            raise ValueError("singular system")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def counts(stage: list[str]) -> list[float]:
    return [float(stage.count(u)) for u in UNITS] + [1.0]


@dataclass
class CostModel:
    fwd: dict = field(default_factory=dict)   # unit -> ms per micro-batch, plus "c0"
    bwd: dict = field(default_factory=dict)
    overhead_ms: float = 0.0                  # optimizer + grad sync at overhead_ref_params
    overhead_ref_params: float = 0.0
    opt_ms_per_param: float = 0.0
    p2p_ms: float = 0.0
    source: str = "flops"
    fit_report: dict = field(default_factory=dict)

    def stage_ms(self, stage: list[str]) -> tuple[float, float]:
        c = counts(stage)
        keys = list(UNITS) + ["c0"]
        return (sum(n * self.fwd.get(k, 0.0) for n, k in zip(c, keys)),
                sum(n * self.bwd.get(k, 0.0) for n, k in zip(c, keys)))

    def overhead_for(self, max_stage_params: float) -> float:
        return self.overhead_ms + self.opt_ms_per_param * (max_stage_params - self.overhead_ref_params)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(path: str) -> "CostModel":
        return CostModel(**json.load(open(path)))

    # ------------------------------------------------------------ priors
    @staticmethod
    def from_flops(cfg: ModelConfig, tflops_effective: float, p2p_ms: float = 0.2) -> "CostModel":
        """Cost model from FLOPs alone (no profiling): ms = FLOPs / effective rate."""
        rate = tflops_effective * 1e12 / 1e3  # FLOPs per ms
        fwd, bwd = {}, {}
        for u in UNITS:
            f, b = cfg.unit_flops_per_microbatch(u)
            fwd[u], bwd[u] = f / rate, b / rate
        fwd["c0"] = bwd["c0"] = 0.0
        return CostModel(fwd=fwd, bwd=bwd, p2p_ms=p2p_ms, source=f"flops@{tflops_effective}TFLOPs")


def fit(cfg: ModelConfig, runs: list[dict], prior_weight: float = 0.05, p2p_ms: float = 0.2) -> CostModel:
    """Fit unit costs from profiling runs.

    Each run: {"stages": [[...], ...], "microbatches": M,
               "fwd_ms": [per-stage forward-compute ms per iteration],
               "bwd_ms": [per-stage backward-compute ms per iteration],
               "step_ms": elapsed median, "fb_ms": forward-backward median,
               "opt_ms": [per-stage optimizer ms per iteration] (optional)}
    """
    flops = {u: cfg.unit_flops_per_microbatch(u) for u in UNITS}
    model = CostModel(p2p_ms=p2p_ms, source="fit")
    report = {}
    for direction, key, idx in (("fwd", "fwd_ms", 0), ("bwd", "bwd_ms", 1)):
        rows, ys, fl = [], [], []
        for r in runs:
            m = r["microbatches"]
            for st, v in zip(r["stages"], r[key]):
                rows.append(counts(st))
                ys.append(v / m)
                fl.append(sum(st.count(u) * flops[u][idx] for u in UNITS))
        kappa = statistics.median(y / f for y, f in zip(ys, fl) if f > 0)  # ms per FLOP
        prior = [kappa * flops[u][idx] for u in UNITS] + [0.0]
        # Ridge toward the FLOP prior: each prior term weighs lam^2 against data rows of
        # weight 1, so measured stages dominate wherever the layouts identify a unit.
        lam2 = (prior_weight ** 2) * len(ys)
        n = len(prior)
        ata = [[sum(row[i] * row[j] for row in rows) for j in range(n)] for i in range(n)]
        aty = [sum(row[i] * y for row, y in zip(rows, ys)) for i in range(n)]
        for i in range(n):
            w = lam2 if i < len(UNITS) else lam2 * 0.01  # c0 is only weakly pulled to 0
            ata[i][i] += w
            aty[i] += w * prior[i]
        coef = _solve(ata, aty)
        pred = [sum(c * x for c, x in zip(coef, row)) for row in rows]
        resid = [abs(p - y) / y for p, y in zip(pred, ys) if y > 0]
        getattr(model, direction).update({u: coef[i] for i, u in enumerate(UNITS)})
        getattr(model, direction)["c0"] = coef[-1]
        report[direction] = {"kappa_ms_per_gflop": kappa * 1e9, "prior": dict(zip(UNITS, prior)),
                             "max_rel_residual": max(resid) if resid else None,
                             "mean_rel_residual": statistics.mean(resid) if resid else None,
                             "points": len(ys)}
    # overhead: elapsed minus forward-backward, adjusted by optimizer time on the largest stage
    over, maxp, opt_pts = [], [], []
    for r in runs:
        params = [cfg.stage_params(st, sum(s.count("t") for s in r["stages"][:i]))
                  for i, st in enumerate(r["stages"])]
        if r.get("step_ms") and r.get("fb_ms"):
            over.append(r["step_ms"] - r["fb_ms"])
            maxp.append(max(params))
        for p, o in zip(params, r.get("opt_ms") or []):
            opt_pts.append((p, o))
    if over:
        model.overhead_ms = statistics.mean(over)
        model.overhead_ref_params = statistics.mean(maxp)
    if len(opt_pts) >= 2:
        mx = statistics.mean(p for p, _ in opt_pts)
        my = statistics.mean(o for _, o in opt_pts)
        sxx = sum((p - mx) ** 2 for p, _ in opt_pts)
        model.opt_ms_per_param = (sum((p - mx) * (o - my) for p, o in opt_pts) / sxx) if sxx else 0.0
    report["overhead_ms"] = model.overhead_ms
    report["opt_ms_per_billion_params"] = model.opt_ms_per_param * 1e9
    model.fit_report = report
    return model


def run_from_log(cfg: ModelConfig, stages: list[list[str]], summary: dict, microbatches: int,
                 ranks_per_stage: int = 1) -> dict:
    """Turn timings.summarize() output into a fit() run (stage s = ranks s*k .. s*k+k-1)."""
    def per_stage(name):
        t = summary["timers"].get(name, {})
        vals = []
        for s in range(len(stages)):
            rs = [t[r] for r in range(s * ranks_per_stage, (s + 1) * ranks_per_stage) if r in t]
            vals.append(max(rs) if rs else None)
        return vals
    fwd, bwd, opt = per_stage("forward-compute"), per_stage("backward-compute"), per_stage("optimizer")
    if None in fwd or None in bwd:
        raise ValueError("missing forward/backward-compute timers: run with "
                         "--timing-log-level 2 --timing-log-option all")
    fb = summary["timers"].get("forward-backward", {})
    return {"stages": stages, "microbatches": microbatches, "fwd_ms": fwd, "bwd_ms": bwd,
            "step_ms": summary["step_ms_median"], "fb_ms": max(fb.values()) if fb else None,
            "opt_ms": opt if None not in opt else None}
