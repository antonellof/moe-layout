"""Search pipeline layouts and score them with the cost model and the 1F1B simulator."""
from __future__ import annotations

from . import layout as L
from .costmodel import CostModel
from .simulate import one_f_one_b
from .sizer import ModelConfig


def score(cfg: ModelConfig, cost: CostModel, stages: list[list[str]], pp: int,
          hbm_gb: float | None = None, act_scale: float = 1.0) -> dict:
    """Predicted step time (ms), per-stage times and memory for one layout (VPP 1)."""
    if len(stages) != pp:
        raise NotImplementedError("only VPP 1 is simulated so far")
    mb = cfg.global_batch // (cfg.micro_batch * cfg.dp)
    fwd, bwd = zip(*(cost.stage_ms(st) for st in stages))
    sim = one_f_one_b(list(fwd), list(bwd), mb, cost.p2p_ms)
    first = [sum(s.count("t") for s in stages[:i]) for i in range(len(stages))]
    params = [cfg.stage_params(st, f) for st, f in zip(stages, first)]
    mem = [cfg.stage_memory_gb(st, i, pp, first_layer=f, act_scale=act_scale)
           for i, (st, f) in enumerate(zip(stages, first))]
    step = sim["makespan_ms"] + cost.overhead_for(max(params))
    tot = [f + b for f, b in zip(fwd, bwd)]
    # Caution flags from stage B (4x A100): layouts with MTP on a stage without the loss
    # ran 12 percent slower than predicted, and a layout with two equally slow stages
    # 6 percent slower. Single-bottleneck, MTP-with-loss layouts were within 4 percent.
    mtp_standalone = any("m" in st and "L" not in st for st in stages)
    critical = sum(1 for t in tot if t >= 0.97 * max(tot))
    return {"layout": L.to_string(stages), "step_ms": step, "fb_ms": sim["makespan_ms"],
            "mtp_standalone": mtp_standalone, "critical_stages": critical,
            "stage_fwd_ms": list(fwd), "stage_bwd_ms": list(bwd),
            "bottleneck_stage": sim["bottleneck_stage"], "idle_frac": sim["idle_frac"],
            "stage_params_b": [p / 1e9 for p in params], "stage_mem_gb": mem,
            "fits": hbm_gb is None or max(mem) <= hbm_gb}


def plan(cfg: ModelConfig, cost: CostModel, pp: int, vpp: int = 1, hbm_gb: float | None = None,
         top: int = 5, act_scale: float = 1.0, exclude_standalone_mtp: bool = False) -> dict:
    if vpp != 1:
        raise NotImplementedError("interleaved (VPP > 1) scoring is not implemented yet")
    cands = []
    for stages in L.enumerate_layouts(cfg.num_layers, pp, cfg.mtp_layers, vpp):
        s = score(cfg, cost, stages, pp, hbm_gb, act_scale)
        if s["fits"] and not (exclude_standalone_mtp and s["mtp_standalone"]):
            cands.append(s)
    # Ties are common (one stage sets the pace for many layouts): prefer lower peak
    # memory, then the most even stages (less fill/drain cost if the model is off).
    def key(c):
        t = [f + b for f, b in zip(c["stage_fwd_ms"], c["stage_bwd_ms"])]
        mean = sum(t) / len(t)
        return (round(c["step_ms"], 1), c["critical_stages"], round(max(c["stage_mem_gb"]), 1),
                sum((x - mean) ** 2 for x in t), c["layout"])
    cands.sort(key=key)
    base = {}
    if cfg.num_layers % pp == 0:
        base["megatron_default"] = score(cfg, cost, L.megatron_default(cfg.num_layers, pp, cfg.mtp_layers),
                                         pp, hbm_gb, act_scale)
    base["count_based"] = score(cfg, cost, L.count_based(cfg.num_layers, pp, cfg.mtp_layers),
                                pp, hbm_gb, act_scale)
    best = cands[0] if cands else None
    for b in base.values():
        b["speedup_of_best"] = (b["step_ms"] / best["step_ms"]) if best else None
    return {"searched_fitting": len(cands), "top": cands[:top], "baselines": base,
            "cost_source": cost.source}


def profiling_layouts(num_layers: int, pp: int, mtp_layers: int) -> list[str]:
    """Two layouts that identify every unit cost: the Megatron default (or count rule)
    and a shifted layout with MTP on its own stage, so m and L separate and decoder
    counts differ per stage."""
    first = (L.megatron_default(num_layers, pp, mtp_layers) if num_layers % pp == 0
             else L.count_based(num_layers, pp, mtp_layers))
    if pp < 3 or not mtp_layers:
        second = L.count_based(num_layers, pp, mtp_layers)
    else:
        # decoders spread over the first pp-1 stages with a +1/-1 tilt, MTP alone
        # at the end of stage pp-2, loss alone on the last stage
        n = pp - 1
        base, rem = divmod(num_layers, n)
        sizes = [base + (1 if i < rem else 0) for i in range(n)]
        if n >= 2 and sizes[-1] > 1:
            sizes[0] += 1
            sizes[-1] -= 1
        stages = [["t"] * k for k in sizes]
        stages[0].insert(0, "E")
        stages[-1].extend(["m"] * mtp_layers)
        stages.append(["L"])
        second = stages
    return [L.to_string(first), L.to_string(second)]
