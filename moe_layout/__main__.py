"""moe-layout CLI.

  python -m moe_layout size CONFIG [--layout STR --pp N]
  python -m moe_layout baselines CONFIG --pp N
  python -m moe_layout profile-layouts CONFIG --pp N
  python -m moe_layout fit CONFIG --run "LAYOUT=train.log" [--run ...] -o costs.json
  python -m moe_layout plan CONFIG --pp N (--costs costs.json | --tflops 150) [--hbm-gb 80]
  python -m moe_layout score CONFIG --pp N --costs costs.json --layout STR [--measured-ms X]

CONFIG is a JSON ModelConfig (see configs/) or a file holding a Megatron command line
(--from-args).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import layout as L
from .costmodel import CostModel, fit, run_from_log
from .plan import plan, profiling_layouts, score
from .sizer import ModelConfig, from_json, from_megatron_args
from .timings import parse_log, summarize


def _cfg(a) -> ModelConfig:
    if a.from_args:
        return from_megatron_args(open(a.config).read())
    return from_json(a.config)


def _cost(a, cfg) -> CostModel:
    if getattr(a, "costs", None):
        return CostModel.from_json(a.costs)
    if getattr(a, "tflops", None):
        return CostModel.from_flops(cfg, a.tflops)
    sys.exit("give --costs (fitted) or --tflops (FLOP-only prior)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="moe_layout", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, pp=True):
        sp.add_argument("config")
        sp.add_argument("--from-args", action="store_true", help="CONFIG is a Megatron command line")
        if pp:
            sp.add_argument("--pp", type=int, required=True)
        return sp

    s = common(sub.add_parser("size", help="parameters, FLOPs, memory"), pp=False)
    s.add_argument("--layout")
    s.add_argument("--pp", type=int)
    common(sub.add_parser("baselines", help="Megatron default and count-based layouts"))
    common(sub.add_parser("profile-layouts", help="two layouts that identify all unit costs"))
    f = common(sub.add_parser("fit", help="fit unit costs from profiled runs"), pp=False)
    f.add_argument("--run", action="append", required=True, help="LAYOUT=path/to/train.log")
    f.add_argument("--warmup", type=int, default=5)
    f.add_argument("--ranks-per-stage", type=int, default=1)
    f.add_argument("--p2p-ms", type=float, default=0.2)
    f.add_argument("-o", "--out", required=True)
    pl = common(sub.add_parser("plan", help="rank layouts"))
    pl.add_argument("--costs")
    pl.add_argument("--tflops", type=float)
    pl.add_argument("--hbm-gb", type=float)
    pl.add_argument("--top", type=int, default=5)
    sc = common(sub.add_parser("score", help="predict one layout"))
    sc.add_argument("--costs")
    sc.add_argument("--tflops", type=float)
    sc.add_argument("--layout", required=True)
    sc.add_argument("--hbm-gb", type=float)
    sc.add_argument("--measured-ms", type=float)
    a = p.parse_args(argv)
    cfg = _cfg(a)

    if a.cmd == "size":
        out = {"total_params_b": cfg.total_params() / 1e9, "active_params_b": cfg.active_params() / 1e9,
               "units": {u: {"params_m": cfg.unit_params(u) / 1e6,
                             "gflop_fwd_per_microbatch": cfg.unit_flops_per_microbatch(u)[0] / 1e9,
                             "gflop_bwd_per_microbatch": cfg.unit_flops_per_microbatch(u)[1] / 1e9}
                         for u in "EtmL" if u != "m" or cfg.mtp_layers}}
        if a.layout:
            st = L.parse(a.layout)
            pp = a.pp or len(st)
            L.validate(st, pp, cfg.num_layers, cfg.mtp_layers)
            first = [sum(x.count("t") for x in st[:i]) for i in range(len(st))]
            out["stages"] = [{"stage": L.to_string([x]), "params_b": cfg.stage_params(x, f0) / 1e9,
                              "mem_gb": cfg.stage_memory_gb(x, i, pp, first_layer=f0)}
                             for i, (x, f0) in enumerate(zip(st, first))]
    elif a.cmd == "baselines":
        out = {"megatron_default": L.to_string(L.megatron_default(cfg.num_layers, a.pp, cfg.mtp_layers))
               if cfg.num_layers % a.pp == 0 else None,
               "count_based": L.to_string(L.count_based(cfg.num_layers, a.pp, cfg.mtp_layers))}
    elif a.cmd == "profile-layouts":
        out = {"profiling_layouts": profiling_layouts(cfg.num_layers, a.pp, cfg.mtp_layers)}
    elif a.cmd == "fit":
        runs = []
        mb = cfg.global_batch // (cfg.micro_batch * cfg.dp)
        for spec in a.run:
            lay, path = spec.split("=", 1)
            st = L.parse(lay)
            summ = summarize(parse_log(path), warmup=a.warmup)
            runs.append(run_from_log(cfg, st, summ, mb, a.ranks_per_stage))
        model = fit(cfg, runs, p2p_ms=a.p2p_ms)
        open(a.out, "w").write(model.to_json())
        out = {"written": a.out, "fwd": model.fwd, "bwd": model.bwd, "report": model.fit_report}
    elif a.cmd == "plan":
        out = plan(cfg, _cost(a, cfg), a.pp, hbm_gb=a.hbm_gb, top=a.top)
    elif a.cmd == "score":
        st = L.parse(a.layout)
        L.validate(st, a.pp, cfg.num_layers, cfg.mtp_layers)
        out = score(cfg, _cost(a, cfg), st, a.pp, a.hbm_gb)
        if a.measured_ms:
            out["measured_ms"] = a.measured_ms
            out["prediction_error"] = (out["step_ms"] - a.measured_ms) / a.measured_ms
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
