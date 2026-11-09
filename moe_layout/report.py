"""Summarize a directory of layout runs (one subdir per layout with train.log, meta.json).

  python -m moe_layout.report RUN_ROOT [--warmup 5] [--config CONFIG --costs costs.json --pp 4]

Prints JSON: per layout, median step ms after warmup, last loss, max allocated GB per
rank, failure flag, and (with --costs) the planner's prediction and its error. KPI runs
(no "+prof") are ranked; the planner's KPI is reported against Megatron's default and
the count-based rule when both were measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import layout as L
from .costmodel import CostModel
from .plan import score
from .sizer import from_json
from .timings import parse_log, summarize


def collect(root: str, warmup: int = 5, cfg=None, cost=None, pp: int = 4) -> dict:
    rows = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        row = {"run": d.name, "layout": meta["layout"], "profiling": meta["profiling"],
               "failed": (d / "failed").exists(), "wall_s": meta.get("wall_s")}
        log = d / "train.log"
        if log.exists():
            p = parse_log(log)
            s = summarize(p, warmup=warmup)
            row.update({"iters_measured": s["iters"], "step_ms_median": s["step_ms_median"],
                        "lm_loss_last": p["loss"][max(p["loss"])] if p["loss"] else None,
                        "max_alloc_gb": {r: v / 1024 for r, v in s["max_alloc_mb"].items()}})
        if cfg is not None and cost is not None:
            lay = (L.megatron_default(cfg.num_layers, pp, cfg.mtp_layers) if meta["layout"] == "default"
                   else L.parse(meta["layout"]))
            pred = score(cfg, cost, lay, pp)
            row["predicted_step_ms"] = pred["step_ms"]
            row["predicted_mem_gb"] = pred["stage_mem_gb"]
            if row.get("step_ms_median"):
                row["prediction_error"] = (pred["step_ms"] - row["step_ms_median"]) / row["step_ms_median"]
        rows.append(row)
    kpi = [r for r in rows if not r["profiling"] and not r["failed"] and r.get("step_ms_median")]
    kpi.sort(key=lambda r: r["step_ms_median"])
    out = {"runs": rows, "kpi_ranking": [(r["layout"], r["step_ms_median"]) for r in kpi]}
    by = {r["layout"]: r["step_ms_median"] for r in kpi}
    if kpi:
        best = kpi[0]
        out["best_measured"] = {"layout": best["layout"], "step_ms": best["step_ms_median"]}
        if "default" in by:
            out["best_vs_default_speedup"] = by["default"] / best["step_ms_median"]
        if cfg is not None:
            cb = L.to_string(L.count_based(cfg.num_layers, pp, cfg.mtp_layers))
            if cb in by:
                out["best_vs_count_based_speedup"] = by[cb] / best["step_ms_median"]
        errs = [abs(r["prediction_error"]) for r in kpi if "prediction_error" in r]
        if errs:
            out["prediction_abs_error_mean"] = sum(errs) / len(errs)
            out["prediction_abs_error_max"] = max(errs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--config")
    ap.add_argument("--costs")
    ap.add_argument("--pp", type=int, default=4)
    a = ap.parse_args(argv)
    cfg = from_json(a.config) if a.config else None
    cost = CostModel.from_json(a.costs) if a.costs else None
    print(json.dumps(collect(a.root, a.warmup, cfg, cost, a.pp), indent=2))


if __name__ == "__main__":
    main()
