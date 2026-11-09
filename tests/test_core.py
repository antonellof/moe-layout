"""CPU tests for moe_layout (stdlib only). Run: python3 -m unittest discover -s tests"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_layout import layout as L  # noqa: E402
from moe_layout.costmodel import CostModel, fit  # noqa: E402
from moe_layout.plan import plan, profiling_layouts, score  # noqa: E402
from moe_layout.simulate import bottleneck_bound, one_f_one_b  # noqa: E402
from moe_layout.sizer import ModelConfig, from_json, from_megatron_args  # noqa: E402
from moe_layout.timings import parse_log, summarize  # noqa: E402

PROXY = from_json(str(ROOT / "configs" / "proxy_a100x4_pp4_mtp1.json"))
HANDBOOK = from_json(str(ROOT / "configs" / "handbook_7b_a1b.json"))


class LayoutTest(unittest.TestCase):
    def test_parse_megatron_examples(self):
        st = L.parse("Et*3|(tt|)*29m|L")  # DeepSeek-V3 layout, Megatron Core MoE report
        flat = [u for s in st for u in s]
        self.assertEqual((flat.count("E"), flat.count("t"), flat.count("m"), flat.count("L")), (1, 61, 1, 1))
        self.assertEqual(len(st), 32)
        self.assertEqual(L.parse("Ettt|(tt|)*2,m|L"), [list("Ettt"), list("tt"), list("tt"), list("m"), list("L")])

    def test_roundtrip(self):
        for s in ["Et*4|t*4|t*4|t*4mL", "Et*7|t*5|t*4m|L", "Et*22|t*24|t*24|t*8m*7L"]:
            self.assertEqual(L.to_string(L.parse(s)), s)

    def test_validator_rules(self):
        ok = L.parse("Et*4|t*4|t*4|t*4mL")
        L.validate(ok, 4, 16, 1)
        bad = {
            "tEt*3|t*4|t*4|t*4mL": "embedding",
            "Et*4|t*4|t*4|t*4Lm": "loss",
            "Et*4|t*4|t*4|t*3mL": "decoder count",
            "Et*4|t*4|t*7m|tL": "before MTP",
        }
        for s, msg in bad.items():
            with self.assertRaises(ValueError) as cm:
                L.validate(L.parse(s), 4, 16, 1)
            self.assertIn(msg, str(cm.exception))
        # VPP 2: MTP in a non-final virtual stage of its rank is rejected
        with self.assertRaises(ValueError):
            L.validate(L.parse("Et*2|t*2|t*2|t*2m|t*2|t*2|t*2|t*2L"), 4, 16, 1)

    def test_baselines(self):
        self.assertEqual(L.to_string(L.megatron_default(16, 4, 1)), "Et*4|t*4|t*4|t*4mL")
        self.assertEqual(L.to_string(L.count_based(16, 4, 1)), "Et*4|t*5|t*5|t*2mL")
        self.assertEqual(L.to_string(L.count_based(6, 4, 0)), "Et|t*2|t*2|tL")  # Lite docstring: 6/pp4 -> [1,2,2,1]

    def test_enumerate(self):
        lays = list(L.enumerate_layouts(16, 4, 1))
        self.assertEqual(len(lays), 816)  # C(18, 3): 19 units, MTP kept whole
        self.assertTrue(all(L.is_valid(x, 4, 16, 1) for x in lays))


class SizerTest(unittest.TestCase):
    def test_handbook_numbers(self):
        # article: 6.86B total, 1.221B active; one expert 3 x 2048 x 1024 = 6,291,456
        self.assertAlmostEqual(HANDBOOK.total_params() / 1e9, 6.86, delta=0.01)
        self.assertAlmostEqual(HANDBOOK.active_params() / 1e9, 1.221, delta=0.002)
        self.assertEqual(3 * HANDBOOK.hidden * HANDBOOK.moe_ffn_hidden, 6291456)

    def test_from_args(self):
        cfg = from_megatron_args(
            "--num-layers 16 --hidden-size 2048 --ffn-hidden-size 8192 --num-experts 64 "
            "--moe-ffn-hidden-size 1024 --moe-router-topk 8 --num-attention-heads 32 "
            "--num-query-groups 4 --kv-channels 64 --seq-length 4096 --micro-batch-size 8 "
            "--global-batch-size 512 --recompute-granularity full --mtp-num-layers 1")
        self.assertEqual((cfg.num_experts, cfg.topk, cfg.mtp_layers, cfg.full_recompute), (64, 8, 1, True))

    def test_head_is_heavy_at_this_width(self):
        t = sum(PROXY.unit_flops_per_microbatch("t"))
        self.assertGreater(sum(PROXY.unit_flops_per_microbatch("L")) / t, 2.5)
        self.assertGreater(sum(PROXY.unit_flops_per_microbatch("m")) / t, 3.5)


class SimulateTest(unittest.TestCase):
    def test_uniform_pipeline_closed_form(self):
        # 1F1B with equal stages: (M + P - 1) * (f + b)
        for pp, m in [(1, 8), (4, 8), (4, 32), (8, 16)]:
            r = one_f_one_b([1.0] * pp, [2.0] * pp, m)
            self.assertAlmostEqual(r["makespan_ms"], (m + pp - 1) * 3.0, places=6)

    def test_slow_stage_bounds(self):
        f, b = [1.0, 1.0, 3.0, 1.0], [2.0, 2.0, 6.0, 2.0]
        r = one_f_one_b(f, b, 32)
        self.assertGreaterEqual(r["makespan_ms"], bottleneck_bound(f, b, 32))
        self.assertEqual(r["bottleneck_stage"], 2)
        self.assertLess(r["makespan_ms"], 1.1 * bottleneck_bound(f, b, 32))


def synthetic_runs(truth: CostModel, layouts, mb=32, overhead=500.0):
    runs = []
    for s in layouts:
        st = L.parse(s)
        fb = [truth.stage_ms(x) for x in st]
        sim = one_f_one_b([a for a, _ in fb], [b for _, b in fb], mb, truth.p2p_ms)
        runs.append({"stages": st, "microbatches": mb, "fwd_ms": [a * mb for a, _ in fb],
                     "bwd_ms": [b * mb for _, b in fb], "fb_ms": sim["makespan_ms"],
                     "step_ms": sim["makespan_ms"] + overhead, "opt_ms": None})
    return runs


TRUTH = CostModel(fwd={"E": 1.5, "t": 4.0, "m": 19.0, "L": 15.0, "c0": 0.8},
                  bwd={"E": 0.5, "t": 11.0, "m": 41.0, "L": 30.0, "c0": 0.4}, p2p_ms=0.2)


class CostModelTest(unittest.TestCase):
    def test_fit_recovers_truth_from_two_profiling_layouts(self):
        prof = profiling_layouts(16, 4, 1)
        model = fit(PROXY, synthetic_runs(TRUTH, prof), prior_weight=0.01)
        for d in ("fwd", "bwd"):
            for u in ("t", "m", "L"):
                self.assertAlmostEqual(getattr(model, d)[u], getattr(TRUTH, d)[u],
                                       delta=0.03 * getattr(TRUTH, d)[u], msg=f"{d} {u}")
        self.assertAlmostEqual(model.overhead_ms, 500.0, places=3)
        # out-of-sample layout predicted within 2 percent
        target = L.parse("Et*5|t*6|t*5|mL")
        truth_step = synthetic_runs(TRUTH, ["Et*5|t*6|t*5|mL"])[0]["step_ms"]
        pred = score(PROXY, model, target, 4)["step_ms"]
        self.assertLess(abs(pred - truth_step) / truth_step, 0.02)

    def test_flops_prior(self):
        c = CostModel.from_flops(PROXY, 150.0)
        self.assertGreater(c.fwd["L"], 3 * c.fwd["t"])


class PlanTest(unittest.TestCase):
    def test_plan_finds_bruteforce_best_and_beats_baselines(self):
        res = plan(PROXY, TRUTH, 4, hbm_gb=80)
        best = res["top"][0]
        brute = min(score(PROXY, TRUTH, st, 4)["step_ms"] for st in L.enumerate_layouts(16, 4, 1))
        self.assertAlmostEqual(best["step_ms"], brute, places=6)
        self.assertGreater(res["baselines"]["megatron_default"]["speedup_of_best"], 1.2)
        self.assertGreater(res["baselines"]["count_based"]["speedup_of_best"], 1.05)

    def test_memory_filter(self):
        res = plan(PROXY, TRUTH, 4, hbm_gb=21)
        self.assertTrue(all(max(c["stage_mem_gb"]) <= 21 for c in res["top"]))


class TimingsTest(unittest.TestCase):
    def test_parse(self):
        log = "\n".join([
            " [2026-11-09 10:00:00] iteration        6/   54164 | consumed samples: 192 | "
            "elapsed time per iteration (ms): 4321.0 | throughput per GPU (TFLOP/s/GPU): 120.0 | "
            "lm loss: 7.500000E+00 | loss scale: 1.0 |",
            "[Rank 0] (after 6 iterations) memory (MB) | allocated: 1000.0 | max allocated: 25000.5 | reserved: 1.0 | max reserved: 2.0",
            "[Rank 3] (after 6 iterations) memory (MB) | allocated: 1000.0 | max allocated: 37000.0 | reserved: 1.0 | max reserved: 2.0",
            "times across ranks (ms):",
            "  forward-backward:",
            "     rank  0: 3900.00",
            "     rank  3: 3901.00",
            "  forward-compute:",
            "     rank  0: 1000.00",
            "     rank  1: 900.00",
            "     rank  2: 900.00",
            "     rank  3: 2500.00",
            "  backward-compute:",
            "     rank  0: 2000.00",
            "     rank  3: 5000.00",
            " [2026-11-09 10:00:05] iteration        7/   54164 | elapsed time per iteration (ms): 4300.0 |",
        ])
        p = Path(tempfile.mkdtemp(), "train.log")
        p.write_text(log)
        d = parse_log(p)
        self.assertEqual(d["steps"], {6: 4321.0, 7: 4300.0})
        self.assertEqual(d["timers"][6]["forward-compute"], {0: 1000.0, 1: 900.0, 2: 900.0, 3: 2500.0})
        self.assertEqual(d["max_alloc_mb"], {0: 25000.5, 3: 37000.0})
        s = summarize(d, warmup=5)
        self.assertEqual(s["timers"]["backward-compute"], {0: 2000.0, 3: 5000.0})


class CliTest(unittest.TestCase):
    def test_baselines_cli(self):
        out = subprocess.run([sys.executable, "-m", "moe_layout", "baselines",
                              str(ROOT / "configs" / "proxy_a100x4_pp4_mtp1.json"), "--pp", "4"],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout)["count_based"], "Et*4|t*5|t*5|t*2mL")


if __name__ == "__main__":
    unittest.main()
