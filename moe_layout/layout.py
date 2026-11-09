"""Megatron-LM pipeline layouts: parse, emit, validate, enumerate.

Mirrors megatron/core/transformer/pipeline_parallel_layer_layout.py at
d6316eb45bf1bfbb6f1f848c0b56e55166564d80:

- units: E (embedding), t (decoder layer), m (MTP layer), L (loss, includes the
  output layer); stages are separated by "|"; "(...)*n" and "x*n" repeat.
- the 1D string lists stages in model order, virtual-stage major:
  stage index s -> pp_rank = s % PP, vpp_rank = s // PP.
- validator: E first and L last, exactly once each; decoder and MTP counts match;
  no decoder after the first MTP layer; MTP only in the last virtual stage of a
  rank, all MTP layers in that one stage; no MTP on PP rank 0 (checked for the
  non-final virtual stages, as at the pin).

A layout here is a list of stages, each a list of unit characters.
"""
from __future__ import annotations

import re
from itertools import combinations

UNITS = "EtmL"


def parse(layout_str: str) -> list[list[str]]:
    """Parse a Megatron layout string into stages of unit characters."""
    s = layout_str.replace(",", "").replace(" ", "")
    for pattern in (r"\(([^)]+)\)\*(\d+)", r"(.)\*(\d+)"):
        s = re.sub(pattern, lambda m: m.group(1) * int(m.group(2)), s)
    stages = [list(stage) for stage in s.split("|")]
    for stage in stages:
        for ch in stage:
            if ch not in UNITS:
                raise ValueError(f"invalid layer character {ch!r} in {layout_str!r}")
    return stages


def _compact(stage: list[str]) -> str:
    out, i = [], 0
    while i < len(stage):
        j = i
        while j < len(stage) and stage[j] == stage[i]:
            j += 1
        n = j - i
        out.append(stage[i] if n == 1 else f"{stage[i]}*{n}")
        i = j
    return "".join(out)


def to_string(stages: list[list[str]]) -> str:
    """Emit a layout string Megatron accepts, compacting runs as x*n."""
    return "|".join(_compact(st) for st in stages)


def validate(stages: list[list[str]], pp: int, num_layers: int, mtp_layers: int = 0) -> None:
    """Raise ValueError where Megatron's validator (at the pin) would assert."""
    if len(stages) % pp:
        raise ValueError(f"{len(stages)} stages not divisible by PP {pp}")
    vpp = len(stages) // pp
    flat = [u for st in stages for u in st]
    if not flat or flat[0] != "E":
        raise ValueError("the first layer must be embedding")
    if flat[-1] != "L":
        raise ValueError("the last layer must be loss")
    if flat.count("E") != 1 or flat.count("L") != 1:
        raise ValueError("embedding and loss must appear exactly once")
    if flat.count("t") != num_layers:
        raise ValueError(f"decoder count {flat.count('t')} != num_layers {num_layers}")
    if flat.count("m") != mtp_layers:
        raise ValueError(f"mtp count {flat.count('m')} != mtp_layers {mtp_layers}")
    if "m" in flat and "t" in flat[flat.index("m"):]:
        raise ValueError("decoder layers must be placed before MTP layers")
    grid = [[stages[v * pp + p] for v in range(vpp)] for p in range(pp)]
    for p in range(pp):
        for v in range(vpp - 1):
            if "m" in grid[p][v]:
                raise ValueError("MTP must be in the last virtual stage of its rank")
        if "m" in grid[p][-1] and grid[p][-1].count("m") != mtp_layers:
            raise ValueError("all MTP layers must be in one virtual stage")
    for v in range(vpp - 1):
        if "m" in grid[0][v]:
            raise ValueError("MTP must not be on the first PP rank")


def is_valid(stages, pp, num_layers, mtp_layers=0) -> bool:
    try:
        validate(stages, pp, num_layers, mtp_layers)
    except ValueError:
        return False
    return True


def megatron_default(num_layers: int, pp: int, mtp_layers: int = 0) -> list[list[str]]:
    """What Megatron builds without --pipeline-model-parallel-layout (VPP 1):
    decoders split evenly, embedding on the first stage, MTP and loss on the last."""
    if num_layers % pp:
        raise ValueError("default split needs num_layers divisible by PP")
    per = num_layers // pp
    stages = [["t"] * per for _ in range(pp)]
    stages[0].insert(0, "E")
    stages[-1].extend(["m"] * mtp_layers + ["L"])
    return stages


def count_based(num_layers: int, pp: int, mtp_layers: int = 0) -> list[list[str]]:
    """Megatron Lite's _auto_layout rule: split [E, t*N, m*K, L] into even
    contiguous chunks by unit count (first `remainder` stages get one more)."""
    units = ["E"] + ["t"] * num_layers + ["m"] * mtp_layers + ["L"]
    base, rem = divmod(len(units), pp)
    out, pos = [], 0
    for s in range(pp):
        size = base + (1 if s < rem else 0)
        out.append(units[pos:pos + size])
        pos += size
    return out


def enumerate_layouts(num_layers: int, pp: int, mtp_layers: int = 0, vpp: int = 1,
                      allow_empty: bool = False):
    """Yield every contiguous split of [E, t*N, m*K, L] into PP*VPP stages that
    passes the validator. MTP layers are kept together (the validator requires it)."""
    seq = ["E"] + ["t"] * num_layers + (["M"] if mtp_layers else []) + ["L"]
    n_stages = pp * vpp
    n = len(seq)
    if allow_empty:
        raise NotImplementedError("empty stages are not searched")
    for cuts in combinations(range(1, n), n_stages - 1):
        bounds = (0,) + cuts + (n,)
        stages = []
        for a, b in zip(bounds, bounds[1:]):
            st = []
            for u in seq[a:b]:
                st.extend(["m"] * mtp_layers if u == "M" else [u])
            stages.append(st)
        if is_valid(stages, pp, num_layers, mtp_layers):
            yield stages
