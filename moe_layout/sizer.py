"""Sizer: parameters, FLOPs and memory per pipeline unit, from a model config.

Units follow the layout grammar: E (embedding), t (decoder layer), m (MTP
layer), L (loss, includes the output layer). Counts are analytic (bias-free
SwiGLU experts, GQA attention, as in the MoE handbook recipe); memory is an
estimate meant to be calibrated against Megatron's own "max allocated" print.

The handbook 7B-A1B recipe gives 6.86B total and 1.22B active parameters here
(see tests), matching the article.
"""
from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    num_layers: int = 16
    hidden: int = 2048
    ffn_hidden: int = 8192            # dense FFN width (dense layers and MTP if dense)
    num_experts: int = 64             # 0 = dense model
    moe_ffn_hidden: int = 1024
    topk: int = 8
    shared_expert_hidden: int = 0
    heads: int = 32
    query_groups: int = 4
    kv_channels: int = 64
    vocab: int = 128256
    seq_length: int = 4096
    micro_batch: int = 1
    global_batch: int = 32
    mtp_layers: int = 0
    tied_embeddings: bool = True
    full_recompute: bool = True       # --recompute-granularity full
    moe_layer_freq: list = field(default_factory=list)  # per-layer 1=MoE 0=dense; empty = all MoE
    dp: int = 1
    distributed_optimizer: bool = True

    # ------------------------------------------------------------ building blocks
    def is_moe(self, layer_idx: int) -> bool:
        if self.num_experts == 0:
            return False
        return bool(self.moe_layer_freq[layer_idx]) if self.moe_layer_freq else True

    def attn_params(self) -> int:
        q = self.heads * self.kv_channels
        kv = 2 * self.query_groups * self.kv_channels
        return self.hidden * (q + kv) + q * self.hidden + 2 * self.kv_channels + 2 * self.hidden

    def attn_macs_per_token(self) -> float:
        q = self.heads * self.kv_channels
        kv = 2 * self.query_groups * self.kv_channels
        proj = self.hidden * (q + kv) + q * self.hidden
        core = self.seq_length * q  # QK^T and AV, causal halves each: 2 * 0.5 * seq * q
        return proj + core

    def mlp_params(self, moe: bool) -> int:
        if not moe:
            return 3 * self.hidden * self.ffn_hidden
        p = self.num_experts * 3 * self.hidden * self.moe_ffn_hidden + self.hidden * self.num_experts
        if self.shared_expert_hidden:
            p += 3 * self.hidden * self.shared_expert_hidden
        return p

    def mlp_macs_per_token(self, moe: bool) -> float:
        if not moe:
            return 3 * self.hidden * self.ffn_hidden
        m = self.topk * 3 * self.hidden * self.moe_ffn_hidden + self.hidden * self.num_experts
        if self.shared_expert_hidden:
            m += 3 * self.hidden * self.shared_expert_hidden
        return m

    # ------------------------------------------------------------ per unit
    def unit_params(self, unit: str, layer_idx: int = 0) -> int:
        h, v = self.hidden, self.vocab
        if unit == "E":
            return v * h
        if unit == "t":
            return self.attn_params() + self.mlp_params(self.is_moe(layer_idx))
        if unit == "m":
            # transformer layer (same block as the last decoder layer) + eh_proj + 2 norms
            return (self.attn_params() + self.mlp_params(self.is_moe(self.num_layers - 1))
                    + 2 * h * h + 2 * h)
        if unit == "L":
            # output layer; with tied embeddings Megatron still keeps a copy on the last stage
            return v * h + h
        raise ValueError(unit)

    def unit_macs_per_token(self, unit: str, layer_idx: int = 0) -> float:
        h, v = self.hidden, self.vocab
        if unit == "E":
            return 0.0
        if unit == "t":
            return self.attn_macs_per_token() + self.mlp_macs_per_token(self.is_moe(layer_idx))
        if unit == "m":
            # MTP transformer layer + eh_proj + its own logits through the shared output layer
            return (self.attn_macs_per_token() + self.mlp_macs_per_token(self.is_moe(self.num_layers - 1))
                    + 2 * h * h + h * v)
        if unit == "L":
            return float(h * v)
        raise ValueError(unit)

    def unit_flops_per_microbatch(self, unit: str, layer_idx: int = 0) -> tuple[float, float]:
        """(forward, backward) FLOPs for one micro-batch, counting the recompute
        forward inside backward for transformer layers under full recompute."""
        tokens = self.seq_length * self.micro_batch
        macs = self.unit_macs_per_token(unit, layer_idx) * tokens
        fwd, bwd = 2 * macs, 4 * macs
        if self.full_recompute and unit in ("t", "m"):
            bwd += 2 * macs if unit == "t" else 2 * (macs - self.hidden * self.vocab * tokens)
        return fwd, bwd

    # ------------------------------------------------------------ totals
    def total_params(self) -> int:
        n = self.unit_params("E") + sum(self.unit_params("t", i) for i in range(self.num_layers))
        n += self.mtp_layers * self.unit_params("m")
        if not self.tied_embeddings:
            n += self.unit_params("L")
        else:
            n += self.hidden  # final norm
        return n

    def active_params(self) -> int:
        n = self.vocab * self.hidden  # embedding
        for i in range(self.num_layers):
            moe = self.is_moe(i)
            n += self.attn_params()
            if moe:
                n += self.topk * 3 * self.hidden * self.moe_ffn_hidden + self.hidden * self.num_experts
                if self.shared_expert_hidden:
                    n += 3 * self.hidden * self.shared_expert_hidden
            else:
                n += 3 * self.hidden * self.ffn_hidden
        return n + self.hidden

    # ------------------------------------------------------------ memory
    def stage_params(self, stage: list[str], first_layer: int = 0) -> int:
        """Parameters held by one stage; first_layer is the global index of its first t."""
        n, li = 0, first_layer
        for u in stage:
            n += self.unit_params(u, li)
            if u == "t":
                li += 1
        return n

    # Calibrated on 40 measured stages (moe-layout stage A and B, 4x A100, BF16, full
    # recompute, micro-batch 1): Megatron's "max allocated" = weights and optimizer
    # state + about 10.3 bytes per logit per output head (L and m) + 0.13 GB.
    # Mean error 2.8 percent, max 13.7 percent. Activations under full recompute at
    # micro-batch 1 were too small to fit separately.
    LOGIT_BYTES = 10.3
    OVERHEAD_GB = 0.13

    def stage_memory_gb(self, stage: list[str], stage_idx: int, pp: int, first_layer: int = 0,
                        act_scale: float = 1.0, overhead_gb: float | None = None) -> float:
        """Estimated peak 'max allocated' of one pipeline stage (1F1B, VPP 1), in GB."""
        params = self.stage_params(stage, first_layer)
        opt_shard = self.dp if self.distributed_optimizer else 1
        static = params * (2 + 4 + 12 / opt_shard)  # bf16 param, fp32 grad, fp32 master + Adam
        in_flight = min(pp - stage_idx, max(1, self.global_batch // (self.micro_batch * self.dp)))
        tok = self.seq_length * self.micro_batch
        n_layers = sum(1 for u in stage if u in "tm")
        # stored layer inputs under full recompute; full activations otherwise (rough, uncalibrated)
        per_layer = tok * self.hidden * 2 * (1 if self.full_recompute else 34)
        stored = n_layers * per_layer * in_flight if not self.full_recompute else 0.0
        logits = tok * self.vocab * self.LOGIT_BYTES * (stage.count("L") + stage.count("m"))
        oh = self.OVERHEAD_GB if overhead_gb is None else overhead_gb
        return (static + act_scale * stored + logits) / 1e9 + oh

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def from_json(path: str) -> ModelConfig:
    return ModelConfig(**json.load(open(path)))


_FLAG_MAP = {
    "--num-layers": ("num_layers", int), "--hidden-size": ("hidden", int),
    "--ffn-hidden-size": ("ffn_hidden", int), "--num-experts": ("num_experts", int),
    "--moe-ffn-hidden-size": ("moe_ffn_hidden", int), "--moe-router-topk": ("topk", int),
    "--moe-shared-expert-intermediate-size": ("shared_expert_hidden", int),
    "--num-attention-heads": ("heads", int), "--num-query-groups": ("query_groups", int),
    "--kv-channels": ("kv_channels", int), "--seq-length": ("seq_length", int),
    "--micro-batch-size": ("micro_batch", int), "--global-batch-size": ("global_batch", int),
    "--mtp-num-layers": ("mtp_layers", int),
}


def from_megatron_args(args: list[str] | str, vocab: int = 128256) -> ModelConfig:
    """Fill a ModelConfig from a Megatron command line (unknown flags ignored)."""
    if isinstance(args, str):
        args = shlex.split(args)
    cfg = ModelConfig(vocab=vocab, num_experts=0, mtp_layers=0, full_recompute=False)
    i = 0
    while i < len(args):
        a = args[i]
        if a in _FLAG_MAP and i + 1 < len(args):
            name, typ = _FLAG_MAP[a]
            setattr(cfg, name, typ(args[i + 1]))
            i += 2
            continue
        if a == "--untie-embeddings-and-output-weights":
            cfg.tied_embeddings = False
        elif a == "--recompute-granularity" and i + 1 < len(args):
            cfg.full_recompute = args[i + 1] == "full"
        i += 1
    return cfg
