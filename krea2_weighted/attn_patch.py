"""Krea 2 per-token V scale and key bias, applied through ComfyUI's ``attn1_patch`` hook.

Native ``comfy.ldm.krea2.model.Attention.forward`` calls every ``attn1_patch`` with
``(q, k, v, pe=, attn_mask=, extra_options=)`` after QK-norm and before RoPE, then runs
``optimized_attention_masked`` with whatever mask the patch returns. Using the hook keeps
the user's optimized attention backend (no ``attention_pytorch`` fallback) and avoids
replacing ``attn.forward`` wholesale.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

WEIGHTS_KEY = "krea2_token_weights"
APPLY_TO_KEY = "krea2_weight_apply_to"


def compose_attn_mask(existing, key_bias):
    """Add a key bias onto an existing additive / bool attention mask.

    Identity Edit ``ref_boost`` passes a ``(1, 1, L, L)`` additive mask. Key bias is
    ``(L,)``, ``(B, L)`` or ``(B, 1, 1, L)``. They add (broadcast) instead of replacing.
    """
    if key_bias is None:
        return existing
    if existing is None:
        kb = key_bias
        while kb.ndim < 4:
            kb = kb.unsqueeze(-2) if kb.ndim >= 2 else kb.unsqueeze(0)
        return kb

    e = existing
    if e.dtype == torch.bool:
        fill = torch.finfo(key_bias.dtype).min
        e = torch.zeros(e.shape, dtype=key_bias.dtype, device=e.device)
        e = e.masked_fill(~existing.bool(), fill)

    kb = key_bias.to(device=e.device, dtype=e.dtype)
    if kb.ndim == 1:
        kb = kb.unsqueeze(0)
    while kb.ndim < e.ndim:
        kb = kb.unsqueeze(-2)
    return e + kb


def rows_to_apply(transformer_options: dict, batch: int) -> Optional[torch.Tensor]:
    """Bool mask over batch rows that should be weighted, or ``None`` for all rows.

    ComfyUI batches cond and uncond together; ``cond_or_uncond`` lists one entry per
    equal-sized chunk (0 = cond, 1 = uncond). Only the requested chunks are weighted.
    """
    apply_to = transformer_options.get(APPLY_TO_KEY, "cond")
    cou = transformer_options.get("cond_or_uncond")
    if apply_to == "both" or not cou:
        return None
    want = 1 if apply_to == "uncond" else 0
    n = len(cou)
    if batch % n != 0:
        # Unexpected layout: fall back to whole-batch decision (old behaviour).
        return None if want in cou else torch.zeros(batch, dtype=torch.bool)
    per = batch // n
    flags = torch.tensor([c == want for c in cou], dtype=torch.bool)
    return flags.repeat_interleave(per)


def _should_apply(transformer_options: dict) -> bool:
    """True when at least one batch chunk is selected by ``apply_to``."""
    apply_to = transformer_options.get(APPLY_TO_KEY, "cond")
    cou = transformer_options.get("cond_or_uncond")
    if apply_to == "both" or not cou:
        return True
    return (1 if apply_to == "uncond" else 0) in cou


class TokenWeights:
    """Precomputed index / factor tensors for weighted cond positions.

    Built once at node time from ``(pos, v_factor, k_bias)`` pairs, so the per-step
    work is a couple of vectorised index ops instead of Python loops.
    """

    def __init__(self, pairs: Sequence[tuple[int, float, float]]):
        merged: dict[int, tuple[float, float]] = {}
        for pos, vf, kb in pairs:
            merged[int(pos)] = (float(vf), float(kb))
        idx = sorted(merged)
        self.pairs = [(i, merged[i][0], merged[i][1]) for i in idx]
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.v = torch.tensor([merged[i][0] for i in idx], dtype=torch.float32)
        self.kb = torch.tensor([merged[i][1] for i in idx], dtype=torch.float32)
        self.has_v = bool((self.v != 1.0).any()) if idx else False
        self.has_kb = bool((self.kb != 0.0).any()) if idx else False
        self._dev: dict = {}

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def __bool__(self):
        return bool(self.pairs)

    def on(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = (self.idx.to(device), self.v.to(device), self.kb.to(device))
        return self._dev[key]

    def scale_vector(self, length: int, device, dtype) -> Optional[torch.Tensor]:
        if not self.has_v:
            return None
        idx, v, _ = self.on(device)
        keep = idx < length
        s = torch.ones(length, device=device, dtype=dtype)
        s[idx[keep]] = v[keep].to(dtype)
        return s

    def bias_vector(self, length: int, device, dtype) -> Optional[torch.Tensor]:
        if not self.has_kb:
            return None
        idx, _, kb = self.on(device)
        keep = idx < length
        b = torch.zeros(length, device=device, dtype=dtype)
        b[idx[keep]] = kb[keep].to(dtype)
        return b


def apply_token_weights(q, k, v, mask, weights: TokenWeights, transformer_options: dict):
    """Return ``(v, mask)`` with weights applied to the selected batch rows."""
    if not weights or not _should_apply(transformer_options):
        return v, mask
    B, _, L, _ = v.shape
    rows = rows_to_apply(transformer_options, B)
    if rows is not None:
        rows = rows.to(v.device)
        if not bool(rows.any()):
            return v, mask
    scale = weights.scale_vector(L, v.device, v.dtype)
    if scale is not None:
        s = scale.view(1, 1, L, 1)
        if rows is not None:
            s = torch.where(rows.view(B, 1, 1, 1), s, torch.ones_like(s))
        v = v * s
    bias = weights.bias_vector(k.shape[2], q.device, q.dtype)
    if bias is not None:
        kb = bias.view(1, 1, 1, -1).expand(B, 1, 1, -1)
        if rows is not None:
            kb = kb * rows.view(B, 1, 1, 1).to(kb.dtype)
        mask = compose_attn_mask(mask, kb)
    return v, mask


def make_attn1_patch(weights: TokenWeights, apply_to: str, block_indices: Sequence[int]):
    """Build an ``attn1_patch`` callable for ``ModelPatcher.set_model_attn1_patch``."""
    blocks = frozenset(int(i) for i in block_indices)

    def krea2_token_weight_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        opts = extra_options or {}
        if opts.get("block_index") not in blocks:
            return {}
        if opts.get("block_type", "single") != "single":
            return {}
        local = dict(opts)
        local[APPLY_TO_KEY] = apply_to
        v2, mask2 = apply_token_weights(q, k, v, attn_mask, weights, local)
        return {"v": v2, "attn_mask": mask2}

    krea2_token_weight_patch.krea2_weights = weights
    return krea2_token_weight_patch
