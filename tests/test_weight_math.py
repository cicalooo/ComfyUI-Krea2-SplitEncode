import torch

from krea2_weighted.attn_patch import (
    APPLY_TO_KEY,
    TokenWeights,
    _should_apply,
    apply_token_weights,
    compose_attn_mask,
    make_attn1_patch,
    rows_to_apply,
)
from krea2_weighted.tokenize import mix_factors


def test_mix_factors():
    assert mix_factors(0.45, "value_scale") == (0.45, 0.0)
    assert mix_factors(1.0, "value_scale") == (1.0, 0.0)
    assert mix_factors(0.5, "k_bias") == (0.5, 0.0)
    assert mix_factors(1.5, "k_bias") == (1.0, 1.5)
    assert mix_factors(1.5, "both") == (1.375, 1.5)


def test_compose_preserves_ref_boost():
    ref = torch.zeros(1, 1, 4, 4)
    ref[:, :, 2:, 1:3] = 2.0
    kb = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
    out = compose_attn_mask(ref, kb)
    assert float(out[0, 0, 2, 1]) == 2.5
    assert float(out[0, 0, 2, 2]) == 2.0


def test_compose_per_row_bias_broadcasts():
    ref = torch.zeros(1, 1, 3, 3)
    kb = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3).repeat(2, 1, 1, 1)
    kb[1] = 0
    out = compose_attn_mask(ref, kb)
    assert out.shape == (2, 1, 3, 3)
    assert float(out[0, 0, 2, 1]) == 1.0 and float(out[1, 0, 2, 1]) == 0.0


def test_apply_to_chunk_selection():
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [0]}) is True
    assert _should_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1]}) is False
    assert _should_apply({APPLY_TO_KEY: "uncond", "cond_or_uncond": [1]}) is True
    assert _should_apply({APPLY_TO_KEY: "both", "cond_or_uncond": [1]}) is True
    assert _should_apply({}) is True
    rows = rows_to_apply({APPLY_TO_KEY: "cond", "cond_or_uncond": [1, 0]}, 4)
    assert rows.tolist() == [False, False, True, True]
    assert rows_to_apply({APPLY_TO_KEY: "both", "cond_or_uncond": [1, 0]}, 4) is None


def _qkv(B=2, H=2, L=5, D=4):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, H, L, D, generator=g), torch.randn(B, H, L, D, generator=g),
            torch.randn(B, H, L, D, generator=g))


def test_v_scale_only_on_cond_rows_when_batched_with_uncond():
    q, k, v = _qkv()
    w = TokenWeights([(1, 0.5, 0.0), (3, 0.25, 0.0)])
    to = {APPLY_TO_KEY: "cond", "cond_or_uncond": [1, 0]}  # row 0 uncond, row 1 cond
    v2, mask = apply_token_weights(q, k, v, None, w, to)
    assert mask is None
    assert torch.equal(v2[0], v[0])  # uncond untouched (was weighted before 2.0)
    assert torch.allclose(v2[1, :, 1], v[1, :, 1] * 0.5)
    assert torch.allclose(v2[1, :, 3], v[1, :, 3] * 0.25)
    assert torch.equal(v2[1, :, 0], v[1, :, 0])


def test_key_bias_only_on_selected_rows_and_composes():
    q, k, v = _qkv()
    w = TokenWeights([(2, 1.0, 1.5)])
    ref = torch.zeros(1, 1, 5, 5)
    ref[..., 4] = 0.7
    v2, mask = apply_token_weights(q, k, v, ref, w, {APPLY_TO_KEY: "cond", "cond_or_uncond": [1, 0]})
    assert torch.equal(v2, v)
    assert mask.shape == (2, 1, 5, 5)
    assert float(mask[0, 0, 0, 2]) == 0.0 and float(mask[1, 0, 0, 2]) == 1.5
    assert abs(float(mask[1, 0, 3, 4]) - 0.7) < 1e-6


def test_attn1_patch_respects_block_range():
    q, k, v = _qkv(B=1)
    patch = make_attn1_patch(TokenWeights([(0, 0.5, 0.0)]), "cond", [1])
    assert patch(q, k, v, extra_options={"block_index": 0}) == {}
    out = patch(q, k, v, extra_options={"block_index": 1, "cond_or_uncond": [0]})
    assert torch.allclose(out["v"][:, :, 0], v[:, :, 0] * 0.5)
    # text-fusion attention has no block_index -> untouched
    assert patch(q, k, v, extra_options={}) == {}


def test_token_weights_vectors_ignore_out_of_range():
    w = TokenWeights([(1, 0.5, 0.0), (99, 0.5, 0.0)])
    s = w.scale_vector(4, "cpu", torch.float32)
    assert s.tolist() == [1.0, 0.5, 1.0, 1.0]
    assert w.bias_vector(4, "cpu", torch.float32) is None
