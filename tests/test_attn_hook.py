"""Run the token-weight patch through ComfyUI's real Krea 2 Attention module
(tiny random weights, CPU). Needs a ComfyUI checkout on the path."""
import pytest
import torch

try:
    from comfy.cli_args import args

    args.cpu = True
    import comfy.ops
    from comfy.ldm.krea2.model import Attention
except Exception:  # pragma: no cover
    pytest.skip("ComfyUI with Krea 2 not importable", allow_module_level=True)

from krea2_weighted.attn_patch import TokenWeights, make_attn1_patch  # noqa: E402


def _run(attn, x, patch, cou):
    to = {"block_index": 0, "cond_or_uncond": cou}
    if patch is not None:
        to["patches"] = {"attn1_patch": [patch]}
    return attn(x, None, None, transformer_options=to)


@pytest.mark.parametrize("vf,kb", [(0.3, 0.0), (1.0, 2.0)])
def test_hook_changes_only_cond_rows(vf, kb):
    torch.manual_seed(0)
    attn = Attention(32, 4, operations=comfy.ops.disable_weight_init)
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(2, 6, 32)
    base = _run(attn, x, None, [1, 0])
    patch = make_attn1_patch(TokenWeights([(1, vf, kb), (2, vf, kb)]), "cond", [0])
    out = _run(attn, x, patch, [1, 0])
    assert torch.allclose(out[0], base[0], atol=1e-5)      # uncond row untouched
    assert not torch.allclose(out[1], base[1], atol=1e-4)  # cond row weighted
    # block outside the range: identical
    patch_other = make_attn1_patch(TokenWeights([(1, vf, kb)]), "cond", [5])
    assert torch.allclose(_run(attn, x, patch_other, [1, 0]), base, atol=1e-6)
