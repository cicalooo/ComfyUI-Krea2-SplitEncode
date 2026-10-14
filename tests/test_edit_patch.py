"""Identity Edit wiring tests (CPU, fake model/VAE). Need ComfyUI's comfy package."""
import pytest
import torch

try:
    from comfy.cli_args import args

    args.cpu = True
    import comfy.patcher_extension  # noqa: F401
except Exception:  # pragma: no cover - ComfyUI not on path
    pytest.skip("ComfyUI not importable", allow_module_level=True)

from krea2_weighted import edit_patch  # noqa: E402
from krea2_weighted.core import encode_krea2  # noqa: E402


class FakeDiT:
    def __init__(self):
        for a in ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp", "pe_embedder",
                  "channels", "patch", "tdim", "_unpack_context"):
            setattr(self, a, object())
        self.blocks = [object()] * 4


class FakeBase:
    def __init__(self):
        self.diffusion_model = FakeDiT()

    def process_latent_in(self, x):
        return x


class FakePatcher:
    def __init__(self):
        self.model = FakeBase()
        self.model_options = {"transformer_options": {}}

    def clone(self):
        c = FakePatcher()
        c.model = self.model
        to = dict(self.model_options["transformer_options"])
        to["wrappers"] = {k: dict(v) for k, v in to.get("wrappers", {}).items()}
        c.model_options = {"transformer_options": to}
        return c

    def get_model_object(self, name):
        return self.model.diffusion_model

    def set_model_attn1_patch(self, fn):
        self.model_options["transformer_options"].setdefault("patches", {}) \
            .setdefault("attn1_patch", []).append(fn)


class FakeVAE:
    def __init__(self):
        self.calls = []

    def encode(self, img):
        self.calls.append(tuple(img.shape))
        b, h, w, _ = img.shape
        return torch.zeros(b, 16, h // 8, w // 8)


class FakeClip:
    def tokenize(self, text, images=None, llama_template=None, **kw):
        return {"qwen3vl_4b": [[(151644, 1.0), (872, 1.0), (198, 1.0), (300, 1.0), (151645, 1.0)]]}

    def encode_from_tokens_scheduled(self, tok):
        return [[torch.zeros(1, 2, 8), {}]]


def test_identity_edit_encodes_only_at_target_size():
    vae = FakeVAE()
    person = torch.rand(1, 1536, 1024, 3)  # large photo: no native-res encode any more
    latent = {"samples": torch.zeros(1, 16, 128, 96)}
    out_model, _, debug = encode_krea2(
        FakeClip(), FakePatcher(), "a portrait", "", 0.45,
        identity=person, vae=vae, latent=latent,
    )
    assert "identity_edit=ON" in debug
    # exactly one encode, fitted inside the 1024x768 target (fit mode), never native 1536x1024
    assert len(vae.calls) == 1
    _, h, w, _ = vae.calls[0]
    assert h <= 1024 and w <= 768 and (h, w) != (1536, 1024)
    wrappers = out_model.model_options["transformer_options"]["wrappers"]
    assert any("krea2_edit" in v for v in wrappers.values())


def test_second_identity_edit_on_same_model_refuses():
    vae = FakeVAE()
    img = torch.rand(1, 64, 64, 3)
    m1 = edit_patch.apply_identity_edit(FakePatcher(), vae=vae, source_images=[img], ref_boosts=[4.0])
    with pytest.raises(RuntimeError, match="already carries an Identity Edit"):
        edit_patch.apply_identity_edit(m1, vae=vae, source_images=[img], ref_boosts=[4.0])


def test_mask_signature_is_content_based():
    a = torch.zeros(1, 8, 8)
    b = torch.zeros(1, 8, 8)
    c = torch.zeros(1, 8, 8)
    c[0, 1, 1] = 1
    assert edit_patch.mask_signature(a) == edit_patch.mask_signature(b)
    assert edit_patch.mask_signature(a) != edit_patch.mask_signature(c)
    assert edit_patch.mask_signature(None) is None
