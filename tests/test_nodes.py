import json
from pathlib import Path

import torch

from krea2_weighted.attn_patch import compose_attn_mask
from krea2_weighted.core import (
    GUIDE_DEFAULT_FOCUS,
    IDENTITY_DEFAULT_FOCUS,
    K2EDIT_DEFAULT_SYSTEM,
    compact_refs,
    encode_krea2,
    k2edit_template,
    prep_grounding_image,
    resolve_layout,
)
from krea2_weighted.tokenize import (
    QWEN_IM_END,
    QWEN_IM_START,
    QWEN_NL,
    QWEN_USER,
    qwen_vl_image_tokens,
)

NODE_IDS = {"Krea2Encode", "Krea2EncodeOptions", "Krea2MultiRef"}


class FakeAttn:
    pass


class FakeBlock:
    def __init__(self):
        self.attn = FakeAttn()


class FakeDiff:
    def __init__(self, n=4):
        self.blocks = [FakeBlock() for _ in range(n)]


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.diff = FakeDiff()
        self.patches = []

    def clone(self):
        other = FakeModel()
        other.model_options = {"transformer_options": dict(self.model_options.get("transformer_options", {}))}
        other.diff = self.diff
        other.patches = list(self.patches)
        return other

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diff

    def add_object_patch(self, path, fn):
        self.patches.append((path, fn))

    def set_model_attn1_patch(self, fn):
        to = self.model_options.setdefault("transformer_options", {})
        to.setdefault("patches", {}).setdefault("attn1_patch", []).append(fn)


def _weights(model):
    patches = model.model_options["transformer_options"].get("patches", {}).get("attn1_patch", [])
    assert patches, "no attn1_patch registered"
    return list(patches[-1].krea2_weights)


VISION_EXPAND = 4  # a 32x32 image is upscaled to 64x64 by min_pixels -> 4 tokens


class FakeClip:
    def __init__(self):
        self.last_images = None
        self.last_template = None

    def tokenize(self, text, images=None, llama_template=None, **kwargs):
        self.last_images = images
        self.last_template = llama_template
        parts = text.split()
        user = []
        for i, w in enumerate(parts):
            key = (" " + w) if (i > 0 or text.startswith(" ")) else w
            user.append(200 + (abs(hash(key)) % 50000))
        vision = []
        if images:
            for img in images:
                vision.extend(
                    [
                        151652,
                        {"type": "image", "data": img, "original_type": "image"},
                        151653,
                    ]
                )
        ids = [QWEN_IM_START, QWEN_USER, QWEN_NL] + vision + user + [QWEN_IM_END]
        return {"qwen3vl_4b": [[(i, 1.0) for i in ids]]}

    def encode_from_tokens_scheduled(self, tok):
        key = next(iter(tok))
        raw = [t[0] for t in tok[key][0]]
        visible = []
        for e in raw[3:]:  # drop im_start,user,nl
            if isinstance(e, dict) and e.get("type") == "image":
                img = e["data"]
                # Real Qwen3-VL expansion depends on image size (32 px cells).
                visible.extend([0] * qwen_vl_image_tokens(img.shape[1], img.shape[2]))
            else:
                visible.append(0)
        t = torch.zeros(1, len(visible), 8)
        return [[t, {}]]


def _rgb(h, w, val=0.2):
    return torch.full((1, h, w, 3), val)


def test_encode_scales_aux_only():
    model = FakeModel()
    clip = FakeClip()
    out_model, cond, debug = encode_krea2(
        clip,
        model,
        prompt="a woman",
        aux="cinematic grain",
        aux_strength=0.45,
    )
    assert out_model is not model
    tw = _weights(out_model)
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    assert "aux id" in debug


def test_encode_zero_strength_matches_standard_text_encode():
    model = FakeModel()
    clip = FakeClip()

    standard_clip = FakeClip()
    standard_tokens = standard_clip.tokenize("a woman")
    standard_cond = standard_clip.encode_from_tokens_scheduled(standard_tokens)

    out_model, cond, debug = encode_krea2(
        clip,
        model,
        prompt="a woman",
        aux="cinematic grain",
        aux_strength=0.0,
    )

    assert out_model is model
    assert torch.equal(cond[0][0], standard_cond[0][0])
    assert cond[0][0].shape == standard_cond[0][0].shape
    assert debug == "aux_strength=0.0; main only, no patch"


def test_encode_zero_strength_matches_grounded_empty_aux():
    image = _rgb(32, 32)

    zero_model = FakeModel()
    zero_clip = FakeClip()
    zero_out, zero_cond, zero_debug = encode_krea2(
        zero_clip,
        zero_model,
        prompt="recolor the car",
        aux="cinematic grain",
        aux_strength=0.0,
        scene=image,
    )

    empty_model = FakeModel()
    empty_clip = FakeClip()
    empty_out, empty_cond, _ = encode_krea2(
        empty_clip,
        empty_model,
        prompt="recolor the car",
        aux="",
        aux_strength=0.45,
        scene=image,
    )

    assert zero_out is zero_model
    assert empty_out is empty_model
    assert torch.equal(zero_cond[0][0], empty_cond[0][0])
    assert zero_cond[0][0].shape == empty_cond[0][0].shape
    assert len(zero_clip.last_images) == 1
    assert zero_clip.last_template == empty_clip.last_template
    assert "aux_strength=0.0; main only, no patch; grounded 1 image(s)" in zero_debug


def test_encode_grounded_one_image():
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = encode_krea2(
        clip,
        model,
        prompt="recolor the car",
        aux="cinematic grain",
        aux_strength=0.45,
        scene=img,
    )
    assert out_model is not model
    assert "grounded 1 image" in debug
    tw = _weights(out_model)
    assert tw
    assert all(abs(p[1] - 0.45) < 1e-6 for p in tw)
    n_vis_markers = 2
    vis_block = n_vis_markers + VISION_EXPAND
    aux_pos = [p[0] for p in tw]
    assert min(aux_pos) >= vis_block
    assert clip.last_images is not None and len(clip.last_images) == 1
    assert K2EDIT_DEFAULT_SYSTEM.split()[0] in (clip.last_template or "")


def test_encode_two_image_ordering():
    clip = FakeClip()
    scene = _rgb(32, 32, 0.1)
    subject = _rgb(32, 32, 0.9)
    encode_krea2(
        clip,
        FakeModel(),
        prompt="put the person in the room",
        aux="film still",
        aux_strength=0.4,
        scene=scene,
        identity=subject,
    )
    assert len(clip.last_images) == 2
    assert float(clip.last_images[0].reshape(-1)[0]) < float(clip.last_images[1].reshape(-1)[0])
    vis = "<|vision_start|><|image_pad|><|vision_end|>"
    assert clip.last_template.count(vis) == 2


def test_encode_aux_after_vision_expansion_only_aux_text():
    clip = FakeClip()
    img = _rgb(28, 28)
    out_model, cond, debug = encode_krea2(
        clip,
        FakeModel(),
        prompt="edit the jacket",
        aux="neon",
        aux_strength=0.3,
        scene=img,
    )
    tw = _weights(out_model)
    aux_pos = [p[0] for p in tw]
    vis_block = 2 + VISION_EXPAND
    assert min(aux_pos) >= vis_block
    assert max(aux_pos) < cond[0][0].shape[1]
    assert not any(p < vis_block for p in aux_pos)


def test_encode_empty_aux_still_grounded():
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = encode_krea2(
        clip,
        model,
        prompt="recolor the car",
        aux="",
        aux_strength=0.45,
        scene=img,
    )
    assert out_model is model
    assert "empty aux" in debug
    assert "grounded 1 image" in debug
    assert cond[0][0].shape[1] > 4
    assert clip.last_images is not None


def test_encode_aux_strength_one_grounded_no_patch():
    model = FakeModel()
    clip = FakeClip()
    img = _rgb(32, 32)
    out_model, cond, debug = encode_krea2(
        clip,
        model,
        prompt="recolor the car",
        aux="cinematic grain",
        aux_strength=1.0,
        scene=img,
    )
    assert out_model is model
    assert "aux_strength=1.0" in debug
    assert "grounded 1 image" in debug
    assert cond[0][0].shape[1] > 4


def test_k2edit_ref_mask_composes_with_key_bias():
    L = 8
    ref = torch.zeros(1, 1, L, L)
    ref[:, :, 5:, 2:4] = 1.5
    kb = torch.zeros(1, L)
    kb[:, 3] = 0.2
    out = compose_attn_mask(ref, kb)
    assert abs(float(out[0, 0, 5, 3]) - 1.7) < 1e-5
    assert abs(float(out[0, 0, 5, 2]) - 1.5) < 1e-5
    assert abs(float(out[0, 0, 0, 3]) - 0.2) < 1e-5
    assert torch.equal(compose_attn_mask(None, kb).reshape(-1), kb.reshape(-1))
    assert compose_attn_mask(ref, None) is ref


def test_prep_32_pixel_grid_and_template():
    big = _rgb(200, 100)
    out = prep_grounding_image(big, 64)
    h, w = out.shape[1], out.shape[2]
    assert h % 32 == 0 and w % 32 == 0
    assert max(h, w) <= 64
    # Images under the cap are snapped too (was: only when downscaling).
    small = prep_grounding_image(_rgb(70, 45), 768)
    assert small.shape[1] % 32 == 0 and small.shape[2] % 32 == 0
    native = prep_grounding_image(_rgb(70, 45), 0)
    assert native.shape[1:3] == (64, 32)
    tpl = k2edit_template(2, "")
    assert K2EDIT_DEFAULT_SYSTEM in tpl
    assert tpl.count("<|image_pad|>") == 2


def test_compact_refs_skips_holes():
    a = _rgb(8, 8, 0.1)
    b = _rgb(8, 8, 0.9)
    out = compact_refs([
        {"image": a, "strength": 1.0, "focus": "face"},
        {"image": None, "strength": 1.0, "focus": ""},
        {"image": b, "strength": 0.8, "focus": "pose"},
    ])
    assert len(out) == 2
    assert out[0]["focus"] == "face"
    assert out[1]["focus"] == "pose"
    assert abs(out[1]["strength"] - 0.8) < 1e-6


def test_k2edit_template_per_image_focus():
    tpl = k2edit_template(2, "", image_prompts=["focus on face", "focus on pose"])
    assert "focus on face" in tpl
    assert "focus on pose" in tpl
    assert tpl.index("focus on face") < tpl.index("<|image_pad|>")


def test_layout_scene_identity_guides():
    scene = _rgb(8, 8, 0.1)
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    hat = _rgb(8, 8, 0.4)
    dit, qwen, notes = resolve_layout(
        identity=person,
        scene=scene,
        refs=[{"image": dress}, {"image": hat}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert [r["role"] for r in qwen] == ["scene", "guide", "guide", "identity"]
    assert abs(dit[0]["boost"] - 4.0) < 1e-6
    assert qwen[1]["focus"] == GUIDE_DEFAULT_FOCUS
    assert qwen[-1]["focus"] == IDENTITY_DEFAULT_FOCUS
    assert "scene is Qwen-only" in "\n".join(notes)


def test_layout_multiref_slot1_becomes_identity_when_missing():
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    dit, qwen, notes = resolve_layout(
        refs=[{"image": person, "strength": 1.0}, {"image": dress}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert dit[0]["image"] is person
    assert abs(dit[0]["boost"] - 4.0) < 1e-6
    assert [r["role"] for r in qwen] == ["guide", "identity"]
    assert any("slot 1 used as identity" in n for n in notes)


def test_layout_multiref_does_not_replace_connected_identity():
    person = _rgb(8, 8, 0.9)
    dress = _rgb(8, 8, 0.3)
    dit, qwen, _ = resolve_layout(
        identity=person,
        refs=[{"image": dress}],
        ref_boost=4.0,
    )
    assert [r["role"] for r in dit] == ["identity"]
    assert dit[0]["image"] is person
    assert [r["role"] for r in qwen] == ["guide", "identity"]
    assert qwen[0]["image"] is dress


def test_encode_options_passthrough():
    opts = {"people_count_lock": "solo", "emphasis_mode": "both"}
    _, _, debug = encode_krea2(
        FakeClip(), FakeModel(), prompt="a woman", aux="photo", aux_strength=0.45, options=opts,
    )
    assert "people_count_lock=solo" in debug
    assert "people_lock injected" in debug


def test_encode_identity_is_last_qwen_image():
    person = _rgb(32, 32, 0.9)
    dress = _rgb(32, 32, 0.1)
    refs = compact_refs([{"image": dress}])
    clip = FakeClip()
    encode_krea2(
        clip, FakeModel(), prompt="restage this person wearing the dress", aux="",
        aux_strength=0.45, identity=person, refs=refs,
    )
    assert len(clip.last_images) == 2
    assert float(clip.last_images[-1].reshape(-1)[0]) > float(clip.last_images[0].reshape(-1)[0])
    assert GUIDE_DEFAULT_FOCUS.split(";")[0] in (clip.last_template or "")
    assert IDENTITY_DEFAULT_FOCUS.split(";")[0] in (clip.last_template or "")


def test_mixed_size_images_weight_exact_vision_tokens():
    """Guide (small) and identity (large) expand to different token counts.

    Old code split expansion evenly across images, so strengths landed on the wrong
    tokens and the aux span shifted. Each image's tokens must get its own strength.
    """
    guide = _rgb(256, 256, 0.1)    # 256 px -> 8x8 = 64 tokens (guide_grounding_px 384)
    person = _rgb(512, 512, 0.9)   # 512 px -> 16x16 = 256 tokens
    refs = [{"image": guide, "strength": 0.45}]
    clip = FakeClip()
    out_model, cond, debug = encode_krea2(
        clip, FakeModel(), prompt="put her in the dress", aux="soft film look",
        aux_strength=0.5, identity=person, refs=refs,
        options={"identity_strength": 1.0},
    )
    n_guide = qwen_vl_image_tokens(256, 256)
    n_person = qwen_vl_image_tokens(512, 512)
    assert (n_guide, n_person) == (64, 256)
    w = _weights(out_model)
    guide_pos = sorted(p for p, v, _ in w if abs(v - 0.45) < 1e-6)
    aux_pos = sorted(p for p, v, _ in w if abs(v - 0.5) < 1e-6)
    # cond layout: [label tokens] <vs> [guide x64] <ve> \n [label] <vs> [person x256] <ve> \n text
    assert len(guide_pos) == n_guide
    assert guide_pos == list(range(guide_pos[0], guide_pos[0] + n_guide))
    # identity strength 1.0 -> no person weights; aux tokens all after both vision blocks
    assert min(aux_pos) > guide_pos[-1] + n_person
    assert max(aux_pos) < cond[0][0].shape[1]
    # exactly the aux words ("soft film look" = 3 fake tokens) are weighted
    assert len(aux_pos) == 3


def test_slot1_identity_uses_identity_strength():
    person = _rgb(8, 8, 0.9)
    dit, qwen, notes = resolve_layout(
        refs=[{"image": person, "strength": 0.45}],
        opts={"identity_strength": 1.0},
    )
    assert qwen[-1]["role"] == "identity"
    assert abs(qwen[-1]["strength"] - 1.0) < 1e-6
    assert any("identity_strength" in n for n in notes)


def test_unknown_option_keys_are_ignored():
    dit, qwen, _ = resolve_layout(identity=_rgb(8, 8), opts={"ref_b_strength": 0.1})
    assert abs(qwen[-1]["strength"] - 1.0) < 1e-6


def test_example_workflows_use_only_core_and_pack_nodes():
    wf_dir = Path(__file__).resolve().parents[1] / "workflows"
    files = sorted(wf_dir.glob("*.json"))
    assert len(files) >= 4
    core_nodes = {
        "CLIPLoader", "UNETLoader", "VAELoader", "LoraLoaderModelOnly", "LoadImage",
        "EmptySD3LatentImage", "KSampler", "VAEDecode", "SaveImage", "Note",
        "ConditioningZeroOut",
    }
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "nodes" in data and "links" in data, f.name
        types = {n["type"] for n in data["nodes"]}
        assert "Krea2Encode" in types, f.name
        assert types <= core_nodes | NODE_IDS, (f.name, types - core_nodes - NODE_IDS)
        for n in data["nodes"]:
            if n["type"] == "Krea2Encode":
                names = {i["name"] for i in n.get("inputs", [])}
                assert not names & {"image", "image_b"}, f.name
