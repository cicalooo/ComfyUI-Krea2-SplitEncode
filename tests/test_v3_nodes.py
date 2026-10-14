"""V3 schema tests. Need ComfyUI's comfy_api on sys.path (see conftest)."""
import asyncio

import pytest
import torch

pytest.importorskip("comfy_api.latest")

from krea2_weighted import nodes  # noqa: E402


def test_entrypoint_registers_three_nodes_with_stable_ids():
    ext = asyncio.run(nodes.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    ids = [n.define_schema().node_id for n in node_list]
    assert ids == ["Krea2Encode", "Krea2EncodeOptions", "Krea2MultiRef"]
    assert not hasattr(nodes, "NODE_CLASS_MAPPINGS")
    assert not hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS")


def test_encode_schema_inputs():
    schema = nodes.Krea2Encode.define_schema()
    names = [i.id for i in schema.inputs]
    assert names[:5] == ["clip", "model", "prompt", "aux", "aux_strength"]
    assert "image" not in names and "image_b" not in names
    rb = next(i for i in schema.inputs if i.id == "ref_boost")
    assert rb.max <= 10
    for i in schema.inputs:
        assert i.tooltip, i.id
    assert schema.display_name == "Krea2 Split Encode"


def test_v3_schema_validates():
    for n in nodes.NODES:
        s = n.define_schema()
        s.validate()
        n.GET_SCHEMA() if hasattr(n, "GET_SCHEMA") else None


def test_options_and_multiref_execute():
    out = nodes.Krea2EncodeOptions.execute(
        people_count_lock="solo", emphasis_mode="both", separator="newline",
        block_range="all", apply_to="cond", grounding_px=768, guide_grounding_px=384,
        system_prompt="", fit_mode="fit", scene_strength=1.0, identity_strength=0.9,
        scene_focus="", identity_focus="", scene_ref_boost=1.0,
    )
    opts = out.result[0]
    assert opts["people_count_lock"] == "solo" and opts["identity_strength"] == 0.9
    assert opts["identity_mask"] is None
    a = torch.zeros(1, 8, 8, 3)
    c = torch.ones(1, 8, 8, 3)
    refs = nodes.Krea2MultiRef.execute(image_1=a, image_3=c, strength_3=0.4).result[0]
    assert len(refs) == 2 and abs(refs[1]["strength"] - 0.4) < 1e-6
    assert "mask" not in refs[0]
