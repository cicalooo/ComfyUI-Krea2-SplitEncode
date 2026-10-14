"""ComfyUI V3 nodes: Krea2 Split Encode, Split Encode Options, Split Ref.

Node ids (``Krea2Encode``, ``Krea2EncodeOptions``, ``Krea2MultiRef``) are kept from
1.x so saved graphs still find the nodes. See README "Migrating" for renamed inputs.
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .core import (
    DEFAULT_OPTS,
    GUIDE_DEFAULT_GROUNDING_PX,
    GUIDE_DEFAULT_STRENGTH,
    MAX_REFS,
    REF_BOOST_MAX,
    compact_refs,
    encode_krea2,
)

CATEGORY = "Krea2/conditioning"

Krea2Options = io.Custom("KREA2_ENCODE_OPTS")
Krea2Refs = io.Custom("KREA2_REFS")


class Krea2Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Krea2Encode",
            display_name="Krea2 Split Encode",
            category=CATEGORY,
            description=(
                "Encode a prompt for Krea 2 with a separately weighted aux text (style, "
                "moodboard). Optionally ground on scene / identity / guide images, and with "
                "a VAE run Identity Edit (person as an in-context frame). Wire both MODEL "
                "and CONDITIONING to the sampler."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="Krea 2 text encoder (Qwen3-VL 4B)."),
                io.Model.Input(
                    "model",
                    tooltip="Krea 2 model. For Identity Edit, load the Identity Edit LoRA first.",
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip="Main prompt. Always encoded at full strength.",
                ),
                io.String.Input(
                    "aux", multiline=True, default="",
                    tooltip="Secondary text (style, moodboard). Weighted by aux_strength.",
                ),
                io.Float.Input(
                    "aux_strength", default=0.45, min=0.0, max=3.0, step=0.05,
                    tooltip=(
                        "Weight of the aux tokens only. 0 = leave aux out, 0.35-0.55 = "
                        "gentle, 1.0 = same as prompt, >1 = enforce. Stock (word:1.2) "
                        "syntax does nothing on Krea 2."
                    ),
                ),
                io.Image.Input(
                    "identity", optional=True,
                    tooltip=(
                        "Person photo. With vae connected it becomes the Identity Edit "
                        "frame (ref_boost applies)."
                    ),
                ),
                io.Image.Input(
                    "scene", optional=True,
                    tooltip=(
                        "Optional scene / layout image. With identity and no Split Ref: "
                        "frame 1 = scene, frame 2 = person. With Split Ref connected the "
                        "scene is Qwen-only."
                    ),
                ),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip=(
                        "Qwen Image VAE. Connecting it (with identity or scene) turns on "
                        "Identity Edit. Leave it off on a CFG>1 negative encode."
                    ),
                ),
                io.Float.Input(
                    "ref_boost", default=4.0, min=0.0, max=REF_BOOST_MAX, step=0.05,
                    optional=True,
                    tooltip=(
                        "Likeness pull toward the person frame (Identity Edit only). "
                        "1 = off, ~4 recommended."
                    ),
                ),
                io.Latent.Input(
                    "latent", optional=True,
                    tooltip=(
                        "The same empty latent you give KSampler. Tells the node the "
                        "output size so the VAE encode runs before sampling."
                    ),
                ),
                Krea2Options.Input(
                    "options", optional=True,
                    tooltip="From Split Encode Options. Disconnected = defaults.",
                ),
                Krea2Refs.Input(
                    "refs", optional=True,
                    tooltip="From Split Ref: clothing / object guides (Qwen only, not VAE frames).",
                ),
            ],
            outputs=[
                io.Model.Output("model", display_name="model", tooltip="Model with weighting / Identity Edit patches."),
                io.Conditioning.Output("conditioning", display_name="conditioning"),
                io.String.Output("debug", display_name="debug", tooltip="What ran, token counts, warnings."),
            ],
        )

    @classmethod
    def execute(cls, clip, model, prompt, aux, aux_strength, identity=None, scene=None,
                vae=None, ref_boost=4.0, latent=None, options=None, refs=None) -> io.NodeOutput:
        out_model, cond, debug = encode_krea2(
            clip, model, prompt, aux, aux_strength,
            identity=identity, scene=scene, vae=vae, ref_boost=ref_boost,
            latent=latent, options=options, refs=refs,
        )
        return io.NodeOutput(out_model, cond, debug)


class Krea2EncodeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        d = DEFAULT_OPTS
        return io.Schema(
            node_id="Krea2EncodeOptions",
            display_name="Krea2 Split Encode Options",
            category=CATEGORY,
            description="Advanced settings for Krea2 Split Encode. Disconnected = defaults.",
            inputs=[
                io.Combo.Input(
                    "people_count_lock", options=["off", "solo", "match_refs"],
                    default=d["people_count_lock"],
                    tooltip="Appends a fixed phrase to the prompt. solo = one person; "
                            "match_refs = same count as the reference images.",
                ),
                io.Combo.Input(
                    "emphasis_mode", options=["value_scale", "k_bias", "both"],
                    default=d["emphasis_mode"],
                    tooltip="How weights act in attention. value_scale (safest) scales values; "
                            "k_bias adds a key bias (stronger above 1); both does both.",
                ),
                io.Combo.Input(
                    "separator", options=["newline", "space", "comma"], default=d["separator"],
                    tooltip="How prompt and aux are joined before encoding.",
                ),
                io.String.Input(
                    "block_range", default=d["block_range"],
                    tooltip="DiT blocks the weighting applies to: 'all', '0-27', '4,8,12' "
                            "or '0-5,20-27'. Narrow it if the effect is too strong.",
                ),
                io.Combo.Input(
                    "apply_to", options=["cond", "uncond", "both"], default=d["apply_to"],
                    tooltip="Which CFG pass gets the weighting. At CFG 1 only cond runs.",
                ),
                io.Int.Input(
                    "grounding_px", default=d["grounding_px"], min=0, max=4096, step=32,
                    tooltip="Longest side for scene / identity images sent to Qwen "
                            "(snapped to 32 px). 0 = native size.",
                ),
                io.Int.Input(
                    "guide_grounding_px", default=GUIDE_DEFAULT_GROUNDING_PX, min=0, max=4096,
                    step=32,
                    tooltip="Longest side for Split Ref guides. Lower keeps clothes from "
                            "overpowering the face.",
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default="",
                    tooltip="Replaces the Krea Edit grounding system prompt. Empty = trained default.",
                ),
                io.Combo.Input(
                    "fit_mode", options=["fit", "crop (legacy)"], default=d["fit_mode"],
                    tooltip="How Identity Edit frames fit the output aspect. fit = training-"
                            "matched; crop (legacy) only for older Identity Edit weights.",
                ),
                io.Float.Input(
                    "scene_strength", default=1.0, min=0.0, max=3.0, step=0.05,
                    tooltip="Qwen vision-token strength for the scene image.",
                ),
                io.Float.Input(
                    "identity_strength", default=1.0, min=0.0, max=3.0, step=0.05,
                    tooltip="Qwen vision-token strength for the identity image.",
                ),
                io.String.Input(
                    "scene_focus", multiline=True, default="",
                    tooltip="Label placed before the scene image. Empty = 'scene and layout only'.",
                ),
                io.String.Input(
                    "identity_focus", multiline=True, default="",
                    tooltip="Label placed before the identity image. Empty = 'preserve this face'.",
                ),
                io.Float.Input(
                    "scene_ref_boost", default=1.0, min=0.0, max=REF_BOOST_MAX, step=0.05,
                    tooltip="Likeness pull toward the scene frame (Identity Edit, no guides). "
                            "1 = off.",
                ),
                io.Mask.Input(
                    "identity_mask", optional=True,
                    tooltip="Limit ref_boost to a region of the identity image (e.g. the face).",
                ),
                io.Mask.Input(
                    "scene_mask", optional=True,
                    tooltip="Limit scene_ref_boost to a region of the scene image.",
                ),
            ],
            outputs=[Krea2Options.Output("options", display_name="options")],
        )

    @classmethod
    def execute(cls, identity_mask=None, scene_mask=None, **kwargs) -> io.NodeOutput:
        opts = dict(DEFAULT_OPTS)
        opts.update({k: v for k, v in kwargs.items() if k in DEFAULT_OPTS})
        opts["identity_mask"] = identity_mask
        opts["scene_mask"] = scene_mask
        return io.NodeOutput(opts)


class Krea2MultiRef(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = []
        for i in range(1, MAX_REFS + 1):
            inputs += [
                io.Image.Input(
                    "image_{}".format(i), optional=True,
                    tooltip="Guide {}: clothing / object photo for Qwen placement. Not a VAE "
                            "frame. Empty slots are skipped.".format(i),
                ),
                io.Float.Input(
                    "strength_{}".format(i), default=GUIDE_DEFAULT_STRENGTH, min=0.0, max=3.0,
                    step=0.05, optional=True,
                    tooltip="Qwen vision-token strength for guide {}. ~0.45 recommended; "
                            "near 1 competes with the face.".format(i),
                ),
                io.String.Input(
                    "focus_{}".format(i), multiline=True, default="", optional=True,
                    tooltip="What to take from this image. Empty = clothing/object only, "
                            "do not copy any person.",
                ),
            ]
        return io.Schema(
            node_id="Krea2MultiRef",
            display_name="Krea2 Split Ref",
            category=CATEGORY,
            description=(
                "Up to 5 clothing / object guides for Krea2 Split Encode. They steer "
                "placement through Qwen only and never take the person's VAE frame."
            ),
            inputs=inputs,
            outputs=[Krea2Refs.Output("refs", display_name="refs")],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        slots = [
            {
                "image": kwargs.get("image_{}".format(i)),
                "strength": kwargs.get("strength_{}".format(i), GUIDE_DEFAULT_STRENGTH),
                "focus": kwargs.get("focus_{}".format(i), ""),
            }
            for i in range(1, MAX_REFS + 1)
        ]
        return io.NodeOutput(compact_refs(slots))


NODES = [Krea2Encode, Krea2EncodeOptions, Krea2MultiRef]


class Krea2SplitEncodeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> Krea2SplitEncodeExtension:
    return Krea2SplitEncodeExtension()
