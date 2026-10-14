"""Krea2 Split Encode core logic (no ComfyUI node API imports; unit-testable)."""

from __future__ import annotations

import logging

import torch

from .attn_patch import TokenWeights, make_attn1_patch
from .tokenize import (
    QWEN_VL_FACTOR,
    aux_id_span_grounded,
    aux_id_span_in_combined,
    clip_token_ids,
    mix_factors,
    parse_block_range,
    qwen_vl_image_tokens,
    slice_to_cond_pairs,
    slice_to_cond_pairs_after_vision,
    token_ids_from_tok,
    user_span_from_ids,
    vision_cond_pairs_per_image,
)

log = logging.getLogger("krea2_split_encode")

MAX_REFS = 5
REF_BOOST_MAX = 10.0
GUIDE_DEFAULT_STRENGTH = 0.45
GUIDE_DEFAULT_GROUNDING_PX = 384
GUIDE_DEFAULT_FOCUS = "clothing or object only; do not copy any person in this image"
SCENE_DEFAULT_FOCUS = "scene and layout only; do not copy any face from this image"
IDENTITY_DEFAULT_FOCUS = "this is the only person; preserve this face and identity exactly"

K2EDIT_DEFAULT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)

PEOPLE_LOCK_PHRASES = {
    "off": "",
    "solo": (
        "exactly one person, solo subject, single subject only, "
        "do not add extra people, no additional figures"
    ),
    "match_refs": (
        "keep the same number of people as in the reference image(s), "
        "do not add extra people or duplicate the subject"
    ),
}

SEPARATORS = {"newline": "\n", "space": " ", "comma": ", "}

DEFAULT_OPTS = {
    "people_count_lock": "off",
    "emphasis_mode": "value_scale",
    "separator": "newline",
    "block_range": "all",
    "apply_to": "cond",
    "grounding_px": 768,
    "guide_grounding_px": GUIDE_DEFAULT_GROUNDING_PX,
    "system_prompt": "",
    "fit_mode": "fit",
    "scene_strength": 1.0,
    "identity_strength": 1.0,
    "scene_focus": "",
    "identity_focus": "",
    "scene_ref_boost": 1.0,
    "identity_mask": None,
    "scene_mask": None,
}


def patch_model(model, pairs, apply_to, block_range):
    """Clone ``model`` and register the token-weight ``attn1_patch`` on chosen blocks."""
    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "Krea2 Split Encode: model has no diffusion_model.blocks "
            "(expected a Krea 2 SingleStreamDiT)."
        )
    indices = parse_block_range(block_range, len(blocks))
    weights = TokenWeights(pairs)
    model_clone.set_model_attn1_patch(make_attn1_patch(weights, apply_to, indices))
    return model_clone, indices


def k2edit_template(nimg, system_prompt="", image_prompts=None):
    """Qwen3-VL chat template with optional per-image focus text."""
    sp = (system_prompt or "").strip() or K2EDIT_DEFAULT_SYSTEM
    prompts = list(image_prompts or [])
    chunks = []
    for i in range(nimg):
        label = (prompts[i] or "").strip() if i < len(prompts) else ""
        if label:
            chunks.append(label + "\n")
        chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
        if i < nimg - 1:
            chunks.append("\n")
    vis = "".join(chunks)
    return (
        "<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
        + vis + "\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


def _snap(v):
    return max(QWEN_VL_FACTOR, int(round(v / QWEN_VL_FACTOR)) * QWEN_VL_FACTOR)


def prep_grounding_image(image, grounding_px):
    """Cap the longest side at ``grounding_px`` (0 = no cap) and always snap both sides
    to the Qwen3-VL 32-pixel grid (patch 16 x merge 2), so ComfyUI's processor does not
    resample again and the vision token count is known exactly."""
    samples = image.movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    s = 1.0
    if grounding_px and max(h, w) > grounding_px:
        s = grounding_px / max(h, w)
    nh, nw = _snap(h * s), _snap(w * s)
    if grounding_px:
        cap = max(QWEN_VL_FACTOR, grounding_px // QWEN_VL_FACTOR * QWEN_VL_FACTOR)
        nh, nw = min(nh, cap), min(nw, cap)
    if (nh, nw) != (h, w):
        try:
            import comfy.utils

            samples = comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")
        except ImportError:
            samples = torch.nn.functional.interpolate(samples.float(), size=(nh, nw), mode="area")
    return samples.movedim(1, -1)[:, :, :, :3]


def merge_opts(options):
    out = dict(DEFAULT_OPTS)
    if options:
        unknown = set(options) - set(DEFAULT_OPTS)
        if unknown:
            log.warning("Krea2 Split Encode: ignoring unknown options %s", sorted(unknown))
        out.update({k: v for k, v in options.items() if k in DEFAULT_OPTS})
    return out


def compact_refs(slots):
    """Drop empty image slots; preserve order. Cap at MAX_REFS."""
    out = []
    for slot in slots:
        img = slot.get("image")
        if img is None:
            continue
        strength = slot.get("strength")
        if strength is None:
            strength = GUIDE_DEFAULT_STRENGTH
        out.append({"image": img, "strength": float(strength), "focus": slot.get("focus") or ""})
        if len(out) >= MAX_REFS:
            break
    return out


def _qwen_entry(image, role, strength, focus, grounding_px):
    return {"image": image, "role": role, "strength": float(strength),
            "focus": focus or "", "grounding_px": grounding_px}


def _dit_entry(image, role, boost, mask):
    return {"image": image, "role": role, "boost": float(boost), "mask": mask}


def resolve_layout(identity=None, scene=None, refs=None, opts=None, ref_boost=4.0):
    """Split appearance (DiT frames, max 2) from placement (Qwen, all images).

    Split Ref guides are Qwen-only, unless no identity is connected: then guide 1
    becomes the identity (using the identity strength/focus options) and the rest
    stay guides.
    """
    opts = merge_opts(opts)
    ident = identity
    scn = scene
    guides = compact_refs(refs) if refs is not None else []
    notes = []
    ident_from_guide = None

    if ident is None and guides:
        ident_from_guide = guides[0]
        ident = ident_from_guide["image"]
        guides = guides[1:]
        notes.append(
            "Split Ref slot 1 used as identity (no identity connected); "
            "identity_strength={:.2f} applies, not the guide strength".format(
                float(opts["identity_strength"]))
        )

    grounding_px = opts["grounding_px"]
    guide_px = opts["guide_grounding_px"]

    # With guides connected the scene stays Qwen-only: a scene VAE frame fitted to the
    # output grid overwrites the face. DiT is then the person only.
    scene_on_dit = scn is not None and not guides
    dit, qwen = [], []
    if scn is not None:
        scene_focus = (opts["scene_focus"] or "").strip() or SCENE_DEFAULT_FOCUS
        if scene_on_dit:
            dit.append(_dit_entry(scn, "scene", opts["scene_ref_boost"], opts["scene_mask"]))
        qwen.append(_qwen_entry(scn, "scene", opts["scene_strength"], scene_focus, grounding_px))
    for g in guides:
        focus = (g.get("focus") or "").strip() or GUIDE_DEFAULT_FOCUS
        qwen.append(_qwen_entry(g["image"], "guide", g["strength"], focus, guide_px))
    if ident is not None:
        ident_focus = (opts["identity_focus"] or "").strip()
        if not ident_focus and ident_from_guide is not None:
            ident_focus = (ident_from_guide.get("focus") or "").strip()
        ident_focus = ident_focus or IDENTITY_DEFAULT_FOCUS
        dit.append(_dit_entry(ident, "identity", ref_boost, opts["identity_mask"]))
        qwen.append(_qwen_entry(ident, "identity", opts["identity_strength"], ident_focus,
                                grounding_px))

    if dit:
        notes.append("DiT refs={} ({})  boosts={}".format(
            len(dit), ", ".join(r["role"] for r in dit), [r["boost"] for r in dit]))
    notes.append("Qwen refs={} ({})".format(
        len(qwen), ", ".join(r["role"] for r in qwen) if qwen else "none"))
    if any(r["role"] == "guide" for r in qwen):
        notes.append("guides are Qwen-only (not VAE frames); identity stays on the person DiT ref")
        if scn is not None and not scene_on_dit:
            notes.append("scene is Qwen-only while guides are connected "
                         "(a scene VAE frame fitted to the output overwrites the face)")
        hot = [r["strength"] for r in qwen if r["role"] == "guide" and r["strength"] >= 0.85]
        if hot:
            notes.append("WARNING: guide vision strength is {:.2f}. Set Split Ref strength "
                         "to ~0.45 or clothing photos compete with the face.".format(hot[0]))
    return dit, qwen, notes


def _apply_identity_edit(model, dit_refs, vae, target_latent, fit_mode):
    from .edit_patch import apply_identity_edit

    return apply_identity_edit(
        model,
        vae=vae,
        source_images=[r["image"] for r in dit_refs],
        ref_boosts=[r["boost"] for r in dit_refs],
        ref_boost_masks=[r["mask"] for r in dit_refs],
        fit_mode=fit_mode,
        target_latent=target_latent,
    )


def _short(s, n=72):
    s = s or ""
    return s[:n] + ("…" if len(s) > n else "")


def encode_krea2(
    clip,
    model,
    prompt,
    aux,
    aux_strength,
    identity=None,
    scene=None,
    vae=None,
    ref_boost=4.0,
    latent=None,
    options=None,
    refs=None,
):
    opts = merge_opts(options)
    separator = opts["separator"]
    emphasis_mode = opts["emphasis_mode"]
    block_range = opts["block_range"]
    apply_to = opts["apply_to"]
    system_prompt = opts["system_prompt"]
    fit_mode = opts["fit_mode"]
    people_count_lock = opts["people_count_lock"]

    dit_refs, qwen_refs, layout_notes = resolve_layout(
        identity=identity,
        scene=scene,
        refs=refs,
        opts=opts,
        ref_boost=ref_boost,
    )

    edit_notes = list(layout_notes)
    identity_on = vae is not None and bool(dit_refs)
    if identity_on:
        model = _apply_identity_edit(model, dit_refs, vae, latent, fit_mode)
        edit_notes.append("identity_edit=ON  fit={}".format(fit_mode))
        if latent is None:
            edit_notes.append(
                "NOTE: connect latent (KSampler empty latent) to pre-encode at target size"
            )
        edit_notes.append("NOTE: Identity Edit LoRA must already be on MODEL (strength 1.0)")
    elif qwen_refs:
        edit_notes.append(
            "identity_edit=OFF — Qwen grounding only; connect vae + identity for "
            "in-context source tokens"
        )
    else:
        edit_notes.append("text-only mix")

    text_main = (prompt or "").strip()
    text_aux = (aux or "").strip()
    if identity_on and text_aux and len(text_aux) > 80 and float(aux_strength) > 0:
        edit_notes.append(
            "WARNING: aux is {} chars at strength {:.2f}. Standard Krea Edit has no "
            "moodboard; set aux_strength=0 or clear aux if the face is drifting.".format(
                len(text_aux), float(aux_strength)
            )
        )
    lock_phrase = PEOPLE_LOCK_PHRASES.get(people_count_lock or "off", "")
    main = text_main
    if lock_phrase:
        main = (main + "\n" + lock_phrase).strip() if main else lock_phrase

    sep = SEPARATORS[separator]
    imgs = [
        prep_grounding_image(r["image"], r["grounding_px"])
        for r in qwen_refs
    ]
    grounded = bool(imgs)
    img_token_counts = [qwen_vl_image_tokens(im.shape[1], im.shape[2]) for im in imgs]
    tok_kw = {}
    if grounded:
        tok_kw = {
            "images": imgs,
            "llama_template": k2edit_template(
                len(imgs),
                system_prompt,
                image_prompts=[r["focus"] for r in qwen_refs],
            ),
        }

    img_strengths = [r["strength"] for r in qwen_refs]

    main_only_reason = None
    if not text_aux:
        main_only_reason = "empty aux"
    elif float(aux_strength) == 0.0:
        main_only_reason = "aux_strength=0.0"

    need_vision_weights = grounded and any(abs(s - 1.0) > 1e-6 for s in img_strengths)
    v_factor_early, k_bias_early = mix_factors(aux_strength, emphasis_mode)
    aux_is_noop = abs(v_factor_early - 1.0) < 1e-6 and abs(k_bias_early) < 1e-6

    def _debug_report(lines, patched_blocks=None, pair_count=0):
        header = [
            "=== Krea2 Split Encode ===",
            "prompt: {!r}".format(_short(text_main)),
            "aux:    {!r}".format(_short(text_aux)),
            "aux_strength={:.2f}  mode={}  people_count_lock={}".format(
                float(aux_strength), emphasis_mode, people_count_lock
            ),
        ]
        if lock_phrase:
            header.append("people_lock injected into main @1.0")
        header.extend(edit_notes)
        if grounded:
            header.append(
                "vision_strengths={}  roles={}".format(
                    img_strengths, [r["role"] for r in qwen_refs]
                )
            )
        else:
            header.append("refs=0 (text-only)")
        header.extend(lines)
        if patched_blocks is not None:
            header.append("patched blocks: {}".format(patched_blocks))
            header.append("total weighted cond tokens: {}".format(pair_count))
        return "\n".join(header)

    simple = (
        options is None
        and refs is None
        and identity is None
        and scene is None
        and people_count_lock == "off"
        and not identity_on
        and not need_vision_weights
    )

    if main_only_reason is not None and not need_vision_weights:
        tok = clip.tokenize(main, **tok_kw)
        cond = clip.encode_from_tokens_scheduled(tok)
        extra = []
        if grounded:
            extra.append("grounded {} image(s)".format(len(imgs)))
        msg = "{}; main only, no patch".format(main_only_reason)
        if extra:
            msg += "; " + "; ".join(extra)
        if simple:
            return (model, cond, msg)
        return (model, cond, _debug_report([msg]))

    if aux_is_noop and not need_vision_weights and main_only_reason is None:
        combined = main + sep + text_aux if main else text_aux
        tok = clip.tokenize(combined, **tok_kw)
        cond = clip.encode_from_tokens_scheduled(tok)
        msg = "aux_strength=1.0; no patch"
        if grounded:
            msg += "; grounded {} image(s)".format(len(imgs))
        if simple:
            return (model, cond, msg)
        return (model, cond, _debug_report([msg]))

    if main_only_reason is not None:
        combined = main
    else:
        combined = main + sep + text_aux if main else text_aux
    tok = clip.tokenize(combined, **tok_kw)
    ids = token_ids_from_tok(tok)
    cond = clip.encode_from_tokens_scheduled(tok)
    cond_len = cond[0][0].shape[1]
    visible_start = len(ids) - cond_len

    pairs: list = []
    lines: list[str] = []

    if need_vision_weights:
        vis_pairs = vision_cond_pairs_per_image(
            ids, cond_len, img_strengths, emphasis_mode,
            image_token_counts=img_token_counts,
        )
        pairs.extend(vis_pairs)
        lines.append(
            "vision: strengths={} → {} cond tokens scaled".format(
                img_strengths, len(vis_pairs)
            )
        )
    elif grounded:
        lines.append("vision: strengths={} (all ≈1.0, not scaled)".format(img_strengths))

    v_factor, k_bias = mix_factors(aux_strength, emphasis_mode)
    aux_is_noop = abs(v_factor - 1.0) < 1e-6 and abs(k_bias) < 1e-6

    if main_only_reason is not None:
        lines.append("{} — no aux text weighting".format(main_only_reason))
        aux_pairs = []
    elif aux_is_noop:
        lines.append("aux: strength≈1.0 — no aux text patch")
        aux_pairs = []
    else:
        if not main:
            cs, ce = user_span_from_ids(ids)
            if grounded:
                aux_pairs = slice_to_cond_pairs_after_vision(
                    ids, cs, ce, cond_len, v_factor, k_bias,
                    image_token_counts=img_token_counts,
                )
            else:
                aux_pairs = slice_to_cond_pairs(
                    cs, ce, visible_start, cond_len, v_factor, k_bias
                )
            lines.append(
                "aux: main empty — scaled all user tokens "
                "({} tokens, v={:.3f}, k_bias={:.3f})".format(
                    len(aux_pairs), v_factor, k_bias
                )
            )
        else:
            # Tokenize-only (no text-encoder pass) to find where aux starts after
            # BPE merges at the join; cheap compared with the encode above.
            ids_main = clip_token_ids(clip, main, **tok_kw)
            if grounded:
                aux_s, aux_e = aux_id_span_grounded(ids, ids_main)
                aux_pairs = slice_to_cond_pairs_after_vision(
                    ids, aux_s, aux_e, cond_len, v_factor, k_bias,
                    image_token_counts=img_token_counts,
                )
            else:
                aux_s, aux_e = aux_id_span_in_combined(ids, ids_main)
                aux_pairs = slice_to_cond_pairs(
                    aux_s, aux_e, visible_start, cond_len, v_factor, k_bias
                )
            lines.append(
                "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                    aux_s, aux_e, len(aux_pairs), v_factor, k_bias
                )
            )
            if grounded:
                lines.append("grounded {} image(s)".format(len(imgs)))
            if len(aux_pairs) == 0:
                lines.append(
                    "WARNING: 0 aux tokens — join/split failed; "
                    "try different separator or shorter main"
                )
        pairs.extend(aux_pairs)

    if not pairs:
        if simple and main_only_reason is None:
            debug = "aux id[{}:{}] → {} cond tokens v={:.4f} k_bias={:.4f}".format(
                0, 0, 0, v_factor, k_bias
            )
            if grounded:
                debug += "; grounded {} image(s)".format(len(imgs))
            return (model, cond, debug + "\nno aux positions in cond window")
        return (model, cond, _debug_report(lines + ["no weighted positions in cond window"]))

    model_clone, indices = patch_model(model, pairs, apply_to, block_range)
    if simple:
        debug = lines[0] if lines else "patched"
        aux_line = next((ln for ln in lines if ln.startswith("aux id")), None)
        if aux_line:
            debug = aux_line
            if grounded:
                debug += "; grounded {} image(s)".format(len(imgs))
        return (model_clone, cond, "{}\npatched blocks: {}".format(debug, indices))
    return (
        model_clone,
        cond,
        _debug_report(lines, patched_blocks=indices, pair_count=len(pairs)),
    )
