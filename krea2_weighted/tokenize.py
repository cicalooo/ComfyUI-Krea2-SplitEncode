"""Map main/aux prompt spans onto Krea 2 / Qwen3-VL token ids (including vision pads)."""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import torch

log = logging.getLogger("krea2_split_encode")

# Qwen3-VL vision geometry as used by ComfyUI (comfy/text_encoders/qwen3vl.py):
# patch 16, spatial merge 2 -> one LLM token per 32x32 pixel cell.
QWEN_VL_PATCH = 16
QWEN_VL_MERGE = 2
QWEN_VL_FACTOR = QWEN_VL_PATCH * QWEN_VL_MERGE
QWEN_VL_MIN_PIXELS = 3136
QWEN_VL_MAX_PIXELS = 12845056

# Qwen chat specials used by Krea 2's template.
QWEN_IM_START = 151644
QWEN_USER = 872
QWEN_NL = 198
QWEN_IM_END = 151645
QWEN_IMAGE_PAD = 151655  # <|image_pad|>; vision tower expands this placeholder.


def user_content_span(ids: Sequence[int]) -> tuple[Optional[int], Optional[int]]:
    """Token span of the user prompt between ``<|im_start|>user\\n`` and ``<|im_end|>``."""
    n = len(ids)
    for i in range(n - 2):
        if ids[i] == QWEN_IM_START and ids[i + 1] == QWEN_USER and ids[i + 2] == QWEN_NL:
            start = i + 3
            end = start
            while end < n and ids[end] != QWEN_IM_END:
                end += 1
            return start, end
    return None, None


def is_image_marker(elem) -> bool:
    """True for Qwen image placeholders (id, dict, or tensor-backed embed)."""
    if isinstance(elem, dict) and elem.get("type") == "image":
        return True
    if isinstance(elem, dict) and elem.get("original_type") == "image":
        return True
    if torch.is_tensor(elem):
        return True
    if isinstance(elem, (int, float)) and not isinstance(elem, bool):
        return int(elem) == QWEN_IMAGE_PAD
    return False


def qwen_vl_resized_hw(height: int, width: int) -> tuple[int, int]:
    """Mirror ``comfy.text_encoders.qwen_vl.process_qwen2vl_images`` resize for Qwen3-VL."""
    f = QWEN_VL_FACTOR
    h_bar = round(height / f) * f
    w_bar = round(width / f) * f
    if h_bar * w_bar > QWEN_VL_MAX_PIXELS:
        beta = math.sqrt((height * width) / QWEN_VL_MAX_PIXELS)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < QWEN_VL_MIN_PIXELS:
        beta = math.sqrt(QWEN_VL_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f
    return max(f, h_bar), max(f, w_bar)


def qwen_vl_image_tokens(height: int, width: int) -> int:
    """LLM tokens one image expands to (after the 2x2 spatial merge)."""
    h_bar, w_bar = qwen_vl_resized_hw(height, width)
    return (h_bar // QWEN_VL_FACTOR) * (w_bar // QWEN_VL_FACTOR)


def _extras_per_pad(image_ix, extra, image_token_counts):
    """Expansion extras (tokens beyond the 1 placeholder) for each ``<|image_pad|>``.

    With ``image_token_counts`` (one per image, from the prepared image sizes) each pad
    gets its exact count. If the counts disagree with the observed expansion, or are
    missing, fall back to an even split (only exact for equal-sized images).
    """
    n = len(image_ix)
    out: dict[int, int] = {}
    if not n or not extra:
        return out
    if image_token_counts is not None and len(image_token_counts) == n:
        per = [max(0, int(c) - 1) for c in image_token_counts]
        if sum(per) == extra:
            return {j: per[k] for k, j in enumerate(image_ix)}
        log.warning(
            "Krea2 Split Encode: vision token counts %s do not match expansion (%d extra); "
            "falling back to an even split", list(image_token_counts), extra,
        )
    base, rem = divmod(extra, n)
    for k, j in enumerate(image_ix):
        out[j] = base + (1 if k >= n - rem else 0)
    return out


def normalize_token_id(elem):
    """Stable id for prefix matching. Image embeds collapse to ``QWEN_IMAGE_PAD``."""
    if is_image_marker(elem):
        return QWEN_IMAGE_PAD
    if isinstance(elem, (int, float)) and not isinstance(elem, bool):
        return int(elem)
    return ("embed", type(elem).__name__)


def token_ids_from_tok(tok) -> list:
    key = next(iter(tok))
    return [normalize_token_id(t[0]) for t in tok[key][0]]


def clip_token_ids(clip, text: str, **tokenize_kwargs) -> list:
    tok = clip.tokenize(text, **tokenize_kwargs)
    return token_ids_from_tok(tok)


def user_span_from_ids(ids: Sequence[int]) -> tuple[int, int]:
    start, end = user_content_span(ids)
    if start is None:
        return 0, len(ids)
    return start, end


def parse_block_range(spec: str, n_blocks: int) -> list[int]:
    """Parse ``all``, ``0-27``, ``4,8,12``, or mixed ``0-5,10,20-27``."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(n_blocks))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a.strip()), int(b.strip())
                if lo > hi:
                    lo, hi = hi, lo
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
        except ValueError:
            raise ValueError(
                "block_range {!r}: could not read {!r}. Use 'all', a range like '0-27', "
                "a list like '4,8,12', or a mix like '0-5,20-27' (blocks 0..{}).".format(
                    spec, part, n_blocks - 1
                )
            ) from None
    seen = set()
    ordered = []
    for i in out:
        if i < 0 or i >= n_blocks or i in seen:
            continue
        seen.add(i)
        ordered.append(i)
    if not ordered:
        raise ValueError("block_range {!r} selected no blocks in 0..{}".format(spec, n_blocks - 1))
    return ordered


def longest_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def aux_id_span_in_combined(ids_combined: Sequence[int], ids_main: Sequence[int]) -> tuple[int, int]:
    """Return ``(aux_start, aux_end)`` in combined token ids (user content suffix).

    Encodes main and main+aux separately, then treats the unmatched suffix of the
    combined user span as the aux/moodboard tokens. Survives BPE glue at the join.
    """
    cs, ce = user_span_from_ids(ids_combined)
    ms, me = user_span_from_ids(ids_main)
    main_user = list(ids_main[ms:me])
    comb_user = list(ids_combined[cs:ce])
    prefix = longest_prefix_len(comb_user, main_user)
    aux_start = cs + prefix
    return aux_start, ce


# Common Qwen3-VL vision specials that can appear around IMAGE_PAD.
_VISION_SPECIALS = frozenset({
    QWEN_IMAGE_PAD,
    151652,  # <|vision_start|>
    151653,  # <|vision_end|>
})


def _text_after_vision(ids: Sequence[int], start: int, end: int) -> tuple[int, list]:
    """Return (first_text_index, list of non-vision token ids) inside [start, end)."""
    text_ids = []
    first = None
    for i in range(start, end):
        t = ids[i]
        if t in _VISION_SPECIALS or is_image_marker(t):
            continue
        if first is None:
            first = i
        text_ids.append(t if not isinstance(t, tuple) else t)
    return (first if first is not None else end), text_ids


def aux_id_span_grounded(
    ids_combined: Sequence,
    ids_main: Sequence,
) -> tuple[int, int]:
    """Aux span when vision tokens are present.

    Strips vision markers, finds the main-text prefix on pure text, then maps
    that offset back into the full combined ids. Falls back to
    ``aux_id_span_in_combined`` if the vision-aware path fails.
    """
    cs, ce = user_span_from_ids(ids_combined)
    ms, me = user_span_from_ids(ids_main)

    _, comb_text = _text_after_vision(ids_combined, cs, ce)
    _, main_text = _text_after_vision(ids_main, ms, me)

    if not comb_text or not main_text:
        return aux_id_span_in_combined(ids_combined, ids_main)

    prefix = longest_prefix_len(comb_text, main_text)
    if prefix == 0:
        return aux_id_span_in_combined(ids_combined, ids_main)

    seen = 0
    aux_start = ce
    for i in range(cs, ce):
        t = ids_combined[i]
        if t in _VISION_SPECIALS or is_image_marker(t):
            continue
        if seen == prefix:
            aux_start = i
            break
        seen += 1
    else:
        aux_start = ce

    return aux_start, ce


def slice_to_cond_pairs(
    id_start: int,
    id_end: int,
    visible_start: int,
    cond_len: int,
    v_factor: float,
    k_bias: float,
) -> list[tuple[int, float, float]]:
    pairs = []
    for i in range(id_start, id_end):
        cp = i - visible_start
        if 0 <= cp < cond_len:
            pairs.append((cp, v_factor, k_bias))
    return pairs


def vision_expansion_extra(ids: Sequence, cond_len: int) -> int:
    """Extra cond tokens Qwen inserts when expanding ``<|image_pad|>`` placeholders."""
    start, _ = user_span_from_ids(ids)
    unexpanded_visible = len(ids) - start
    return max(0, int(cond_len) - unexpanded_visible)


def slice_to_cond_pairs_after_vision(
    ids: Sequence,
    id_start: int,
    id_end: int,
    cond_len: int,
    v_factor: float,
    k_bias: float,
    image_token_counts: Optional[Sequence[int]] = None,
) -> list[tuple[int, float, float]]:
    """Map an unexpanded id span onto cond positions after vision-token expansion.

    Image placeholders (and system / chat-template tokens outside the span) are
    never weighted. Expansion extras are attributed to ``QWEN_IMAGE_PAD`` slots
    so aux text sits after the vision block.
    """
    start, _ = user_span_from_ids(ids)
    extra = vision_expansion_extra(ids, cond_len)
    image_ix = [j for j, t in enumerate(ids) if t == QWEN_IMAGE_PAD]
    extras_at = _extras_per_pad(image_ix, extra, image_token_counts)

    pairs: list[tuple[int, float, float]] = []
    acc_extra = 0
    for i, tok in enumerate(ids):
        if id_start <= i < id_end and tok != QWEN_IMAGE_PAD:
            cp = i + acc_extra - start
            if 0 <= cp < cond_len:
                pairs.append((cp, v_factor, k_bias))
        acc_extra += extras_at.get(i, 0)
    return pairs


def mix_factors(aux_strength: float, emphasis_mode: str = "value_scale") -> tuple[float, float]:
    """Map a 0..N strength to (v_factor, k_bias). 1.0 is a no-op.

    Below 1.0, value_scale / k_bias / both all scale V (or return s).
    Above 1.0, k_bias and both use a steeper key ramp so enforce terms
    can compete with vision tokens. value_scale stays a plain V multiply.
    """
    s = float(aux_strength)
    if emphasis_mode == "k_bias":
        if s > 1.0:
            return 1.0, (s - 1.0) * 3.0
        return s, 0.0
    if emphasis_mode == "both":
        if s > 1.0:
            return 1.0 + (s - 1.0) * 0.75, (s - 1.0) * 3.0
        return s, 0.0
    return s, 0.0


def vision_cond_pairs_per_image(
    ids: Sequence,
    cond_len: int,
    strengths: Sequence[float],
    emphasis_mode: str = "value_scale",
    image_token_counts: Optional[Sequence[int]] = None,
) -> list[tuple[int, float, float]]:
    """Weight pairs for each image's expanded vision tokens.

    ``strengths[i]`` controls the i-th ``<|image_pad|>``. Strength ≈ 1.0
    produces no pairs. Surrounding vision_start / vision_end stay at 1.0.
    """
    start, _ = user_span_from_ids(ids)
    extra = vision_expansion_extra(ids, cond_len)
    image_ix = [j for j, t in enumerate(ids) if t == QWEN_IMAGE_PAD]
    if not image_ix:
        return []

    extras_at = _extras_per_pad(image_ix, extra, image_token_counts)

    pairs: list[tuple[int, float, float]] = []
    acc_extra = 0
    for i, tok in enumerate(ids):
        n_extra = extras_at.get(i, 0)
        if tok == QWEN_IMAGE_PAD:
            pad_idx = image_ix.index(i)
            if pad_idx < len(strengths):
                s = float(strengths[pad_idx])
                v_factor, k_bias = mix_factors(s, emphasis_mode)
                if abs(v_factor - 1.0) > 1e-6 or abs(k_bias) > 1e-6:
                    # placeholder slot + its expansion extras
                    for e in range(n_extra + 1):
                        cp = (i - start) + acc_extra + e
                        if 0 <= cp < cond_len:
                            pairs.append((cp, v_factor, k_bias))
        acc_extra += n_extra
    return pairs
