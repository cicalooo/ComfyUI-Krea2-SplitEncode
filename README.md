# Krea2 Split Encode

A ComfyUI prompt encoder for **Krea 2**. Type your prompt and an extra "aux" text (a style or a moodboard), and choose how strongly the aux counts without it drowning the prompt. You can also give it a photo of a person to keep their face (Identity Edit), a scene, and up to five clothing or object photos. The person stays the only face the model copies.

Krea 2 reads prompts through Qwen3-VL, so the usual `(word:1.4)` weighting does nothing. This pack weights tokens inside the model's attention instead.

![Krea2 Split Encode identity workflow](split-encode.png)

| | |
|---|---|
| **Nodes** | Krea2 Split Encode · Krea2 Split Encode Options · Krea2 Split Ref |
| **Category** | `Krea2/conditioning` |
| **Examples** | [`workflows/`](workflows) (core nodes only) |
| **Docs** | [docs/reference.md](docs/reference.md) |

## Requirements

- ComfyUI **0.29.0 or newer** (V3 node API and the Krea 2 attention patch hook).
- Krea 2 model (Raw or Turbo), loaded with `UNETLoader`.
- Krea 2 text encoder: Qwen3-VL 4B, loaded with `CLIPLoader` type `krea2`.
- For Identity Edit: the Qwen Image VAE and the Krea 2 **Identity Edit LoRA** ([comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)).
- No extra pip packages.

## Install

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cicalooo/ComfyUI-Krea2-SplitEncode.git
```

Then restart ComfyUI.

## Quick start

| Workflow | What it shows |
|---|---|
| [`01_text_mix`](workflows/01_text_mix.json) | Prompt + weighted aux text. No images. |
| [`02_identity_edit`](workflows/02_identity_edit.json) | Person photo + VAE → keep the face. |
| [`03_identity_split_ref`](workflows/03_identity_split_ref.json) | Person + a clothing guide through Split Ref. |
| [`04_cfg_negative`](workflows/04_cfg_negative.json) | CFG > 1 with the correct negative encode. |

Wire **both** outputs, `model` and `conditioning`, to the sampler. The weighting lives on the model.

## What runs

| Connected | Result |
|---|---|
| Nothing but text | Prompt at full strength, aux scaled by `aux_strength`. |
| Images, no VAE | Qwen grounding only: the images are read as vision tokens. |
| Person + VAE | Identity Edit. With a scene: scene is frame 1, person frame 2. |
| Person + VAE + Split Ref | The person is the **only** VAE frame. Scene and guides are Qwen-only, so they can't overwrite the face. |

If Split Ref is connected but `identity` is not, guide slot 1 is used as the person.

## Inputs

### Krea2 Split Encode

| Input | Default | Meaning |
|---|---|---|
| `clip`, `model` | — | Krea 2 text encoder and model (LoRA already applied). |
| `prompt` | | Main prompt, always at full strength. |
| `aux` | | Secondary text (style, moodboard). |
| `aux_strength` | 0.45 | 0 = leave aux out, 0.35–0.55 = gentle, 1 = same as prompt, >1 = enforce. |
| `identity` | | Person photo. |
| `scene` | | Scene or layout photo. |
| `vae` | | Qwen Image VAE. Turns Identity Edit on. |
| `ref_boost` | 4 | Likeness pull toward the person (Identity Edit only). 1 = off, max 10. |
| `latent` | | The same empty latent as KSampler, so the VAE encode runs before sampling. |
| `options` | | From Split Encode Options. |
| `refs` | | From Split Ref. |

Outputs: `model`, `conditioning`, `debug`.

### Krea2 Split Encode Options

`people_count_lock`, `emphasis_mode`, `separator`, `block_range`, `apply_to`, `grounding_px` (768), `guide_grounding_px` (384), `system_prompt`, `fit_mode`, `scene_strength`, `identity_strength`, `scene_focus`, `identity_focus`, `scene_ref_boost`, `identity_mask`, `scene_mask`. Each input has a tooltip; the full table is in [docs/reference.md](docs/reference.md).

### Krea2 Split Ref

Up to 5 × (`image_N`, `strength_N` = 0.45, `focus_N`). These are clothing or object guides read by Qwen only. Keep the strength near 0.45, because at 1.0 a clothing photo competes with the face.

## Sampler settings

| Job | Setup |
|---|---|
| Text mix, most edits | Turbo, CFG **1**, 8–12 steps. Negative: `ConditioningZeroOut` (unused at CFG 1). |
| Removals | Raw, CFG **~3**, ~20 steps. Negative: a **second** Split Encode with an empty prompt, the same images and **no VAE**. Leave its `model` output unconnected. |

Stay at or below about 2 MP.

## Reading `debug`

- `aux id[a:b] → N cond tokens v=0.45` means N aux tokens were weighted. `0 aux tokens` is a warning: the prompt/aux split failed, so try another `separator`.
- `vision: strengths=[…] → N cond tokens scaled` shows the per-image Qwen strengths.
- `DiT refs=1 (identity) boosts=[4.0]` lists the Identity Edit frames. With guides you should see `scene is Qwen-only while guides are connected`.
- `identity_edit=ON fit=fit` means the pixel path is active.
- `patched blocks: [...]` shows which DiT blocks carry the weighting.

## Troubleshooting

- **"already carries an Identity Edit patch"**: two encodes with a VAE are feeding the same model. Disconnect the VAE from the negative encode, or remove a Mustyrocks `Krea2EditModelPatch` node.
- **The face drifts**: a long aux text outweighs a short prompt. Lower `aux_strength`, set it to 0, or clear aux. Check that the guide strengths are around 0.45.
- **Weighting seems to do nothing**: make sure the node's `model` output feeds the sampler.
- **Slow steps with Identity Edit**: connect `latent` so the VAE encode happens before sampling.
- **`block_range` error**: use `all`, `0-27`, `4,8,12` or `0-5,20-27`.
- Don't combine with Krea2 Apply Regional, or with another token-weighting node on the same blocks.

## Migrating

**From 1.x.** Node ids are unchanged, so saved graphs still find the nodes, but some inputs were renamed or removed. Re-check the widget values after loading.

| 1.x | 2.0 |
|---|---|
| `image` (Encode) | removed: use `scene` |
| `image_b` (Encode) | removed: use `identity` |
| `ref_a_strength` / `ref_b_strength` | `scene_strength` / `identity_strength` |
| `ref_a_focus` / `ref_b_focus` | `scene_focus` / `identity_focus` |
| `ref_boost_a` | `scene_ref_boost` |
| `ref_boost_mask` / `ref_boost_mask_a` | `identity_mask` / `scene_mask` |
| `ref_boost` max 1000 | max 10 |
| `grounding_px` step 64 | step 32 (32 px Qwen3-VL grid) |

**From 0.3 (Krea2 Prompt Mix).** Use Krea2 Split Encode: `text_main` → `prompt`, `text_aux` → `aux`, `image` → `scene`, `image_b` → `identity`.

## Credits

The Identity Edit pixel path is vendored from [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) 1.3.0 by lbouaraba (Mustyrocks, Apache-2.0), with its per-ref focus and vision-strength ideas. "Birds Weighted Conditioning" inspired the per-reference weighting. Moodboard aux text works well with [ComfyUI-Krea-Moodboards](https://github.com/Andro-Meta/ComfyUI-Krea-Moodboards) (Andro-Meta); it isn't bundled.

## License

Apache-2.0. See [LICENSE](LICENSE).
