# Krea2 Split Encode

A Krea 2 prompt encoder for ComfyUI. It weights a separate style/aux text against your prompt and can keep a person's face (Identity Edit) while placing scenes and clothing.

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 0.29.0 or newer
- Krea 2 model and its Qwen3-VL 4B text encoder (`CLIPLoader` type `krea2`)
- For Identity Edit: the Qwen Image VAE and the Krea 2 Identity Edit LoRA ([comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit))

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cicalooo/ComfyUI-Krea2-SplitEncode.git
```

Restart ComfyUI. No extra pip packages.

## Nodes

- **Krea2 Split Encode**: `prompt` plus `aux` weighted by `aux_strength`. Add `identity` and `vae` for Identity Edit, and optionally a `scene`. Wire both `model` and `conditioning` to the sampler.
- **Krea2 Split Encode Options**: advanced settings. If it isn't connected, the defaults apply.
- **Krea2 Split Ref**: up to 5 clothing/object guides. Qwen reads them; they never take the person's frame.

## Negative and CFG

At CFG 1, use `ConditioningZeroOut` as the negative. At CFG > 1, the negative is a second Split Encode with an empty prompt, the same images and **no VAE**. Leave its `model` output unconnected.

## More

- Full inputs, debug output, troubleshooting and migration notes: [docs/reference.md](docs/reference.md)
- Example workflows: [example_workflows/](example_workflows)

## Credits and license

The Identity Edit pixel path is vendored from [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) (lbouaraba / Mustyrocks, Apache-2.0). Aux text pairs well with [ComfyUI-Krea-Moodboards](https://github.com/Andro-Meta/ComfyUI-Krea-Moodboards).

Apache-2.0. See [LICENSE](LICENSE).
