"""Krea2 Split Encode: ComfyUI V3 entry point.

ComfyUI loads this file by path (the folder name contains hyphens) and awaits
``comfy_entrypoint``. The node module (which needs ``comfy_api``) is imported lazily so
pytest can import this package without a ComfyUI checkout.
"""

try:
    from .krea2_weighted import __version__
except ImportError:
    from krea2_weighted import __version__


async def comfy_entrypoint():
    try:
        from .krea2_weighted.nodes import comfy_entrypoint as _entrypoint
    except ImportError:
        from krea2_weighted.nodes import comfy_entrypoint as _entrypoint
    return await _entrypoint()


__all__ = ["comfy_entrypoint", "__version__"]
