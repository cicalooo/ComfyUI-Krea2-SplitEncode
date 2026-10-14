"""Krea2 Split Encode: ComfyUI V3 entry point.

ComfyUI loads this file by path (the folder name contains hyphens) and calls
``comfy_entrypoint``. Pytest may import it directly; fall back to an absolute import.
"""

try:
    from .krea2_weighted import __version__
    from .krea2_weighted.nodes import comfy_entrypoint
except ImportError:
    from krea2_weighted import __version__
    from krea2_weighted.nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint", "__version__"]
