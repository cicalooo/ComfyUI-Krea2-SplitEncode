"""Test path setup.

The pack's pure logic (``krea2_weighted.core``, ``tokenize``, ``attn_patch``) needs only
torch. The V3 node tests also need ComfyUI's ``comfy_api``; point ``COMFYUI_PATH`` at a
ComfyUI checkout (default: ``../ComfyUI`` next to this repo) or they are skipped.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_comfy = os.environ.get("COMFYUI_PATH") or str(ROOT.parent / "ComfyUI")
if Path(_comfy, "comfy_api").is_dir() and _comfy not in sys.path:
    sys.path.append(_comfy)
