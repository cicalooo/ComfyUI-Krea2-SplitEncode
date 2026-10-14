"""Krea2 Split Encode."""

from importlib import metadata as _md
from pathlib import Path as _Path


def _read_version() -> str:
    # Single source of truth: pyproject.toml [project].version.
    pp = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        for line in pp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("version") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    try:
        return _md.version("comfyui-krea2-splitencode")
    except _md.PackageNotFoundError:
        return "0.0.0"


__version__ = _read_version()
