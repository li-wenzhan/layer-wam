from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def bootstrap_repo_src() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return repo_root


def assert_huggingface_hub_compatible() -> None:
    try:
        hub_version = version("huggingface-hub")
    except PackageNotFoundError as exc:
        raise ImportError(
            "Missing dependency `huggingface-hub`. Run: "
            "python -m pip install 'huggingface-hub==0.29.2'"
        ) from exc

    parsed = _parse_version_prefix(hub_version)
    if parsed < (0, 26, 0) or parsed >= (1, 0, 0):
        raise ImportError(
            "Incompatible `huggingface-hub` version for transformers in this project: "
            f"found {hub_version}, expected >=0.26.0,<1.0. "
            "Run: python -m pip install 'huggingface-hub==0.29.2' && python -m pip install -e ."
        )


def _parse_version_prefix(value: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value))
    if not match:
        return (0, 0, 0)
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    return major, minor, patch
