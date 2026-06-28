from __future__ import annotations

import os
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
    configure_rank_local_datasets_cache()
    return repo_root


def configure_rank_local_datasets_cache() -> None:
    enabled = os.environ.get("CLCF_PER_RANK_DATASETS_CACHE", "").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    cache_base = os.environ.get("HF_DATASETS_CACHE_BASE") or os.environ.get("HF_DATASETS_CACHE")
    if not cache_base:
        return

    rank_cache = Path(cache_base).expanduser() / f"rank_{local_rank}"
    rank_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(rank_cache)


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
