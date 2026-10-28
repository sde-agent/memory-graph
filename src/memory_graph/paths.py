"""Canonical filesystem locations.

The repo root is found by walking up from this file until a `pyproject.toml` is seen,
so the engine works whether installed or run from a checkout. All personal, generated,
and cached state lives under `data/`, which is gitignored (no personal data in the repo).
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: three levels up from src/memory_graph/paths.py
    return start.parents[2]


REPO_ROOT: Path = _find_repo_root(Path(__file__).resolve())

# Default config ships with the repo; a gitignored local overlay holds personal values.
DEFAULT_CONFIG: Path = REPO_ROOT / "config.toml"
LOCAL_CONFIG: Path = REPO_ROOT / "config.local.toml"

# All generated / personal state. Never tracked.
DATA_DIR: Path = REPO_ROOT / "data"
DB_PATH: Path = DATA_DIR / "memory_graph.db"
CACHE_DIR: Path = DATA_DIR / "cache"
GALLERY_DIR: Path = DATA_DIR / "gallery"
TASTE_MODEL_PATH: Path = DATA_DIR / "taste" / "model.json"
TASTE_META_PATH: Path = DATA_DIR / "taste" / "model_meta.json"


def ensure_data_dirs() -> None:
    """Create the data directories if they do not exist."""
    for d in (DATA_DIR, CACHE_DIR, GALLERY_DIR):
        d.mkdir(parents=True, exist_ok=True)
