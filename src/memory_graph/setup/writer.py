"""Validated, atomic writes of confirmed crop factors to the local TOML overlay."""

from __future__ import annotations

import math
import os
import tempfile
import tomllib
from pathlib import Path

from memory_graph.config import load_config
from memory_graph.paths import DEFAULT_CONFIG, LOCAL_CONFIG


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _without_sections(text: str, names: set[str]) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = stripped[1:-1] in names
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()


def _render_overlay(existing: str, crop: dict[str, float]) -> str:
    body = _without_sections(existing, {"framing.crop"})
    section = "[framing.crop]\n" + "\n".join(
        f"{_toml_string(model)} = {factor:g}" for model, factor in crop.items()
    )
    return "\n\n".join(part for part in (body, section) if part) + "\n"


def _validate_crop(crop: dict[str, float]) -> None:
    for model, factor in crop.items():
        if not model.strip():
            raise ValueError("camera model must not be empty")
        if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"invalid crop factor for {model!r}: {factor!r}")


def update_crop_factors(
    crop: dict[str, float],
    *,
    path: Path = LOCAL_CONFIG,
    default_path: Path = DEFAULT_CONFIG,
) -> None:
    """Replace the ``[framing.crop]`` table and atomically rewrite the local overlay.

    Unrelated local configuration tables are preserved. The merged result is parsed
    and passed through ``load_config`` before replacing the file, so a bad write can
    never leave an unloadable overlay in place.
    """
    _validate_crop(crop)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    rendered = _render_overlay(existing, crop)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        tomllib.loads(rendered)
        load_config(default_path=default_path, local_path=temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
