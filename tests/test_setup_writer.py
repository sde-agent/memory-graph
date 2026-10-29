"""Atomic crop-factor writes to the local overlay preserve unrelated config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from memory_graph.config import load_config
from memory_graph.setup.writer import update_crop_factors


def test_writer_preserves_unrelated_tables_and_loads(tmp_path: Path) -> None:
    local = tmp_path / "config.local.toml"
    local.write_text(
        textwrap.dedent(
            """
            [ingest]
            thumb_long_edge = 512

            [framing.crop]
            "Old Camera" = 2.0

            [custom]
            value = "preserve me"
            """
        ).strip()
        + "\n"
    )
    update_crop_factors({"Canon EOS Rebel XTi": 1.6}, path=local)
    text = local.read_text()
    assert 'value = "preserve me"' in text
    assert '"Canon EOS Rebel XTi" = 1.6' in text
    assert "Old Camera" not in text
    cfg = load_config(local_path=local)
    assert cfg.ingest.thumb_long_edge == 512
    assert cfg.framing.crop == {"Canon EOS Rebel XTi": 1.6}


def test_writer_creates_overlay_when_absent(tmp_path: Path) -> None:
    local = tmp_path / "config.local.toml"
    update_crop_factors({"Full Frame": 1.0}, path=local)
    assert load_config(local_path=local).framing.crop == {"Full Frame": 1.0}


def test_writer_rejects_invalid_factor_without_replacing_file(tmp_path: Path) -> None:
    local = tmp_path / "config.local.toml"
    local.write_text('[custom]\nvalue = "safe"\n')
    with pytest.raises(ValueError, match="invalid crop factor"):
        update_crop_factors({"Camera": -1.0}, path=local)
    assert local.read_text() == '[custom]\nvalue = "safe"\n'
