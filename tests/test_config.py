"""Config loader + validation (P0)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from memory_graph.config import Config, load_config
from memory_graph.paths import DEFAULT_CONFIG


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    # A few anchor values straight from config.toml.
    assert cfg.ingest.thumb_long_edge == 768
    assert ".jpg" in cfg.ingest.hdd_extensions
    # Weights were tuned to the strict labels: Q3 (people/tender) highest, Q5 lowest.
    # Exact values are tunable, so assert the ordering property instead.
    assert cfg.resonance_weights.q3 == max(cfg.resonance_weights.as_tuple())
    assert cfg.resonance_weights.q5 == min(cfg.resonance_weights.as_tuple())
    assert cfg.narration.backend == "foundation_models"
    assert "Hidden" in cfg.privacy.hidden_albums


def test_local_overlay_overrides_single_key(tmp_path: Path) -> None:
    local = tmp_path / "config.local.toml"
    local.write_text(
        textwrap.dedent(
            """
            [ingest]
            thumb_long_edge = 512
            """
        )
    )
    cfg = load_config(default_path=DEFAULT_CONFIG, local_path=local)
    # Overridden...
    assert cfg.ingest.thumb_long_edge == 512
    # ...while other keys keep their defaults.
    assert ".nef" in cfg.ingest.hdd_extensions


def test_quadrant_weights_tuned_q3_top_q5_bottom() -> None:
    w = load_config().resonance_weights
    # Tuned to the STRICT dispatch labels: Q3 (people/tender) is the top weight (the
    # strongest dispatch-worthy quadrant) and Q5 (tension) is lowest.
    assert w.q3 == max(w.as_tuple())
    assert w.q5 == min(w.as_tuple())


def test_invalid_thumb_size_rejected(tmp_path: Path) -> None:
    bad_default = tmp_path / "config.toml"
    good = DEFAULT_CONFIG.read_text().replace("thumb_long_edge = 768", "thumb_long_edge = 50")
    bad_default.write_text(good)
    with pytest.raises(ValueError, match="thumb_long_edge"):
        load_config(default_path=bad_default, local_path=tmp_path / "nope.toml")


def test_framing_bounds_must_be_increasing(tmp_path: Path) -> None:
    bad = tmp_path / "config.toml"
    text = DEFAULT_CONFIG.read_text().replace(
        "bounds = [20.0, 35.0, 71.0, 135.0, 300.0]",
        "bounds = [20.0, 35.0, 35.0, 135.0, 300.0]",
    )
    bad.write_text(text)
    with pytest.raises(ValueError, match="framing.bounds"):
        load_config(default_path=bad, local_path=tmp_path / "nope.toml")
