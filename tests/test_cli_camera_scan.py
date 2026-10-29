"""The camera-scan command proposes and (on --confirm) writes crop factors.

Uses monkeypatched metadata streams, so no real Photos library or HDD is needed and no
master or pixel is ever opened.
"""

from __future__ import annotations

from pathlib import Path

from memory_graph import cli


def _meta(model: str, focal: float, focal35: float | None) -> dict:
    return {"make": "Canon", "model": model, "focal": focal, "focal35": focal35}


def _isolate_config(monkeypatch) -> None:
    """Load config with no ambient local overlay so the merged write is deterministic."""
    from memory_graph import config as config_mod
    from memory_graph.paths import DEFAULT_CONFIG

    original_load = config_mod.load_config
    monkeypatch.setattr(
        "memory_graph.config.load_config",
        lambda **kw: original_load(
            default_path=DEFAULT_CONFIG, local_path=DEFAULT_CONFIG.parent / "nonexistent.toml"
        ),
    )


def test_scan_without_confirm_writes_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)
    monkeypatch.setattr(
        "memory_graph.ingest.library.stream_library_camera_meta",
        lambda dbfile=None: iter((_meta("Canon EOS Rebel XTi", 18.0, None),)),
    )
    writes: list[dict] = []
    monkeypatch.setattr("memory_graph.setup.writer.update_crop_factors",
                        lambda crop, **kw: writes.append(crop))

    assert cli.main(["camera-scan"]) == 0

    out = capsys.readouterr().out
    assert "missing a 35mm-equivalent" in out
    assert "Canon EOS Rebel XTi: 1.6" in out
    assert "--confirm" in out
    assert writes == []


def test_scan_with_confirm_writes_resolvable_factor(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)
    monkeypatch.setattr(
        "memory_graph.ingest.library.stream_library_camera_meta",
        lambda dbfile=None: iter((_meta("Canon EOS Rebel XTi", 18.0, None),)),
    )
    writes: list[dict] = []
    monkeypatch.setattr("memory_graph.setup.writer.update_crop_factors",
                        lambda crop, **kw: writes.append(crop))

    assert cli.main(["camera-scan", "--confirm"]) == 0

    assert writes and writes[0] == {"Canon EOS Rebel XTi": 1.6}
    assert "Wrote 1 crop factor" in capsys.readouterr().out


def test_scan_reports_unresolved_and_skips_resolved(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)
    # A camera already carrying a 35mm-equivalent is not proposed; an unknown one is
    # reported unresolved rather than given a guessed factor.
    stream = (
        _meta("Known Equivalent Cam", 24.0, 36.0),
        _meta("Mystery Compact", 6.3, None),
    )
    monkeypatch.setattr(
        "memory_graph.ingest.library.stream_library_camera_meta",
        lambda dbfile=None: iter(stream),
    )
    writes: list[dict] = []
    monkeypatch.setattr("memory_graph.setup.writer.update_crop_factors",
                        lambda crop, **kw: writes.append(crop))

    # --confirm with only unresolved models writes nothing and signals the gap.
    assert cli.main(["camera-scan", "--confirm"]) == 1
    out = capsys.readouterr().out
    assert "Mystery Compact: UNRESOLVED" in out
    assert "Known Equivalent Cam" not in out
    assert writes == []


def test_scan_handles_missing_photos_database(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)

    def _raise(dbfile=None):
        raise FileNotFoundError("Could not get path to photo library database")
        yield  # pragma: no cover — makes this a generator

    monkeypatch.setattr(
        "memory_graph.ingest.library.stream_library_camera_meta", _raise
    )

    assert cli.main(["camera-scan"]) == 1
    assert "Photos Library database not found" in capsys.readouterr().out


def test_hdd_only_requires_hdd_root(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)
    assert cli.main(["camera-scan", "--hdd-only"]) == 1
    assert "--hdd-only needs --hdd-root" in capsys.readouterr().out


def test_hdd_only_scans_only_the_archive(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "block_network", lambda: None)
    _isolate_config(monkeypatch)

    def _library_should_not_run(dbfile=None):
        raise AssertionError("library scan must be skipped with --hdd-only")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "memory_graph.ingest.library.stream_library_camera_meta", _library_should_not_run
    )
    monkeypatch.setattr(
        "memory_graph.ingest.hdd.stream_hdd_camera_meta",
        lambda root, cfg: iter((_meta("Mystery Compact", 6.3, None),)),
    )

    assert cli.main(["camera-scan", "--hdd-only", "--hdd-root", str(tmp_path)]) == 0
    assert "Mystery Compact: UNRESOLVED" in capsys.readouterr().out
