"""ZERO-EGRESS guard: the MCP server must NEVER return image bytes (photos stay local).

Showing a photo to a hosted LLM sends it to that service. The hard rule is that photo
pixels never leave the Mac, so daily_dispatch returns TEXT + a loopback link only. These
tests fail if an ImageContent block or raw image bytes ever creep back in.
"""

from __future__ import annotations

import inspect

from memory_graph import mcp_server, viewer


def test_mcp_server_source_has_no_image_content() -> None:
    # The server module must not import or construct ImageContent anywhere.
    src = inspect.getsource(mcp_server)
    assert "ImageContent" not in src, "daily_dispatch must never return image bytes"
    assert "b64" not in src and "base64" not in src, "no base64 image encoding in the MCP path"


def test_view_url_is_loopback_only() -> None:
    url = viewer.view_url("lib:abc")
    assert url.startswith("http://127.0.0.1:"), "view links must be loopback-only"
    assert "lib:abc" not in url or "%3A" in url  # id is url-encoded


def test_viewer_binds_loopback_host() -> None:
    assert viewer.HOST == "127.0.0.1"   # never 0.0.0.0 (off-machine reachable)


def test_dispatch_formats_text_link_not_image(tmp_path, monkeypatch) -> None:
    # Verify the pure dispatch logic yields a candidate and a loopback link — WITHOUT
    # starting the server (which binds a socket pytest-socket blocks). The server wraps
    # exactly this into TextContent; the no-image guarantee is covered by the source
    # scan above.
    import memory_graph.db as dbmod
    from memory_graph.config import load_config
    from memory_graph.db import connect, init_db, upsert_photo
    from memory_graph.scoring.dispatch import select_dispatch

    db = tmp_path / "t.db"
    monkeypatch.setattr(dbmod, "DB_PATH", db)
    conn = connect(db)
    init_db(conn)
    upsert_photo(conn, {
        "id": "hdd:x", "source": "hdd", "original_ref": "/v/x.jpg", "phash": "ff",
        "sharpness": 100.0, "is_canonical": 1, "resonance": 0.9, "dominant_quadrant": 3,
        "private": 0, "deriv_path": str(tmp_path / "x.jpg"),
    })
    conn.commit()
    rows = dbmod.dispatch_candidates(conn)
    conn.close()

    cfg = load_config()
    items = select_dispatch(rows, cfg.dispatch, now=1_700_000_000.0, n=1)
    assert items, "a non-private photo should be dispatchable"
    link = viewer.view_url(items[0].id)
    assert link.startswith("http://127.0.0.1:")   # a loopback link, never image bytes


def test_private_photo_excluded_from_dispatch(tmp_path, monkeypatch) -> None:
    import memory_graph.db as dbmod
    from memory_graph.db import connect, init_db, upsert_photo

    db = tmp_path / "t.db"
    monkeypatch.setattr(dbmod, "DB_PATH", db)
    conn = connect(db)
    init_db(conn)
    upsert_photo(conn, {
        "id": "lib:secret", "source": "library", "original_ref": "u", "phash": "ff",
        "sharpness": 1.0, "is_canonical": 1, "resonance": 9.9, "private": 1,
    })
    conn.commit()
    rows = dbmod.dispatch_candidates(conn)   # excludes private by default
    conn.close()
    assert all(r["id"] != "lib:secret" for r in rows)
