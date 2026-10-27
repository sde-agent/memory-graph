"""The network guardrail refuses outbound connections (P0).

Note: pytest already runs with --disable-socket (pytest-socket), which blocks even
constructing a socket. These tests therefore verify memory_graph's OWN netguard by
checking the functions it replaces directly — without instantiating a real socket —
so the two guardrails do not collide.
"""

from __future__ import annotations

import socket

import pytest

from memory_graph.netguard import _OFFLINE_MSG, block_network


def test_block_network_replaces_connect(monkeypatch) -> None:
    # Baseline: a harmless connect we can prove block_network() overrides.
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: None, raising=True)
    block_network()
    # Call the (now guarded) method object directly — no socket instance needed.
    with pytest.raises(OSError, match="netguard"):
        socket.socket.connect(object(), ("example.com", 80))


def test_block_network_refuses_getaddrinfo(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [], raising=True)
    block_network()
    with pytest.raises(OSError, match="netguard"):
        socket.getaddrinfo("example.com", 80)


def test_offline_message_is_identifiable() -> None:
    assert "netguard" in _OFFLINE_MSG
