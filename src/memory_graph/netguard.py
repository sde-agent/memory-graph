"""Block outbound network access inside the running process.

The engine is local-first. The ONLY sanctioned egress is the explicit MCP dispatch
tool and the optional arm's-length LLM helper process — neither of which goes through
this process's sockets. Everything else must stay offline. `block_network()` is called
first by every command; pytest-socket covers tests.

This is a guardrail, not a sandbox: it does not stop system daemons or child processes,
it refuses socket connections made from within THIS interpreter.
"""

from __future__ import annotations

import os
import socket

_OFFLINE_MSG = "network access blocked by memory_graph netguard"


def _refuse(*_args, **_kwargs):
    raise OSError(_OFFLINE_MSG)


def block_network() -> None:
    """Refuse all outbound socket connections from this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    socket.socket.connect = _refuse
    socket.socket.connect_ex = _refuse
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse
