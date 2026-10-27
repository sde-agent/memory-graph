#!/usr/bin/env python3
"""Fail-closed PII guard — run BEFORE any commit/push to the public repo.

The product rule is absolute: no personal data in the repo, ever (homes, eras, people,
archive paths live only in gitignored config/data). This scans every file that git would
track against a denylist of personal strings (names, usernames, drive labels) read from
the GITIGNORED config.local.toml `[publish_check] deny` — so the denylist itself, which
contains the real names, never ships. Also catches absolute /Users/<name> paths and the
owner's own self_names.

Exit 0 = clean (safe to publish). Exit 1 = a leak was found (DO NOT publish). Exit 2 =
misconfigured (no denylist). The check is intentionally dumb and literal: a substring
match is a failure, because a false alarm is cheap and a real leak is not.

Usage:  python scripts/publish_check.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "config.local.toml"

# Files/dirs the scan skips (binaries, caches, the venv, and the gitignored local files).
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
              "data", "node_modules", ".mypy_cache"}
_SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".db",
                  ".sqlite", ".bak"}
_SKIP_FILES = {"config.local.toml"}


def _denylist() -> list[str]:
    """Personal strings to forbid, from the gitignored local config (never shipped)."""
    deny: list[str] = []
    if LOCAL.is_file():
        raw = tomllib.loads(LOCAL.read_text())
        deny += list(raw.get("publish_check", {}).get("deny", []))
        # The owner's own names are PII too — fold them in automatically.
        for nm in raw.get("owner", {}).get("self_names", []):
            deny += nm.split()
    # De-dup, drop empties, keep case-insensitive uniqueness.
    seen: set[str] = set()
    out: list[str] = []
    for d in deny:
        d = d.strip()
        if d and d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out


def _candidate_files() -> list[Path]:
    """Every file under the repo that is NOT skipped — the would-be-committed surface."""
    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if p.suffix.lower() in _SKIP_SUFFIXES or p.name in _SKIP_FILES:
            continue
        files.append(p)
    return files


# Absolute home paths that reveal a username, e.g. /Users/alice/... — never commit these.
_USER_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/")


def main() -> int:
    deny = _denylist()
    if not deny:
        print("publish-check: NO denylist found (config.local.toml [publish_check] deny).")
        print("  Refusing to pass without a denylist — add one before publishing.")
        return 2

    # Match each denied string on WORD BOUNDARIES so a short name ("Riti") flags the
    # name but not the substring inside ordinary words ("c-riti-que", "c-riti-cal"). The
    # match is case-insensitive. This keeps the guard strict about identities without
    # drowning in false alarms that would train the user to ignore it.
    patterns = [(d, re.compile(rf"\b{re.escape(d)}\b", re.IGNORECASE)) for d in deny]
    leaks: list[str] = []
    for f in _candidate_files():
        if f.resolve() == Path(__file__).resolve():
            continue   # don't flag this script's own /Users/<name> example in a comment
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        rel = f.relative_to(ROOT)
        for i, line in enumerate(text.splitlines(), 1):
            for original, pat in patterns:
                if pat.search(line):
                    leaks.append(f"  {rel}:{i}: contains {original!r}")
            if _USER_PATH.search(line):
                leaks.append(f"  {rel}:{i}: absolute /Users/<name> path")

    if leaks:
        print("publish-check: FAILED — personal data found in would-be-committed files:")
        print("\n".join(sorted(set(leaks))))
        print(f"\n{len(set(leaks))} leak(s). Redact before publishing. DO NOT commit.")
        return 1

    print(f"publish-check: clean. Scanned {len(_candidate_files())} files against "
          f"{len(deny)} denied strings. Safe to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
