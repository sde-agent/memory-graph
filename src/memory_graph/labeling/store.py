"""Resumable label storage (evaluation.md A.2-A.4).

Labels are appended to jsonl files under data/labels/, one per label type, so a session
resumes where it left off (a crash or a break loses nothing). Each store knows which
photo ids it has already labeled, so the UI never re-asks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memory_graph.paths import DATA_DIR

LABELS_DIR = DATA_DIR / "labels"


@dataclass
class LabelStore:
    """Append-only jsonl store for one label kind, keyed by a label's id field."""

    kind: str                       # "base_rate" | "quadrant" | "pairwise"
    key_field: str = "id"           # the field identifying an already-labeled item

    @property
    def path(self) -> Path:
        return LABELS_DIR / f"{self.kind}.jsonl"

    def _rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # a line cut short by an interrupted write is skipped
        return out

    def labeled_keys(self) -> set[str]:
        return {str(r[self.key_field]) for r in self._rows() if self.key_field in r}

    def count(self) -> int:
        return len(self._rows())

    def append(self, record: dict) -> None:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def all(self) -> list[dict]:
        return self._rows()
