"""The per-user taste / "memory" model (spec.md §6, V1.5; evaluation.md Part A).

A per-user model that predicts "would I want to see this again" from an image
embedding, trained on the user's OWN Favorite flag. Its output is a resonance INPUT
(the `taste_score` multiplier), never a label oracle. Each user trains their own.

Design choices:
  - **Embedding = Apple Vision feature print** (vision.feature_print), not CLIP. This
    keeps the engine free of torch and iOS-portable (the V3 north star). The embedding
    is L2-normalized before training/scoring.
  - **Model = plain L2-regularized logistic regression** fit with gradient descent in
    numpy — no sklearn, no torch. Small, inspectable, fast, and ships anywhere.
  - **Evaluation is base-rate-first** (the hard-won lesson): we report the favorite
    base rate and AUC via a train/test split BEFORE claiming the model is good, and the
    success bar is set relative to that base rate, never guessed in advance.

Scores are mapped to [0,1] (sigmoid output), which `scoring.resonance` turns into a
multiplier around 1.0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from memory_graph.paths import TASTE_META_PATH


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms



def privacy_policy_fingerprint(privacy: Any) -> str:
    """Stable fingerprint of the hard-exclusion policy used by a model."""
    payload = {
        "people": sorted(str(v).casefold() for v in privacy.exclude_people),
        "anonymous_clusters": sorted(str(v) for v in privacy.exclude_anonymous_clusters),
        "subjects": sorted(str(v).casefold() for v in privacy.exclude_subjects),
        "camera_models": sorted(str(v).casefold() for v in privacy.exclude_camera_models),
        "places": sorted(str(v).casefold() for v in privacy.exclude_places),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def write_taste_meta(**values: Any) -> None:
    TASTE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASTE_META_PATH.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def load_embedding_cache(path) -> dict[str, np.ndarray]:
    """Load the local photo->feature-vector cache; missing cache means empty."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"].item()))
            keys = meta.get("keys", [])
            matrix = np.asarray(archive["vectors"], dtype=np.float32)
        if len(keys) != len(matrix):
            return {}
        return {key: matrix[index] for index, key in enumerate(keys)}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def save_embedding_cache(path, vectors: dict[str, np.ndarray]) -> None:
    """Atomically save derived vectors, never image pixels."""
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(vectors)
    if keys:
        matrix = np.vstack([vectors[key] for key in keys]).astype(np.float32)
    else:
        matrix = np.empty((0, 0), np.float32)
    metadata = json.dumps({"format_version": 1, "keys": keys}, separators=(",", ":"))
    fd, temp_name = tempfile.mkstemp(prefix=".taste-embeddings-", suffix=".npz", dir=path.parent)
    import os
    os.close(fd)
    try:
        np.savez_compressed(temp_name, vectors=matrix, metadata=np.array(metadata))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def mark_taste_stale(privacy: Any, reason: str = "policy changed") -> None:
    """Invalidate a personal model without deleting it or touching photo history."""
    write_taste_meta(
        status="stale",
        reason=reason,
        policy_fingerprint=privacy_policy_fingerprint(privacy),
    )


def taste_model_status() -> dict[str, Any] | None:
    if not TASTE_META_PATH.is_file():
        return None
    try:
        return json.loads(TASTE_META_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class TasteModel:
    """A trained logistic-regression taste model over normalized embeddings."""

    weights: np.ndarray       # shape (d,)
    bias: float
    mean: np.ndarray          # feature mean used to center at train time
    dim: int

    def score(self, embedding: list[float] | np.ndarray | None) -> float | None:
        """P(would want to see again) in [0,1] for one embedding, or None if unusable."""
        if embedding is None:
            return None
        v = np.asarray(embedding, dtype=np.float64)
        if v.shape[0] != self.dim:
            return None
        v = _normalize(v.reshape(1, -1))[0] - self.mean
        z = float(np.dot(self.weights, v) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TasteModel:
        return cls(
            weights=np.asarray(d["weights"], dtype=np.float64),
            bias=float(d["bias"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            dim=int(d["dim"]),
        )

    def save(self, path) -> None:
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> TasteModel | None:
        import json

        p = Path(path)
        if not p.is_file():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:
            return None


def train(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
    lr: float = 0.5,
    epochs: int = 300,
) -> TasteModel:
    """Fit logistic regression on (embeddings, favorite labels) with gradient descent.

    `embeddings` is (n, d); `labels` is (n,) in {0,1}. Class imbalance (favorites are
    rare) is handled by weighting positive examples up to balance the two classes.
    """
    x = _normalize(np.asarray(embeddings, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0)
    x = x - mean
    n, d = x.shape

    pos = max(1.0, y.sum())
    neg = max(1.0, n - y.sum())
    w_pos = n / (2 * pos)
    w_neg = n / (2 * neg)
    sample_w = np.where(y == 1, w_pos, w_neg)

    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sample_w
        grad_w = x.T @ err / n + l2 * w / n
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return TasteModel(weights=w, bias=b, mean=mean, dim=d)


@dataclass(frozen=True)
class TasteEval:
    n: int
    n_favorites: int
    base_rate: float          # fraction of favorites — the number every bar is set against
    auc: float                # ROC AUC on the held-out split
    note: str = ""


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic with TIE-CORRECTED ranks; 0.5 = chance.

    Average ranks for tied scores so that all-equal predictions correctly give 0.5
    (chance), not 0.0.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    # Average (fractional) ranks: tied values share the mean of their rank positions.
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    n = len(sorted_v)
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_pos = ranks[: len(pos)].sum()
    u = rank_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def evaluate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
    **train_kw,
) -> tuple[TasteModel, TasteEval]:
    """Train on a split and report base rate + held-out AUC (base-rate-first discipline).

    Returns the model trained on the TRAIN split and the evaluation on the TEST split.
    The caller sets any success bar AFTER seeing `base_rate` — never before.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    model = train(x[tr], y[tr], **train_kw)
    probs = np.array([model.score(x[i]) or 0.5 for i in te])
    ev = TasteEval(
        n=n,
        n_favorites=int(y.sum()),
        base_rate=float(y.mean()),
        auc=auc_score(y[te], probs),
    )
    return model, ev
