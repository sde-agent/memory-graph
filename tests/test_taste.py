"""Taste model: training, evaluation, persistence (P9).

Uses synthetic embeddings so the test is deterministic and needs no Apple Vision.
"""

from __future__ import annotations

import numpy as np

from memory_graph.scoring.taste import TasteModel, auc_score, evaluate, train


def _separable(n=200, d=32, seed=0):
    """Favorites cluster around one direction; non-favorites around another."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pos = rng.normal(0.0, 1.0, (half, d)) + np.r_[2.0, np.zeros(d - 1)]
    neg = rng.normal(0.0, 1.0, (half, d)) + np.r_[-2.0, np.zeros(d - 1)]
    x = np.vstack([pos, neg])
    y = np.r_[np.ones(half), np.zeros(half)]
    return x, y


def test_trains_and_separates() -> None:
    x, y = _separable()
    model = train(x, y)
    pos_score = np.mean([model.score(x[i]) for i in range(len(y)) if y[i] == 1])
    neg_score = np.mean([model.score(x[i]) for i in range(len(y)) if y[i] == 0])
    assert pos_score > neg_score  # favorites score higher


def test_auc_high_on_separable_heldout() -> None:
    x, y = _separable(n=300)
    _, ev = evaluate(x, y, test_frac=0.3, seed=1)
    assert ev.auc > 0.9  # clearly separable => strong held-out AUC


def test_auc_chance_on_random() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 16))
    y = rng.integers(0, 2, size=200).astype(float)
    _, ev = evaluate(x, y, test_frac=0.3, seed=2)
    assert 0.3 < ev.auc < 0.7  # no signal => near chance


def test_base_rate_reported() -> None:
    x, y = _separable(n=100)
    _, ev = evaluate(x, y)
    assert abs(ev.base_rate - 0.5) < 0.05   # balanced synthetic set
    assert ev.n == 100
    assert ev.n_favorites == 50


def test_score_in_unit_range() -> None:
    x, y = _separable()
    model = train(x, y)
    for i in range(10):
        s = model.score(x[i])
        assert 0.0 <= s <= 1.0


def test_score_none_for_bad_input() -> None:
    x, y = _separable(d=32)
    model = train(x, y)
    assert model.score(None) is None
    assert model.score(np.zeros(8)) is None   # wrong dimensionality


def test_persistence_round_trip(tmp_path) -> None:
    x, y = _separable()
    model = train(x, y)
    path = tmp_path / "taste" / "model.json"
    model.save(path)
    loaded = TasteModel.load(path)
    assert loaded is not None
    # Same scores after a round trip.
    assert abs(loaded.score(x[0]) - model.score(x[0])) < 1e-9


def test_load_missing_returns_none(tmp_path) -> None:
    assert TasteModel.load(tmp_path / "nope.json") is None


def test_auc_score_perfect_and_chance() -> None:
    y = np.r_[np.ones(5), np.zeros(5)]
    perfect = np.r_[np.ones(5) * 0.9, np.ones(5) * 0.1]
    assert auc_score(y, perfect) == 1.0
    assert abs(auc_score(y, np.full(10, 0.5)) - 0.5) < 1e-9
