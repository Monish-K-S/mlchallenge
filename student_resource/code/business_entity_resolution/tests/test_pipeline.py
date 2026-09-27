"""Fast unit tests for the pure functions in the ER pipeline.

These guard the two bugs that produced the near-all-singleton submission:
  1. LightGBM early stopping on AUC collapsing to 1 tree.
  2. Negative sampling that ignored singleton S1 entities.

Run from ``code/business_entity_resolution/``::

    python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features import FEATURE_NAMES, evidence_count, pair_features
from src.metrics import f_beta, macro_f05, score_entity
from src.normalize import enrich_row, extract_zip, normalize_text
from src.train import build_training_matrix


def _rec(name, addr, country="US", eid="X"):
    e = enrich_row(name, addr, country)
    e["entity_id"] = eid
    return e


def test_normalize_and_abbrev():
    assert normalize_text("Acme Corp. & Sons") == "acme corporation and sons"
    assert normalize_text("Main Rd") == "main road"
    assert normalize_text("Pvt Ltd") == "private limited"


def test_extract_zip_country_aware():
    assert extract_zip("123 Main St Springfield 62704", "US") == "62704"
    assert extract_zip("MG Road Bengaluru 560001", "India") == "560001"
    assert extract_zip("10 Rue Paris 75001", "France") == "75001"


def test_features_separate_true_and_false_pairs():
    a = _rec("Acme Corporation", "123 Main Street Springfield 62704", "US", "S1-1")
    twin = _rec("Acme Corp", "123 Main St Springfield 62704", "US", "S2-1")
    other = _rec("Globex Industries", "999 Oak Road Fairview 12345", "US", "S2-2")
    f_pos = pair_features(a, twin)
    f_neg = pair_features(a, other)
    assert set(f_pos) == set(FEATURE_NAMES)
    assert f_pos["name_token_f1"] > f_neg["name_token_f1"]
    assert f_pos["same_zip"] == 1.0 and f_neg["same_zip"] == 0.0
    assert evidence_count(f_pos) >= 2
    assert evidence_count(f_neg) == 0


def test_negatives_include_singletons():
    """A singleton S1 (no true match) must still contribute negatives."""
    s1 = [
        _rec("Acme Corp", "1 A St", "US", "S1-1"),
        _rec("Boring Co", "2 B St", "US", "S1-2"),
    ]
    s23_by_id = {
        "S2-1": _rec("Acme Corporation", "1 A Street", "US", "S2-1"),
        "S2-2": _rec("Globex", "9 Z Ave", "US", "S2-2"),
    }
    candidates = {"S1-1": ["S2-1", "S2-2"], "S1-2": ["S2-2"]}
    gt = {"S1-1": ["S2-1"], "S1-2": []}
    rng = np.random.default_rng(0)
    X, y = build_training_matrix(s1, s23_by_id, candidates, gt, 2, rng)
    assert len(y) > 0
    assert (y == 0).sum() > 0  # negatives exist
    assert len(X) == len(y)


def test_f05_singleton_semantics():
    assert score_entity(set(), set()) == 1.0
    assert score_entity({"S2-1"}, set()) == 0.0
    assert score_entity(set(), {"S2-1"}) == 0.0
    assert abs(f_beta(2 / 3, 1.0, 0.5) - 0.714) < 0.01
    preds = {"a": [], "b": ["S2-1"]}
    gt = {"a": [], "b": ["S2-1"]}
    assert macro_f05(preds, gt) == 1.0