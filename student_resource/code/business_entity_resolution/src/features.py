"""Pairwise similarity features for the matching model."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

try:
    from rapidfuzz.distance import JaroWinkler

    def edit_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return JaroWinkler.normalized_similarity(a, b)

except ImportError:
    from difflib import SequenceMatcher

    def edit_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()


def jaccard_sets(sa: frozenset, sb: frozenset) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    return jaccard_sets(frozenset(a), frozenset(b))


def token_f1_sets(sa: frozenset, sb: frozenset) -> float:
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    p = inter / len(sa)
    r = inter / len(sb)
    return 2 * p * r / (p + r)


def dice_sets(sa: frozenset, sb: frozenset) -> float:
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


def _partial_ratio_sets(sa: frozenset, sb: frozenset) -> float:
    """Overlap relative to the smaller set (partial/containment similarity)."""
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _numeric_tokens(tokens) -> frozenset:
    return frozenset(t for t in tokens if t.isdigit())


def _first_token(tokens) -> str:
    return tokens[0] if tokens else ""


def pair_features(left: Dict, right: Dict) -> Dict[str, float]:
    """Build a flat feature dict from two enrich_row() outputs."""
    ln = left["name_token_set"]
    rn = right["name_token_set"]
    la = left["addr_token_set"]
    ra = right["addr_token_set"]
    same_country = float(
        bool(left["country"])
        and bool(right["country"])
        and left["country"].lower() == right["country"].lower()
    )
    same_zip = float(bool(left["zip"]) and left["zip"] == right["zip"])
    same_street = float(
        bool(left["street_no"]) and left["street_no"] == right["street_no"]
    )
    phon_match = float(
        bool(left["phonetic"]) and left["phonetic"] == right["phonetic"]
    )
    exact_core = float(
        left["core_name"] == right["core_name"] and bool(left["core_name"])
    )
    exact_sorted = float(
        left["sorted_name"] == right["sorted_name"] and bool(left["sorted_name"])
    )
    first_name_match = float(
        bool(_first_token(left["name_tokens"]))
        and _first_token(left["name_tokens"]) == _first_token(right["name_tokens"])
    )
    num_overlap = jaccard_sets(
        _numeric_tokens(left["addr_tokens"]), _numeric_tokens(right["addr_tokens"])
    )

    return {
        "name_jaccard": jaccard_sets(ln, rn),
        "name_token_f1": token_f1_sets(ln, rn),
        "name_partial_ratio": _partial_ratio_sets(ln, rn),
        "name_dice3": dice_sets(left["name_ngrams"], right["name_ngrams"]),
        "name_edit": edit_ratio(left["core_name"], right["core_name"]),
        "sorted_name_jaccard": jaccard(
            left["sorted_name"].split(), right["sorted_name"].split()
        ),
        "first_name_token_match": first_name_match,
        "addr_jaccard": jaccard_sets(la, ra),
        "addr_token_f1": token_f1_sets(la, ra),
        "addr_partial_ratio": _partial_ratio_sets(la, ra),
        "addr_dice3": dice_sets(left["addr_ngrams"], right["addr_ngrams"]),
        "addr_edit": edit_ratio(left["addr_norm"], right["addr_norm"]),
        "addr_numeric_jaccard": num_overlap,
        "same_country": same_country,
        "same_zip": same_zip,
        "same_street_no": same_street,
        "phonetic_match": phon_match,
        "exact_core_name": exact_core,
        "exact_sorted_name": exact_sorted,
        "name_len_ratio": _len_ratio(left["core_name"], right["core_name"]),
        "addr_len_ratio": _len_ratio(left["addr_norm"], right["addr_norm"]),
        "both_have_zip": float(bool(left["zip"] and right["zip"])),
        "zip_mismatch": float(
            bool(left["zip"] and right["zip"] and left["zip"] != right["zip"])
        ),
    }


# Strong positive signals used by the optional singleton guard.  A predicted
# link for an S1 entity that satisfies none of these is likely a false merge.
EVIDENCE_FEATURES = (
    "exact_core_name",
    "exact_sorted_name",
    "same_zip",
    "same_street_no",
    "phonetic_match",
)


def evidence_count(feats: Dict[str, float]) -> int:
    return sum(1 for k in EVIDENCE_FEATURES if feats.get(k, 0.0) >= 0.5)


FEATURE_NAMES = [
    "name_jaccard",
    "name_token_f1",
    "name_partial_ratio",
    "name_dice3",
    "name_edit",
    "sorted_name_jaccard",
    "first_name_token_match",
    "addr_jaccard",
    "addr_token_f1",
    "addr_partial_ratio",
    "addr_dice3",
    "addr_edit",
    "addr_numeric_jaccard",
    "same_country",
    "same_zip",
    "same_street_no",
    "phonetic_match",
    "exact_core_name",
    "exact_sorted_name",
    "name_len_ratio",
    "addr_len_ratio",
    "both_have_zip",
    "zip_mismatch",
]


def features_to_vector(feats: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(float(feats[k]) for k in FEATURE_NAMES)


def features_to_array(feats: Dict[str, float]):
    import numpy as np

    return np.fromiter(
        (float(feats[k]) for k in FEATURE_NAMES), dtype=np.float32, count=len(FEATURE_NAMES)
    )


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def cheap_pre_score(left: Dict, right: Dict) -> float:
    """Fast ranker for blocking — token overlap + zip only (no n-grams)."""
    ln, rn = left["name_token_set"], right["name_token_set"]
    if not ln or not rn:
        name_j = 0.0
    else:
        name_j = len(ln & rn) / len(ln | rn)
    la, ra = left["addr_token_set"], right["addr_token_set"]
    if not la or not ra:
        addr_j = 0.0
    else:
        addr_j = len(la & ra) / len(la | ra)
    zip_bonus = 1.0 if (left["zip"] and left["zip"] == right["zip"]) else 0.0
    exact = 1.0 if left["core_name"] and left["core_name"] == right["core_name"] else 0.0
    return 0.50 * name_j + 0.25 * addr_j + 0.15 * zip_bonus + 0.10 * exact
