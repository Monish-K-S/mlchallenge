"""Run trained matcher on test (or train) and write submission TSVs."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.blocking import build_block_indexes, generate_candidates, load_and_enrich
from src.features import (
    FEATURE_NAMES,
    evidence_count,
    features_to_array,
    pair_features,
)
from src.io_utils import write_id_map_rows


def _index_by_id(records):
    return {r["entity_id"]: r for r in records}


def _check_model_compat(model) -> None:
    """Fail loudly if the pickled model predates a feature-set change."""
    n_model = getattr(model, "n_features_in_", None)
    if n_model is not None and n_model != len(FEATURE_NAMES):
        raise SystemExit(
            f"Model expects {n_model} features but this code builds "
            f"{len(FEATURE_NAMES)}. The trained artifact is stale — retrain with "
            f"`python -m src.train` before running inference."
        )


def score_candidates(
    s1_records,
    s23_by_id,
    candidates,
    model,
    threshold: float,
    min_evidence: int = 0,
):
    pairs = []
    rows = []
    evidence = []
    for s1 in s1_records:
        sid = s1["entity_id"]
        for cid in candidates.get(sid, []):
            rec = s23_by_id.get(cid)
            if rec is None:
                continue
            feats = pair_features(s1, rec)
            pairs.append((sid, cid))
            rows.append(features_to_array(feats))
            evidence.append(evidence_count(feats))
    matches = {s1["entity_id"]: [] for s1 in s1_records}
    if not rows:
        return matches
    # Predict in batches to cap peak RAM
    probs = np.empty(len(rows), dtype=np.float32)
    batch = 100_000
    for i in range(0, len(rows), batch):
        X = np.vstack(rows[i : i + batch])
        probs[i : i + len(X)] = model.predict_proba(X)[:, 1]
    for (sid, cid), p, ev in zip(pairs, probs, evidence):
        if float(p) < threshold:
            continue
        if min_evidence and ev < min_evidence:
            continue
        matches[sid].append(cid)
    return matches


def main():
    parser = argparse.ArgumentParser(description="Infer ER matches")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--nrows", type=int, default=None, help="Dev subsample")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.INFER_CHUNK_SIZE,
        help="S1 entities per blocking/scoring chunk",
    )
    parser.add_argument("--matching-out", type=Path, default=config.MATCHING_OUT)
    parser.add_argument("--candidate-out", type=Path, default=config.CANDIDATE_OUT)
    parser.add_argument(
        "--min-evidence",
        type=int,
        default=None,
        help="Require at least N strong evidence features (exact core/zip/street/"
        "phonetic) for a link. 0 disables. Default from config.MIN_EVIDENCE_FEATURES.",
    )
    args = parser.parse_args()

    model_path = config.ARTIFACTS / "lgbm_matcher.pkl"
    meta_path = config.ARTIFACTS / "matcher_meta.json"
    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run: python -m src.train --nrows 5000 first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    _check_model_compat(model)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(meta.get("threshold", config.DEFAULT_THRESHOLD))
    )
    min_evidence = (
        args.min_evidence
        if args.min_evidence is not None
        else config.MIN_EVIDENCE_FEATURES
    )
    print(
        f"Using threshold={threshold:.3f}  min_evidence={min_evidence}",
        flush=True,
    )

    if args.split == "test":
        s1_path, s2_path, s3_path = config.TEST_S1, config.TEST_S2, config.TEST_S3
    else:
        s1_path, s2_path, s3_path = config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3

    t0 = time.perf_counter()
    print("Loading sources...", flush=True)
    s1 = load_and_enrich(str(s1_path), nrows=args.nrows)
    s2 = load_and_enrich(str(s2_path), nrows=args.nrows)
    s3 = load_and_enrich(str(s3_path), nrows=args.nrows)
    s23 = s2 + s3
    s23_by_id = _index_by_id(s23)
    s1_ids = [r["entity_id"] for r in s1]
    print(
        f"Loaded S1={len(s1)} S2+S3={len(s23)} in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    print("Building S2/S3 block indexes once...", flush=True)
    t1 = time.perf_counter()
    indexes = build_block_indexes(s23)
    print(f"Indexes in {time.perf_counter()-t1:.1f}s", flush=True)

    # Stream outputs per chunk so peak RAM stays bounded: holding every
    # candidate/match list for ~1.7M S1 entities would cost tens of GB.
    chunk = max(1000, args.chunk_size)
    n_chunks = (len(s1) + chunk - 1) // chunk
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    args.matching_out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = n_links = 0
    with (
        open(args.candidate_out, "w", encoding="utf-8", newline="\n") as fc,
        open(args.matching_out, "w", encoding="utf-8", newline="\n") as fm,
    ):
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for ci, start in enumerate(range(0, len(s1), chunk), 1):
            part = s1[start : start + chunk]
            part_ids = [r["entity_id"] for r in part]
            t2 = time.perf_counter()
            cands = generate_candidates(
                part, s23, indexes=indexes, n_jobs=args.n_jobs
            )
            matches = score_candidates(
                part, s23_by_id, cands, model, threshold, min_evidence=min_evidence
            )
            write_id_map_rows(fc, part_ids, cands)
            write_id_map_rows(fm, part_ids, matches)
            n_empty += sum(1 for sid in part_ids if not matches.get(sid))
            n_links += sum(len(matches.get(sid, ())) for sid in part_ids)
            print(
                f"  chunk {ci}/{n_chunks}: {len(part)} S1 in {time.perf_counter()-t2:.1f}s",
                flush=True,
            )
    print(f"Wrote {args.candidate_out}", flush=True)
    print(f"Wrote {args.matching_out}", flush=True)

    print(
        f"S1={len(s1_ids)}  singletons={n_empty}  links={n_links}  "
        f"total {time.perf_counter()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
