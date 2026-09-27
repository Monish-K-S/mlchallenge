"""Evaluate the matcher on a split using the FULL S2/S3 pool.

This is the missing feedback loop that would have caught the near-all-singleton
submission: it runs the real blocking + scoring pipeline against the complete
candidate pool (not the small smoke sample used during training) and reports
macro-F0.5, blocking recall, and the prediction distribution.

For ``--split train`` it scores against ``train_ground_truth.tsv``.  For
``--split test`` there is no ground truth, so it only reports how many links the
pipeline predicts (use this to sanity-check that you are not predicting almost
everything as a singleton).
"""
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
from src.infer import _index_by_id, score_candidates
from src.io_utils import load_ground_truth
from src.metrics import macro_f05


def _subsample(records, max_n, seed):
    if not max_n or max_n >= len(records):
        return records
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=max_n, replace=False)
    return [records[i] for i in sorted(idx.tolist())]


def _blocking_recall(s1_records, candidates, gt):
    pair_hit = pair_total = ent_hit = ent_total = 0
    miss_examples = []
    for r in s1_records:
        truth = set(gt.get(r["entity_id"], []))
        if not truth:
            continue
        cands = set(candidates.get(r["entity_id"], []))
        pair_total += len(truth)
        pair_hit += len(truth & cands)
        ent_total += 1
        if truth & cands:
            ent_hit += 1
        elif len(miss_examples) < 10:
            miss_examples.append((r["entity_id"], sorted(truth)[:3]))
    pr = pair_hit / pair_total if pair_total else 1.0
    er = ent_hit / ent_total if ent_total else 1.0
    return pr, er, (pair_hit, pair_total), miss_examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate ER pipeline on full pool")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--max-eval-s1",
        type=int,
        default=50_000,
        help="Cap number of S1 entities scored (0 = all). Default 50k for a fast signal.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-evidence", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--losses", type=int, default=10, help="Show N FP/FN S1 examples")
    args = parser.parse_args()

    model_path = config.ARTIFACTS / "lgbm_matcher.pkl"
    meta_path = config.ARTIFACTS / "matcher_meta.json"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Run `python -m src.train` first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
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
        f"Model meta: {meta.get('nrows')=} {meta.get('max_train_s1')=} "
        f"stored_threshold={meta.get('threshold')} stored_val_f05={meta.get('val_macro_f05')}"
    )
    print(f"Eval threshold={threshold:.3f} min_evidence={min_evidence}")

    if args.split == "train":
        s1_path, s2_path, s3_path = config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3
    else:
        s1_path, s2_path, s3_path = config.TEST_S1, config.TEST_S2, config.TEST_S3

    t0 = time.perf_counter()
    s1 = load_and_enrich(str(s1_path))
    s1 = _subsample(s1, args.max_eval_s1, config.RANDOM_SEED)
    s2 = load_and_enrich(str(s2_path))
    s3 = load_and_enrich(str(s3_path))
    s23 = s2 + s3
    s23_by_id = _index_by_id(s23)
    print(
        f"Loaded S1={len(s1)} S2+S3={len(s23)} in {time.perf_counter()-t0:.1f}s"
    )

    indexes = build_block_indexes(s23)
    t1 = time.perf_counter()
    cands = generate_candidates(s1, s23, indexes=indexes, n_jobs=args.n_jobs)
    print(f"Blocking done in {time.perf_counter()-t1:.1f}s")

    t2 = time.perf_counter()
    matches = score_candidates(
        s1, s23_by_id, cands, model, threshold, min_evidence=min_evidence
    )
    print(f"Scoring done in {time.perf_counter()-t2:.1f}s")

    n_empty = sum(1 for v in matches.values() if not v)
    n_links = sum(len(v) for v in matches.values())
    print(
        f"Predictions: S1={len(matches)}  singletons={n_empty} "
        f"({n_empty/max(1,len(matches)):.1%})  links={n_links}  "
        f"avg_links/entity={n_links/max(1,len(matches)):.2f}"
    )

    if args.split != "train":
        print("No ground truth for test split — distribution printed above.")
        return

    gt = load_ground_truth(str(config.TRAIN_GT))
    gt = {r["entity_id"]: gt.get(r["entity_id"], []) for r in s1}
    n_singleton_true = sum(1 for v in gt.values() if not v)
    n_links_true = sum(len(v) for v in gt.values())
    print(
        f"Ground truth: singletons={n_singleton_true} "
        f"({n_singleton_true/max(1,len(gt)):.1%})  links={n_links_true}  "
        f"avg_links/entity={n_links_true/max(1,len(gt)):.2f}"
    )

    pr, er, (ph, pt), misses = _blocking_recall(s1, cands, gt)
    print(f"Blocking recall: pair-level={pr:.4f} ({ph}/{pt})  entity-level={er:.4f}")
    if misses:
        print("  blocking misses (S1, some true matches):")
        for sid, truth in misses:
            print(f"    {sid} -> {truth}")

    score = macro_f05(matches, gt)
    print(f"MACRO F0.5 (full pool) = {score:.5f}")

    # Error analysis
    fp_examples, fn_examples = [], []
    for r in s1:
        sid = r["entity_id"]
        pred = set(matches.get(sid, []))
        truth = set(gt.get(sid, []))
        fp = pred - truth
        fn = truth - pred
        if fp and len(fp_examples) < args.losses:
            fp_examples.append((sid, sorted(fp)[:3], sorted(truth)[:3]))
        if fn and len(fn_examples) < args.losses:
            fn_examples.append((sid, sorted(fn)[:3]))
    if fp_examples:
        print("  false positives (predicted IDs, some truth):")
        for sid, fp, truth in fp_examples:
            print(f"    {sid} pred+{fp} truth{truth}")
    if fn_examples:
        print("  false negatives (missed truth IDs):")
        for sid, fn in fn_examples:
            print(f"    {sid} missed{fn}")


if __name__ == "__main__":
    main()