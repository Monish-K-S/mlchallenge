"""Train LightGBM matcher on blocked pairs + tune F₀.₅ threshold."""
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
from src.blocking import (
    build_block_indexes,
    generate_candidates,
    load_and_enrich,
    load_linked_smoke_sample,
)
from src.features import FEATURE_NAMES, features_to_array, pair_features
from src.io_utils import load_ground_truth
from src.metrics import macro_f05


def _index_by_id(records):
    return {r["entity_id"]: r for r in records}


def build_training_matrix(
    s1_records,
    s23_by_id,
    candidates,
    ground_truth,
    neg_per_pos: int,
    rng: np.random.Generator,
    max_neg_per_s1: int | None = None,
):
    """Build (X, y) from blocked pairs.

    Negatives are sampled per S1 *independently of positives* so that genuine
    singletons (S1 with no true match) still contribute hard negatives.  The old
    implementation only sampled negatives inside the positive loop, so singleton
    S1 entities contributed nothing and the model never learned the abstain
    decision the macro-F0.5 metric rewards.
    """
    rows, y = [], []
    for s1 in s1_records:
        sid = s1["entity_id"]
        truth = set(ground_truth.get(sid, []))
        cands = candidates.get(sid, [])
        pos = [c for c in cands if c in truth]
        neg_pool = [c for c in cands if c not in truth]

        for pid in pos:
            rec = s23_by_id.get(pid)
            if rec is None:
                continue
            rows.append(features_to_array(pair_features(s1, rec)))
            y.append(1)

        if neg_pool:
            # Scale negatives with positives, but always keep some for singletons
            # so the model sees hard non-matches for entities with no true match.
            take = max(neg_per_pos * max(1, len(pos)), neg_per_pos)
            if max_neg_per_s1 is not None:
                take = min(take, max_neg_per_s1)
            take = min(take, len(neg_pool))
            for nid in rng.choice(neg_pool, size=take, replace=False):
                nrec = s23_by_id.get(nid)
                if nrec is None:
                    continue
                rows.append(features_to_array(pair_features(s1, nrec)))
                y.append(0)

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.int8)
    return np.vstack(rows), np.asarray(y, dtype=np.int8)


def score_all_pairs(s1_records, s23_by_id, candidates, model, batch_size: int = 500_000):
    """Score every blocked pair, returning per-S1 (candidate_id, prob) lists.

    Predicts in batches so peak RAM stays bounded on full-data validation runs
    (the naive version vstacked tens of millions of rows at once).
    """
    out = {s1["entity_id"]: [] for s1 in s1_records}
    pairs = []
    rows = []
    for s1 in s1_records:
        sid = s1["entity_id"]
        for cid in candidates.get(sid, []):
            rec = s23_by_id.get(cid)
            if rec is None:
                continue
            pairs.append((sid, cid))
            rows.append(features_to_array(pair_features(s1, rec)))
            if len(rows) >= batch_size:
                probs = model.predict_proba(np.vstack(rows))[:, 1]
                for (s, c), p in zip(pairs, probs):
                    out[s].append((c, float(p)))
                pairs, rows = [], []
    if rows:
        probs = model.predict_proba(np.vstack(rows))[:, 1]
        for (s, c), p in zip(pairs, probs):
            out[s].append((c, float(p)))
    return out


def predict_map_from_scores(scored, threshold: float):
    return {
        sid: [cid for cid, p in items if p >= threshold]
        for sid, items in scored.items()
    }


def sweep_threshold(scored, gt, thresholds):
    best_t, best_s = thresholds[0], -1.0
    for t in thresholds:
        preds = predict_map_from_scores(scored, t)
        score = macro_f05(preds, gt)
        print(f"  threshold={t:.2f}  macro_F0.5={score:.5f}", flush=True)
        if score > best_s:
            best_s, best_t = score, t
    return best_t, best_s


def main():
    parser = argparse.ArgumentParser(description="Train ER matcher")
    parser.add_argument("--nrows", type=int, default=None, help="Linked smoke sample size")
    parser.add_argument(
        "--max-train-s1",
        type=int,
        default=None,
        help="When training full data, cap S1 count (faster; still uses full S2/S3 pool)",
    )
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--no-cache", action="store_true", help="Skip linked-sample cache")
    args = parser.parse_args()

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)
    t0 = time.perf_counter()

    print("Loading & enriching sources...", flush=True)
    if args.nrows:
        cache_dir = None if args.no_cache else str(config.ARTIFACTS)
        s1, s23, gt = load_linked_smoke_sample(
            str(config.TRAIN_S1),
            str(config.TRAIN_S2),
            str(config.TRAIN_S3),
            str(config.TRAIN_GT),
            n_s1=args.nrows,
            extra_negatives=max(4000, args.nrows * 2),
            cache_dir=cache_dir,
        )
        print(
            f"Linked smoke sample: S1={len(s1)} S2+S3={len(s23)}",
            flush=True,
        )
    else:
        s1 = load_and_enrich(str(config.TRAIN_S1))
        s2 = load_and_enrich(str(config.TRAIN_S2))
        s3 = load_and_enrich(str(config.TRAIN_S3))
        s23 = s2 + s3
        gt_full = load_ground_truth(str(config.TRAIN_GT))
        if args.max_train_s1 and args.max_train_s1 < len(s1):
            pick = rng.choice(len(s1), size=args.max_train_s1, replace=False)
            s1 = [s1[i] for i in pick]
            print(f"Capped train S1 to {len(s1)}", flush=True)
        s1_ids = [r["entity_id"] for r in s1]
        gt = {k: gt_full[k] for k in s1_ids if k in gt_full}
    s23_by_id = _index_by_id(s23)

    perm = rng.permutation(len(s1))
    n_val = max(1, int(len(s1) * args.val_fraction))
    val_idx = set(perm[:n_val].tolist())
    train_s1 = [r for i, r in enumerate(s1) if i not in val_idx]
    val_s1 = [r for i, r in enumerate(s1) if i in val_idx]
    print(
        f"S1 train={len(train_s1)} val={len(val_s1)} | S2+S3 pool={len(s23)} "
        f"| load {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    print("Building block indexes (once)...", flush=True)
    t1 = time.perf_counter()
    indexes = build_block_indexes(s23)
    print(f"Indexes ready in {time.perf_counter()-t1:.1f}s", flush=True)

    print("Blocking candidates (single pass)...", flush=True)
    t2 = time.perf_counter()
    all_cands = generate_candidates(
        train_s1 + val_s1, s23, indexes=indexes, n_jobs=args.n_jobs
    )
    train_ids = {r["entity_id"] for r in train_s1}
    train_cands = {k: v for k, v in all_cands.items() if k in train_ids}
    val_cands = {k: v for k, v in all_cands.items() if k not in train_ids}
    print(f"Blocking done in {time.perf_counter()-t2:.1f}s", flush=True)

    hit, total, ent_hit, ent_total = 0, 0, 0, 0
    for r in val_s1:
        truth = set(gt.get(r["entity_id"], []))
        if not truth:
            continue
        total += len(truth)
        hit += len(truth & set(val_cands.get(r["entity_id"], [])))
        ent_total += 1
        if truth & set(val_cands.get(r["entity_id"], [])):
            ent_hit += 1
    if total:
        print(
            f"Val blocking recall (pair-level): {hit/total:.4f} ({hit}/{total})  "
            f"(entity-level): {ent_hit/ent_total:.4f} ({ent_hit}/{ent_total})",
            flush=True,
        )

    print("Building training matrix...", flush=True)
    X, y = build_training_matrix(
        train_s1,
        s23_by_id,
        train_cands,
        gt,
        config.NEG_PER_POS,
        rng,
        max_neg_per_s1=config.MAX_NEG_PER_S1,
    )
    print(f"Pairs: {len(y)}  positives={int(y.sum())}  negatives={int((1-y).sum())}", flush=True)

    if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
        raise SystemExit(
            f"Not enough labeled pairs to train (n={len(y)}, pos={int(y.sum())}). "
            "Try a larger --nrows."
        )

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required. pip install -r requirements.txt") from exc

    from sklearn.model_selection import train_test_split

    stratify = y if (y.sum() >= 2 and (1 - y).sum() >= 2) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.1, random_state=config.RANDOM_SEED, stratify=stratify
    )
    model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xte, yte)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(
        f"Trained trees={model.booster_.num_trees()} "
        f"(best_iteration={getattr(model, 'best_iteration_', None)})",
        flush=True,
    )

    print("Scoring val + sweeping thresholds...", flush=True)
    scored = score_all_pairs(val_s1, s23_by_id, val_cands, model)
    # Sweep from low to high: the failure mode to guard against is a grid that
    # starts too high and silently pins the model at all-singletons.
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.96, 0.02)]
    val_gt = {r["entity_id"]: gt.get(r["entity_id"], []) for r in val_s1}
    best_t, best_s = sweep_threshold(scored, val_gt, thresholds)
    print(f"Best threshold={best_t:.2f}  val macro_F0.5={best_s:.5f}", flush=True)

    model_path = config.ARTIFACTS / "lgbm_matcher.pkl"
    meta_path = config.ARTIFACTS / "matcher_meta.json"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": best_t,
                "val_macro_f05": best_s,
                "feature_names": FEATURE_NAMES,
                "nrows": args.nrows,
                "max_train_s1": args.max_train_s1,
                "elapsed_sec": round(time.perf_counter() - t0, 1),
            },
            f,
            indent=2,
        )
    print(f"Wrote {model_path}", flush=True)
    print(f"Wrote {meta_path}", flush=True)
    print(f"Total train time: {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
