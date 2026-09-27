# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We built a four-stage entity-resolution pipeline — normalize, multi-key blocking,
LightGBM pair scoring, precision-tuned thresholding — targeting the precision-heavy
macro F₀.₅ metric. The central engineering effort went into the feedback loop:
full-pool evaluation and blocking-recall instrumentation that catch the two failure
modes that silently destroy leaderboard score (collapsed probabilities predicting
near-all-singletons, and blocking that drops true matches before scoring).

---

## 2. Methodology

### 2.1 Problem Analysis

- Training covers US and India; the test set adds France, so `country` is treated as
  a free string scope, never a hard-coded class.
- Noise patterns observed: legal-suffix churn (Corp/Corporation, Pvt/Private,
  Ltd/Limited), abbreviation drift (St/Street, Rd/Road), punctuation and `&`/`and`
  variants, word-order swaps, case/whitespace noise, missing PIN/postal codes, and
  landmark-based Indian addresses.
- The official metric is macro F₀.₅ over every Source-1 entity, singletons included
  (a singleton scores 1.0 when predicted empty, 0.0 otherwise). Roughly 94% of
  training S1 entities have ≥1 match at ~3.5 links each, so the prediction
  distribution itself is a first-class check: a submission predicting near-100%
  singletons is definitionally broken.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier
**Core Innovation:** Treating the *evaluation harness*, not the model, as the
critical path — full-pool macro-F₀.₅ reporting, per-key blocking recall, and
FP/FN sampling after every training run.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** exact core name (+country scope), sorted-token name
  signature, phonetic prefix of the first two name tokens, ZIP/PIN + first name
  token, and shared rare name tokens via a DF-capped inverted index.
- **Candidate pairs generated:** up to `MAX_CANDIDATES_PER_S1` (default 200) per S1,
  ranked by a cheap token-overlap + ZIP pre-score; exact-core / exact-sorted / ZIP
  evidence is always retained before the cap so a true match cannot be pruned.
- **How you ensured true matches were not lost:** both pair-level and entity-level
  blocking recall are printed by every training run and by `src/evaluate.py`; the
  target is pair recall ≥ 0.995. `candidate_pairs.tsv` is the exact set fed to the
  scorer.
- **Scaling:** blocking runs in worker processes (`ProcessPoolExecutor`) with
  shared pool state passed once per worker — the previous thread pool gave no
  speedup on this GIL-bound code.

---

## 4. Matching Model

**Features used:**
- Name features: token Jaccard, token F1, partial (containment) ratio, char-3-gram
  Dice, Jaro-Winkler on the core name, sorted-token Jaccard, first-token match,
  phonetic match, length ratio, exact-core / exact-sorted flags.
- Address features: token Jaccard, token F1, partial ratio, char-3-gram Dice,
  Jaro-Winkler, numeric-token Jaccard, length ratio, ZIP equality, street-number
  equality, ZIP presence / ZIP mismatch flags.
- Other: same-country flag (free string comparison, works for unseen countries).

**Model type:** LightGBM binary classifier (MIT license, far below the 8B limit).
Trained with log-loss early stopping — **not** AUC, because a single tree already
reaches AUC≈1 on this task and truncates the model to one stump with near-constant
probabilities. Negatives are hard blocked non-matches sampled per S1 independently
of positives, so true singletons contribute abstain examples.
**Threshold selection method:** sweep 0.30–0.95 (step 0.02) on a held-out S1 split
scored against the *full* candidate pool; pick max macro F₀.₅. An optional
`--min-evidence` guard requires ≥N strong signals before emitting any link.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro):** [fill after the EC2 full-pool run: `python -m src.evaluate --split train --max-eval-s1 0`]
- **Blocking recall (val):** [pair-level / entity-level from training output]
- **Common false positives (wrong merges):** to be filled from `src/evaluate.py --losses` output on the real data — expected classes are same-name different-branch businesses and shared street/ZIP coincidences.
- **Common false negatives (missed matches):** to be filled from the same report — expected classes are heavy transliteration/DBA renames that share no blocking key.

Sanity gate applied before every submission: predicted `avg_links/entity` must be
near the training ~3.5 and singleton rate far from 100%; otherwise the output is
not uploaded.

---

## 6. Conclusion

The deliverable is a reproducible, full-pool-validated blocking + LightGBM pipeline
whose blocking recall, calibration, and threshold are all measured against the real
candidate distribution rather than a small sample. Key lessons: validate on the
full pool, early-stop on log-loss instead of AUC, and let true singletons teach the
model to abstain.

---

## Appendix

### A. Code Artefacts

Runnable pipeline under `code/business_entity_resolution/`: `src/config.py`,
`src/normalize.py`, `src/blocking.py`, `src/features.py`, `src/metrics.py`,
`src/io_utils.py`, `src/train.py`, `src/infer.py`, `src/evaluate.py`, plus
`tests/test_pipeline.py`, pinned `requirements.txt`, and this README. Reproduce:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python -m src.train
python -m src.infer --split test
python -m src.evaluate --split train --max-eval-s1 50000
```

### B. Additional Results

[Attach threshold sweep, feature importances, and error-analysis excerpts from the EC2 run.]
