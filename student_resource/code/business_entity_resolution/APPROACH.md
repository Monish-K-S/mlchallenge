# Approach: Targeting F₀.₅ ≈ 0.98

> Official metric is **F₀.₅** (precision-heavy), not classic F1.  
> Macro-averaged over every Source-1 entity, including singletons.

## Why 0.98 is hard (and reachable)

| Fact | Implication |
|------|-------------|
| ~2.2M train / ~1.7M test S1 entities | Blocking must be O(N log N), not O(N²) |
| ~94% of S1 have ≥1 match; ~avg 3.5 matches | Recall still matters, but **false merges kill F₀.₅** |
| Names/addresses are noisy but often near-duplicates | Strong normalization + similarity features work |
| Test adds **France** (unseen in train) | Never hard-code `{US, India}` |
| Singletons score 1.0 if empty, 0.0 if any wrong match | Prefer abstaining over guessing |

To reach **~0.98**, you roughly need:
- Blocking recall ≥ **0.995** (true match must enter the candidate set)
- Matching precision ≥ **0.99** at operating threshold
- Near-perfect singleton handling

## Pipeline (4 stages)

```
1. Normalize  →  cleaned name/address + extracted geo tokens
2. Block      →  multi-key candidate generation  →  candidate_pairs.tsv
3. Score      →  LightGBM on pair features (+ optional embedding cosine)
4. Decide     →  precision-tuned threshold → matching_results.tsv
```

### Stage 1 — Normalize
- Lowercase, strip punctuation, unify `&`/`and`
- Expand Rd/St/Ave, Pvt/Ltd/Corp, remove legal suffixes for a *core name*
- Extract zip/PIN, city-ish tokens, street numbers
- Keep `country` as a free string (works for France)

### Stage 2 — Multi-key blocking (recall ceiling)
Union of several cheap keys (country enforced as key scope):

1. Exact core-name + country  
2. Sorted-token name signature + country (handles word-order swaps like `O.D., Sofie Greenman`)  
3. Phonetic prefix of first 2 name tokens + country (Double Metaphone optional, off by default)  
4. ZIP/PIN + first name token  
5. Shared rare name tokens (DF-capped inverted index)

Cap candidates per S1 (default 200, env `ER_MAX_CANDIDATES_PER_S1`) with exact/zip
evidence always retained before cheap pre-score ranking, so a true match is never
pruned by the cap. Whatever is fed to the model **is** `candidate_pairs.tsv`.
Blocking runs in worker **processes** (`ProcessPoolExecutor`), not threads, because
the ranking is GIL-bound pure Python.

### Stage 3 — Pair features + LightGBM
Features (fast, no external lookup):
- Name: Jaccard, token F1, partial (containment) ratio, char 3-gram Dice, Jaro-Winkler ratio, sorted-token Jaccard, first-token match, phonetic match
- Address: same + ZIP equality, street-number equality, numeric-token Jaccard, token overlap
- Meta: same country, length ratios, exact-normalized match flags, zip-presence / zip-mismatch flags

Model: **LightGBM** binary classifier (MIT license, tiny vs 8B limit).
Lesson learned: never early-stop on AUC here — a single tree already reaches AUC≈1,
which truncates the model to one tree and collapses probabilities to near-constant
(the near-all-singleton submission). We train with `binary_logloss` early stopping,
which stays informative and calibrates odds.
Optional Phase-2: small Apache sentence embedding (e.g. `all-MiniLM-L6-v2`) cosine as an extra feature / reranker — only if you can run it locally offline after one Hub download during development (no live external lookup at inference time beyond the licensed model weights you ship).

### Stage 4 — Threshold for F₀.₅
- Hold out ~5% of S1 IDs for validation
- Sweep probability thresholds (0.30–0.95, step 0.02); pick max **macro F₀.₅**
- Always validate on the **full** S2/S3 pool via `src/evaluate.py` — validating on a
  small linked sample hid the under-matching bug that produced the first submission
- Optional `--min-evidence` guard: require ≥N strong features (exact core/zip/street/
  phonetic) before emitting any link, to protect singleton precision
- Bias slightly toward higher threshold if public LB is noisy (precision > recall)

## Training recipe

1. Build positives from `train_ground_truth.tsv` (only those present in the
   candidate set; the rest are counted as blocking-recall misses)  
2. Hard negatives = blocked non-matches (not random distant pairs)  
3. Sample ~2 negatives per positive with a per-S1 cap — sampled per S1
   *independently of positives* so true singletons still contribute hard negatives  
4. Train LightGBM with early stopping on validation **log-loss** (not AUC)  
5. Calibrate / threshold on held-out S1 entities against the **full** pool  
6. Full retrain on all train (optional) with frozen threshold from val

## Iteration path to 0.98

| Step | Focus | Target |
|------|-------|--------|
| A | Baseline normalize + name blocking + rules | F₀.₅ ≥ 0.70 |
| B | Multi-key blocking + LightGBM | ≥ 0.90 |
| C | Better features, hard-neg mining, threshold | ≥ 0.95 |
| D | Embedding rerank + error analysis on FPs | ≥ 0.98 |

**Error analysis loop (critical):**
- False positives → raise threshold / add mismatch features (different ZIP, different street #)
- False negatives → expand blocking keys / lower pre-filter
- Singleton FPs → require higher score when S1 has weak evidence

## Fair play
No geocoding APIs, business registries, or external ER services.  
Only provided TSVs + licensed local models ≤ 8B params (MIT/Apache 2.0).
