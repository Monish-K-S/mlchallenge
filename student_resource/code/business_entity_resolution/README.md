# Business Entity Resolution — Runnable Pipeline

Target metric: **macro F₀.₅** (precision-heavy). See `APPROACH.md` for the strategy toward ~0.98.

## Layout

```
business_entity_resolution/
├── APPROACH.md          # how we aim for ~0.98 F₀.₅
├── README.md            # this file
├── requirements.txt
├── artifacts/           # trained model + threshold (created by train)
└── src/
    ├── config.py
    ├── normalize.py
    ├── blocking.py
    ├── features.py
    ├── metrics.py
    ├── io_utils.py
    ├── train.py
    ├── infer.py
    └── evaluate.py       # full-pool macro-F0.5 + blocking recall (the feedback loop)
```

Outputs are written to `student_resource/output/`:
- `matching_results.tsv` — leaderboard file
- `candidate_pairs.tsv` — last blocking set fed to the model

## Setup

From this folder:

```bash
cd student_resource/code/business_entity_resolution
python -m pip install -r requirements.txt
```

## Quick smoke test (small subsample)

```bash
# Train on a linked slice (cached after first run — much faster on repeat)
python -m src.train --nrows 3000

# Infer on a test slice
python -m src.infer --split test --nrows 3000
```

Second `--nrows 3000` train reuses `artifacts/linked_smoke_s1*.pkl`.

## Faster full-ish train (recommended before full infer)

Train the matcher on a capped S1 sample while still using the full S2/S3 pool:

```bash
python -m src.train --max-train-s1 100000
python -m src.infer --split test --n-jobs 8 --chunk-size 25000
```

## Full run

```bash
python -m src.train
python -m src.infer --split test
```

## Evaluate before submitting (do this every time)

`src.evaluate` is the feedback loop that catches the classic failure mode where the
pipeline predicts almost everything as a singleton. It runs blocking + scoring over
the **full** S2/S3 pool and reports macro-F0.5, blocking recall, and how many links
you predict (vs the ground-truth distribution):

```bash
# Score on a held-out slice of the training pool (has ground truth)
python -m src.evaluate --split train --max-eval-s1 50000

# Just check predicted-link distribution on test (no ground truth)
python -m src.evaluate --split test --max-eval-s1 50000
```

If `avg_links/entity` is far below the training average (~3.5) or singletons are
near 100%, your threshold/model is wrong — do **not** submit.

## Speed / behaviour knobs (env vars or `config.py`)

| Flag / setting | Effect |
|----------------|--------|
| `--nrows N` | Linked smoke sample + disk cache (train only) |
| `--max-train-s1 N` | Cap S1 when training on full files |
| `--n-jobs` | Blocking worker **processes** |
| `--chunk-size` | Infer S1 batch size |
| `--min-evidence N` (infer) | Require N strong features to emit a link |
| `ER_USE_METAPHONE=1` | Enable Double Metaphone blocking key (slower) |
| `ER_MAX_CANDIDATES_PER_S1` | Candidates kept per S1 (default 200) |
| `ER_MAX_BLOCK_BUCKET` | Max records per blocking bucket (default 5000) |
| `ER_MIN_EVIDENCE_FEATURES` | Default for `--min-evidence` (0 = off) |
| `ER_DATA_DIR` / `ER_OUTPUT_DIR` / `ER_ARTIFACTS_DIR` | Relocate data/outputs/artifacts |
## Validate submission (required before Portal upload)

Run from **`student_resource/`** (not from `code/`):

### Windows (PowerShell)

```powershell
cd E:\mlchallenge\student_resource

# Fast format check (recommended every time)
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test

# Stricter: also verify every S2/S3 ID exists (uses more RAM)
python utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test `
  --check-ids
```

### Linux / macOS

```bash
cd student_resource

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

You want: **`PASS — no blocking issues found. Safe to submit.`**

Notes from the official validator:
- Exit `0` = safe; exit `1` = fix listed errors
- `--check-ids` is optional (diagnostic; loads all S2/S3 IDs into memory)
- Matching IDs absent from `candidate_pairs.tsv` only **warn** (still fix them)

## Leaderboard upload

Upload only:

`student_resource/output/matching_results.tsv`

## Path to ~0.98 F₀.₅

1. Confirm val blocking recall ≥ 0.99 (`src.train` prints it)
2. Sweep threshold for max macro F₀.₅ (already in `src.train`)
3. Raise `MAX_CANDIDATES_PER_S1` / add blocking keys if recall is low
4. Add hard-negative mining + more features if precision is low
5. Optional: offline embedding cosine feature (Apache MiniLM) as Phase-2

Details: `APPROACH.md`.
