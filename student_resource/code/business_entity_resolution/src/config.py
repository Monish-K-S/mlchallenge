"""Central paths and hyperparameters for the ER pipeline."""
from pathlib import Path
import os

# Repo root = student_resource/
ROOT = Path(__file__).resolve().parents[3]
# ER_DATA_DIR lets you point at an alternative dataset location (e.g. a fast
# local SSD on EC2, or a tiny synthetic set for smoke tests).
DATA = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset"))
TRAIN = DATA / "train"
TEST = DATA / "test"
# Where submission TSVs are written; override to test without touching output/.
OUTPUT = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))
ARTIFACTS = Path(os.environ.get("ER_ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))

TRAIN_S1 = TRAIN / "train_source1.tsv"
TRAIN_S2 = TRAIN / "train_source2.tsv"
TRAIN_S3 = TRAIN / "train_source3.tsv"
TRAIN_GT = TRAIN / "train_ground_truth.tsv"

TEST_S1 = TEST / "test_source1.tsv"
TEST_S2 = TEST / "test_source2.tsv"
TEST_S3 = TEST / "test_source3.tsv"

MATCHING_OUT = OUTPUT / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT / "candidate_pairs.tsv"

# --- Pipeline knobs (env-overridable; see _env_* helpers) ---
def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"ER_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"ER_{name}")
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"ER_{name}")
    return float(raw) if raw not in (None, "") else default


VAL_FRACTION = _env_float("VAL_FRACTION", 0.05)
RANDOM_SEED = _env_int("RANDOM_SEED", 42)
N_JOBS = _env_int("N_JOBS", max(1, (os.cpu_count() or 4) - 1))

# Blocking.  On the full dataset the pool is ~1.7M S2/S3 records; the previous
# defaults (50 candidates, 600-record bucket cap) silently dropped true matches
# and were the main driver of the near-all-singleton submission.  Defaults are
# now recall-oriented and can be tightened later via env vars.
MAX_CANDIDATES_PER_S1 = _env_int("MAX_CANDIDATES_PER_S1", 200)
MAX_BLOCK_BUCKET = _env_int("MAX_BLOCK_BUCKET", 5000)
BLOCK_RARE_TOKEN_MAX_DF = _env_int("BLOCK_RARE_TOKEN_MAX_DF", 20000)
BLOCK_RARE_TOKEN_MIN_LEN = _env_int("BLOCK_RARE_TOKEN_MIN_LEN", 4)
PHONETIC_PREFIX_TOKENS = _env_int("PHONETIC_PREFIX_TOKENS", 2)
# Metaphone is accurate but slow; prefix codes are much faster for blocking
USE_METAPHONE = _env_str("USE_METAPHONE", "0") not in {"0", "false", "False", ""}

# Matching / training
NEG_PER_POS = _env_int("NEG_PER_POS", 2)
MAX_NEG_PER_S1 = _env_int("MAX_NEG_PER_S1", 4)
INFER_CHUNK_SIZE = _env_int("INFER_CHUNK_SIZE", 25_000)
# Number of S1 records blocked/scored per process-pool task (per worker).
BLOCK_CHUNK_SIZE = _env_int("BLOCK_CHUNK_SIZE", 20_000)
# Minimum number of strong evidence features required to emit any link for an S1
# (0 disables the guard).  Guards against singleton false positives.
MIN_EVIDENCE_FEATURES = _env_int("MIN_EVIDENCE_FEATURES", 0)
LGBM_PARAMS = {
    "objective": "binary",
    # NOTE: do NOT use "auc" here.  On this task a single tree can already reach
    # AUC=1.0, which makes LightGBM's early stopping truncate the model to one
    # tree and leaves predictions almost constant (near-zero links).  Log loss
    # keeps improving, so it is a safe early-stopping signal and calibrates odds.
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 40,
    "verbosity": -1,
    "n_estimators": 800,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
DEFAULT_THRESHOLD = _env_float("DEFAULT_THRESHOLD", 0.72)
