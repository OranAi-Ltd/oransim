"""Repository-local paths for optional research commands."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
KUAIRAND_RAW = DATA_DIR / "kuairand" / "raw"
KUAIRAND_PROCESSED = DATA_DIR / "kuairand" / "processed"
