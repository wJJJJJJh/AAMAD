"""Configuration matching the manuscript; importing this module performs no I/O."""
from __future__ import annotations
import os
from pathlib import Path

def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).lower() in {"1", "true", "yes"}

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AAMAD_DATA_DIR", str(ROOT.parent / "data")))
TRAIN_XML, TRAIN_PDF = DATA_DIR / "train/XML", DATA_DIR / "train/PDF"
TEST_XML, TEST_PDF = DATA_DIR / "test/XML", DATA_DIR / "test/PDF"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
CACHE_DIR = Path(os.environ.get("AAMAD_CACHE_DIR", str(ROOT / ".cache")))
OUT_DIR = Path(os.environ.get("AAMAD_OUT_DIR", str(ROOT / "outputs")))
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_KEY = LLM_API_KEY
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4096
LLM_CACHE = _flag("LLM_CACHE", True)
LLM_OFFLINE = _flag("LLM_OFFLINE")
LLM_TIMEOUT, LLM_MAX_RETRY = 60, 3
DATACITE_ENABLED = _flag("DATACITE_ENABLED", True)
DATACITE_TIMEOUT = 20
SUPPRESS_CONFIDENCE_THRESHOLD = 0.85  # Strictly greater than, not >=.
CONTEXT_TOKEN_LIMIT = 900
FOCUS_CHARS = 900
MAX_FULLTEXT_CHARS = 50000
DISCOVERY_BATCH_SIZE = 20
BULK_FILTER_THRESHOLD = 50
ABLATION: dict = {}
ABLATION_PRESETS = {
    "discovery_only": {"SKIP_DEBATE": True},
    "conventional_debate": {"ANNOTATION_ALIGNED": False},
    "AAMAD": {"ANNOTATION_ALIGNED": True},
}
def apply_ablation(name: str) -> dict:
    if name not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation: {name}")
    ABLATION.clear()
    ABLATION.update(ABLATION_PRESETS[name])
    return dict(ABLATION)
