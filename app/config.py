"""Runtime settings, read once from environment variables (and .env if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Postgres in production, e.g. postgresql+asyncpg://user:pass@localhost:5432/docparse
# Falls back to a local SQLite file so the app runs with zero setup. 
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'docparse.db'}")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

# LLM provider: "anthropic" (Claude), "gemini" (Google) or "ollama" (local Ollama server).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
if LLM_PROVIDER not in {"anthropic", "gemini", "ollama"}:
    raise ValueError(f"LLM_PROVIDER must be 'anthropic', 'gemini' or 'ollama', got {LLM_PROVIDER!r}")
_DEFAULT_MODELS = {"anthropic": "claude-opus-5", "gemini": "gemini-2.5-flash", "ollama": "qwen3.5:9b"}
# LLM_MODEL wins; ANTHROPIC_MODEL is still honoured for the Claude provider.
LLM_MODEL = (os.getenv("LLM_MODEL") or (os.getenv("ANTHROPIC_MODEL") if LLM_PROVIDER == "gemini" else None)
             or _DEFAULT_MODELS[LLM_PROVIDER]).strip()
# Gemini key: GEMINI_API_KEY or GOOGLE_API_KEY. Claude key: ANTHROPIC_API_KEY (read by the SDK).
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
# Ollama: base URL of the local server (no key needed). Default matches a stock Ollama install.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))
# Context window (prompt + output). Ollama defaults to a tiny 4096, which overflows on real
# documents; raise it here. Bigger = more RAM/VRAM, so lower it if the model won't load.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

CLASSIFY_EFFORT = os.getenv("CLASSIFY_EFFORT", "low")
EXTRACT_EFFORT = os.getenv("EXTRACT_EFFORT", "high")
# Claude only: server-side refusal fallbacks (if the primary model declines, the API retries on a fallback model).
LLM_FALLBACKS = _bool("LLM_FALLBACKS", True)

# Pipeline
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
# Below this, the job is still processed but flagged for the reviewer, and a newly discovered type is
# not saved to the type registry (so one uncertain document can't define the schema for later ones).
CLASSIFY_MIN_CONFIDENCE = float(os.getenv("CLASSIFY_MIN_CONFIDENCE", "0.6"))
FIELD_MIN_CONFIDENCE = float(os.getenv("FIELD_MIN_CONFIDENCE", "0.7"))  # per-field, discovered types
AMOUNT_TOLERANCE = float(os.getenv("AMOUNT_TOLERANCE", "0.02"))  # absolute, in currency units
PO_PRICE_TOLERANCE_PCT = float(os.getenv("PO_PRICE_TOLERANCE_PCT", "2.0"))
ANOMALY_MULTIPLIER = float(os.getenv("ANOMALY_MULTIPLIER", "3.0"))

# ERP integration. If ERP_WEBHOOK_URL is empty, approved records go to the built-in mock ERP table.
ERP_WEBHOOK_URL = os.getenv("ERP_WEBHOOK_URL", "").strip()
ERP_API_KEY = os.getenv("ERP_API_KEY", "").strip()
AUTO_EXPORT_VALIDATED = _bool("AUTO_EXPORT_VALIDATED", False)
