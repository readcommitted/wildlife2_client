"""
settings.py — Central config for Wildlife Image Processing & Semantic Search

Precedence for config values:
1) Streamlit secrets (if available)
2) Environment variables
3) Sensible defaults

All file paths derive from MEDIA_ROOT.
Secrets (API keys, tokens) should live in .streamlit/secrets.toml for Streamlit.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit secrets (optional) ---
_ST_SECRETS = None
try:
    import streamlit as st  # noqa: F401
    _ST_SECRETS = getattr(st, "secrets", None)
except Exception:
    _ST_SECRETS = None


# --- Helpers ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (adjust if needed)

def from_secrets_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return value from st.secrets[key] if available, else os.getenv(key), else default."""
    if _ST_SECRETS is not None:
        try:
            val = _ST_SECRETS.get(key, None)
            if val is not None:
                return str(val)
        except Exception:
            pass
    return os.getenv(key, default)

def as_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse common truthy strings to bool."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default

def resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """
    Resolve a filesystem path. If absolute or starts with ~, respect it.
    If relative, resolve under `base`.
    """
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base / p)

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# --- Cloud or Local Storage ------------------------------------------------
LOCAL = True

# --- Core Paths ------------------------------------------------------------

MEDIA_ROOT = resolve_path(from_secrets_or_env("MEDIA_ROOT", "media"))

STAGE_DIR            = MEDIA_ROOT / "stage"
RAW_DIR              = MEDIA_ROOT / "raw"
JPG_DIR              = MEDIA_ROOT / "jpg"
IMAGE_DIR            = MEDIA_ROOT / "species_images"

PREDICTIONS_JSON = MEDIA_ROOT / from_secrets_or_env("PREDICTIONS_JSON", "speciesnet_results.json")

# Create dirs that should always exist locally
ensure_dirs(MEDIA_ROOT, STAGE_DIR,  RAW_DIR, JPG_DIR, IMAGE_DIR)


# --- Environment / Services -----------------------------------------------

ENVIRONMENT  = from_secrets_or_env("ENV", "development")
DEBUG        = as_bool(from_secrets_or_env("DEBUG", "false"), default=False)

DATABASE_URL = from_secrets_or_env("DATABASE_URL")

DEFAULT_CONFIDENCE_THRESHOLD = as_float(from_secrets_or_env("DEFAULT_CONFIDENCE_THRESHOLD"), 0.6)

# Wikipedia / HTTP
WIKI_API_URL = from_secrets_or_env("WIKI_API_URL", "https://en.wikipedia.org/w/api.php")
USER_AGENT   = from_secrets_or_env("USER_AGENT", "WildlifeImageBot/1.0 (example@example.com)")
HEADERS      = {"User-Agent": USER_AGENT}

# OpenAI
OPENAI_API_KEY = from_secrets_or_env("OPENAI_API_KEY")
EMBED_MODEL = from_secrets_or_env("EMBED_MODEL")
GPTMODEL = from_secrets_or_env("GPTMODEL")

# Spaces
SPACE_NAME = from_secrets_or_env("SPACE_NAME")
REGION = from_secrets_or_env("REGION")
ACCESS_KEY = from_secrets_or_env("ACCESS_KEY")
SECRET_KEY = from_secrets_or_env("SECRET_KEY")

# Watcher
WATCHER_IMAGES = int(from_secrets_or_env("WATCHER_IMAGES"))
WATCHER_WAIT = int(from_secrets_or_env("WATCHER_WAIT"))

# Demo
APP_MODE = from_secrets_or_env("APP_MODE")