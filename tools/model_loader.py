# tools/model_loader.py
import os, hashlib, tempfile, requests
from pathlib import Path
import streamlit as st
from config.settings import (
    SPACES_BUCKET, SPACES_ENDPOINT, SPACES_REGION,
    SPACES_ACCESS_KEY_ID, SPACES_SECRET_ACCESS_KEY,
)
import boto3
from botocore.client import Config

DEMO = os.getenv("APP_MODE", "full").lower() == "demo"
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/wildlife"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

def _s3():
    return boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=SPACES_ACCESS_KEY_ID,
        aws_secret_access_key=SPACES_SECRET_ACCESS_KEY,
        config=Config(s3={"addressing_style": "virtual"}),
    )

def _download_to_tmp_from_spaces(key: str) -> Path:
    dest = TMP_DIR / Path(key).name
    if dest.exists():
        return dest
    _s3().download_file(SPACES_BUCKET, key, str(dest))
    return dest

def _download_to_tmp_from_url(url: str) -> Path:
    dest = TMP_DIR / Path(url.split("?")[0]).name
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    return dest

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

@st.cache_resource(show_spinner=True)
def load_speciesnet(model_key_or_url: str, expected_sha256: str | None = None):
    if DEMO:
        st.info("Demo mode: skipping heavy model load; using stub predictor.")
        return None

    # Resolve to local temp file
    if model_key_or_url.startswith("s3://"):
        # s3://bucket/key
        _, _, bucket_key = model_key_or_url.partition("s3://")
        bucket, _, key = bucket_key.partition("/")
        path = _download_to_tmp_from_spaces(key if bucket == SPACES_BUCKET else bucket_key)
    elif model_key_or_url.startswith("http"):
        path = _download_to_tmp_from_url(model_key_or_url)
    else:
        # treat as Spaces key
        path = _download_to_tmp_from_spaces(model_key_or_url)

    if expected_sha256:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Model checksum mismatch: {actual} != {expected_sha256}")

    # Lazy import torch here to keep demo build tiny
    import torch
    model = torch.load(path, map_location="cpu")
    if hasattr(model, "eval"):
        model.eval()
    return model
