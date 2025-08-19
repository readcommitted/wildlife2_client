"""
embedding_utils.py — Image Embedding Generator (CLIP Model)
------------------------------------------------------------

This module provides utilities to generate image embeddings using OpenAI's CLIP model (ViT-B/32),
used for species identification, semantic search, and similarity comparisons within the Wildlife Vision System.

Features:
- Loads CLIP model (GPU if available, fallback to CPU)
- Generates normalized 1024-dimensional image embeddings
- Supports standardized preprocessing for all images

Dependencies:
- PyTorch for tensor operations
- PIL for image handling
- Open CLIP model

"""
# embedding_utils.py
import os
from functools import lru_cache
from typing import Optional, Tuple, Union
import torch
import open_clip
from PIL import Image
import numpy as np
from tools.spaces import download_from_spaces_to_temp


# --- Model config ---
MODEL_NAME = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

if DEVICE == "cuda":
    # modest perf boost for matmul kernels
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

_model = None
_preprocess = None
_tokenizer = None


def ensure_model_loaded() -> Tuple[torch.nn.Module, callable, callable]:
    """Load CLIP model/transforms/tokenizer once and cache globally."""
    global _model, _preprocess, _tokenizer
    if _model is None or _preprocess is None or _tokenizer is None:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model = model.to(DEVICE)
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer
    return _model, _preprocess, _tokenizer


def warm_openclip(batch_size: int = 1) -> None:
    """Optional one-time warmup to JIT kernels & cudnn. Call at app start."""
    model, preprocess, _ = ensure_model_loaded()
    # Build a dummy image and run one forward pass
    dummy = Image.new("RGB", (224, 224), color=0)
    x = preprocess(dummy).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(DEVICE)
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model.encode_image(x)
        else:
            _ = model.encode_image(x)


def get_openclip_model():
    """Backcompat shim if other modules import this name."""
    return ensure_model_loaded()


def _resolve_path_if_remote(image_path: Union[str, os.PathLike]) -> str:
    """Download from Spaces only when the path isn’t absolute/local."""
    if isinstance(image_path, str) and not os.path.isabs(image_path):
        # heuristic for your setup; tweak if you pass 'stage/...' etc.
        return download_from_spaces_to_temp(image_path)
    return str(image_path)


def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Coerce input to PIL.Image without extra copies when possible."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        # assume RGB uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img, mode="RGB")
    # else treat as path
    path = _resolve_path_if_remote(img)
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def generate_openclip_image_embedding(
    image: Union[str, Image.Image, np.ndarray],
    *,
    model: Optional[torch.nn.Module] = None,
    preprocess: Optional[callable] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a normalized 1024-D embedding for an image input (path, PIL, or numpy).
    You can pass (model, preprocess, device) to avoid global lookups.
    """
    mdl, prep, _ = (model, preprocess, device)
    if mdl is None or prep is None or device is None:
        mdl, prep, _ = ensure_model_loaded()
        device = DEVICE

    pil = _to_pil(image)
    x = prep(pil).unsqueeze(0).to(device, non_blocking=(device == "cuda"))

    # Use inference_mode (faster + no grad state) and AMP on CUDA
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = mdl.encode_image(x)
        else:
            feats = mdl.encode_image(x)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().reshape(-1)


def generate_openclip_text_embedding(
    text: str,
    *,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[callable] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    mdl, _, tok = (model, None, tokenizer)
    if mdl is None or tok is None or device is None:
        mdl, _, tok = ensure_model_loaded()
        device = DEVICE

    tokens = tok([text]).to(device)
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = mdl.encode_text(tokens)
        else:
            feats = mdl.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().reshape(-1)
