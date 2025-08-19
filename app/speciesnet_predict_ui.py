# speciesnet_predict_ui.py — Streamlit UI for single-image SpeciesNet inference
# ---------------------------------------------------------------------------
# Adds:
# - Display of trained species list from the selected run
# - Optional auto-cropping via your YOLODetector (or smart center-crop / none)
# - Multi-crop selection UI if YOLO returns several boxes
# - Thumbnail + crop preview, then prediction

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sqlalchemy import text
from io import BytesIO
from pathlib import Path
import requests
from tools.spaces import list_objects, generate_signed_url  # uses your existing module
from db.db import SessionLocal
from config.settings import APP_MODE


# --- YOLO integration (your module) -----------------------------------------
# Your yolo_detector.py returns: List[(crop_img, label, (x1,y1,x2,y2))] or a single full-image fallback
try:
    if APP_MODE.lower() == "demo":
        YOLO_OK = False
        YOLODetector = None
    else:
        from tools.yolo_detector import YOLODetector
        YOLO_OK = True
except Exception:
    YOLO_OK = False
    YOLODetector = None

st.caption("Upload an image and score it with your latest trained SpeciesNet model.")

# --- Model helpers -----------------------------------------------------------
@st.cache_resource
def load_model(model_path: str, num_classes: int, device: str = "cpu"):
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    m.load_state_dict(state)
    m = m.to(device)
    m.eval()
    return m

@st.cache_resource
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

IGNORED_KEYS = {"accuracy", "macro avg", "weighted avg", "micro avg"}

def extract_class_names(report: dict) -> list[str]:
    keys = [k for k, v in report.items() if isinstance(v, dict) and k not in IGNORED_KEYS]
    return sorted(keys)

def smart_center_crop(img: Image.Image, padding_ratio: float = 0.10) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    cx, cy = w // 2, h // 2
    half = side // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    # Padding
    pad = int(side * padding_ratio)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img.crop((x1, y1, x2, y2))

def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

@st.cache_resource
def get_yolo(model_path: str | None = None, device: str | None = None):
    """
    Cache the YOLODetector instance.
    """
    if not YOLO_OK:
        return None
    model_path = model_path or st.secrets.get("YOLO_MODEL_PATH", "yolov8n.pt")
    return YOLODetector(model_path=model_path, device=device)

# --- Select a model run ------------------------------------------------------
st.subheader("1) Select a saved model run")
with SessionLocal() as db:
    runs = db.execute(text(
        """
        SELECT model_run_id, model_name, model_version, tag,
               model_path, classification_report, top1_accuracy,
               num_classes, started_at
        FROM wildlife.model_run
        ORDER BY started_at DESC
        LIMIT 20
        """
    )).mappings().all()

if not runs:
    st.warning("No saved model runs found. Train a model first in the Trainer tab.")
    st.stop()

options = []
for r in runs:
    started = r.get("started_at")
    try:
        when = started.strftime("%Y-%m-%d %H:%M") if hasattr(started, "strftime") else str(started)
    except Exception:
        when = str(started)
    label = f"Run {r['model_run_id']} • {r['model_name']} {r['model_version']} • {r.get('tag') or ''} • {r['top1_accuracy']*100:.2f}% • {when}"
    options.append((label, r))

choice = st.selectbox("Choose a run", options=options, format_func=lambda x: x[0])
selected_run = choice[1] if isinstance(choice, tuple) else choice

report = selected_run["classification_report"]
if not isinstance(report, dict) or not report:
    st.error("Selected run is missing a classification_report. Retrain to populate it.")
    st.stop()

class_names = extract_class_names(report)
num_classes = len(class_names)
model_path = selected_run["model_path"]

device = "cuda" if torch.cuda.is_available() else "cpu"
col_a, col_b = st.columns([1,1])
with col_a:
    st.caption("Model path")
    st.code(model_path, language="bash")
with col_b:
    st.caption("Classes")
    st.write(f"{num_classes} species")

# --- Trained species list + nudge -------------------------------------------
st.subheader("2) What this model knows")
with st.expander(f"See trained species list ({num_classes})", expanded=False):
    left, right = st.columns(2)
    mid = (num_classes + 1) // 2
    left.write("\n".join(f"• {n}" for n in class_names[:mid]))
    right.write("\n".join(f"• {n}" for n in class_names[mid:]))

# --- Load classifier ---------------------------------------------------------
try:
    model = load_model(model_path, num_classes=num_classes, device=device)
except Exception as e:
    st.error(f"Failed to load model from {model_path}: {e}")
    st.stop()

preprocess = get_preprocess()

# --- Upload + cropping config -----------------------------------------------
st.subheader("3) Choose an image to classify")

DEMO_PREFIX = "demo/"

@st.cache_data(ttl=60)
def get_demo_image_keys(prefix: str):
    keys = list_objects(prefix=prefix) or []
    valid_ext = (".jpg", ".jpeg", ".png")
    keys = [k for k in keys if k.lower().endswith(valid_ext)]
    keys.sort(key=lambda k: k.lower())
    return keys

image_keys = get_demo_image_keys(DEMO_PREFIX)
labels = ["-- Select an image --"] + [
    k[len(DEMO_PREFIX):] if k.startswith(DEMO_PREFIX) else k
    for k in image_keys
]

options = ["-- Select an image --"] + image_keys
choice = st.selectbox(
    "Select a demo image",
    options,
    index=0,
    format_func=lambda k: (
        k if k == "-- Select an image --"
        else (k[len(DEMO_PREFIX):] if k.startswith(DEMO_PREFIX) else k)
    ),
)

uploaded = None
if choice != "-- Select an image --":
    selected_key = choice  # already the real key
    try:
        url = generate_signed_url(selected_key, expires_in=600)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        uploaded = BytesIO(resp.content)
        uploaded.name = Path(selected_key).name
        uploaded.seek(0)
    except Exception as e:
        st.error(f"Could not fetch image from Spaces: {e}")

crop_method = "YOLO detector (recommended)"
yolo_conf = 0.30

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    # --- Thumbnail only ---
    st.markdown("**Thumbnail**")
    st.image(img, caption="Original", width=160)

    crop_candidates: list[tuple[Image.Image, str, tuple[int,int,int,int]]] = []

    if APP_MODE.lower() == "demo":
        # Cheap + predictable behavior for demo
        if crop_method == "No crop (full image)":
            crop_candidates = [(img, "full", (0, 0, img.width, img.height))]
        else:
            # default to smart center-crop in demo
            crop_candidates = [(smart_center_crop(img), "center", (0, 0, img.width, img.height))]
    else:
        if crop_method == "YOLO detector (recommended)":
            if YOLO_OK:
                det = get_yolo()
                try:
                    crop_candidates = det.detect_and_crop(img, conf_threshold=yolo_conf)  # [(crop, label, bbox), ...]
                except Exception as e:
                    st.warning(f"YOLO failed ({e}). Falling back to center-crop.")
            else:
                st.info("YOLO not available in this environment. Using center-crop fallback.")

        if not crop_candidates:
            if crop_method == "Smart center-crop":
                crop_candidates = [(smart_center_crop(img), "center", (0, 0, *img.size))]
            elif crop_method == "No crop (full image)":
                crop_candidates = [(img, "full", (0, 0, img.width, img.height))]
            else:
                # YOLO requested but returned nothing: try center-crop
                crop_candidates = [(smart_center_crop(img), "center-fallback", (0, 0, *img.size))]

    # Choose best crop (largest area preselected)
    if len(crop_candidates) > 1:
        areas = [bbox_area(bbox) for _, _, bbox in crop_candidates]
        default_idx = int(np.argmax(areas))
        crop_labels = [f"{i+1}: {lbl} [{bbox[2]-bbox[0]}×{bbox[3]-bbox[1]}]" for i, (_, lbl, bbox) in enumerate(crop_candidates)]
        sel_idx = st.selectbox("Multiple crops found — pick one to classify", options=list(range(len(crop_candidates))),
                                   index=default_idx, format_func=lambda i: crop_labels[i])
    else:
        sel_idx = 0

    crop_img, crop_lbl, crop_bbox = crop_candidates[sel_idx]


    # --- Inference on chosen crop -------------------------------------------
    with torch.no_grad():
        x = preprocess(crop_img).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    k = min(5, num_classes)
    top_idx = np.argsort(-probs)[:k]
    top_pairs = [(class_names[i], float(probs[i])) for i in top_idx]
    pred_label, pred_prob = top_pairs[0]

    st.markdown("### Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Top-1", pred_label)
    m2.metric("Confidence", f"{pred_prob*100:.2f}%")
    m3.metric("Crop", "YOLO" if crop_method.startswith("YOLO") else ("Center" if "center" in crop_method.lower() else "None"))

    st.markdown("### Top-5")
    top_df = pd.DataFrame({"Species": [p[0] for p in top_pairs], "Probability": [p[1] for p in top_pairs]})
    st.bar_chart(top_df.set_index("Species"))

    with st.expander("See raw probabilities"):
        raw_df = pd.DataFrame({"Species": class_names, "Prob": probs})
        raw_df = raw_df.sort_values("Prob", ascending=False).reset_index(drop=True)
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

st.info("Retrained a model? Reselect the latest run above to use it here.")
