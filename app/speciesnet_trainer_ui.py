# train_speciesnet_ui.py — Streamlit UI for Training SpeciesNet

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import text

from tools.speciesnet_trainer import train_and_evaluate_speciesnet
from db.db import SessionLocal

from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

st.title("Train & Evaluate SpeciesNet")

# --- 1) Load labeled images from DB ---
st.markdown("### 1. Load labeled images")
with SessionLocal() as db:
    rows = db.execute(text("""
        SELECT ih.image_id, ih.jpeg_path, il.label_value AS common_name
        FROM wildlife.image_label il
        JOIN wildlife.image_header ih ON il.image_id = ih.image_id
        WHERE il.label_value IS NOT NULL
          AND ih.jpeg_path IS NOT NULL
    """)).mappings().all()

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No labeled images found. Add labels and try again.")
    st.stop()

species_counts = df["common_name"].value_counts()
st.success(f"Found {len(df)} labeled images across {len(species_counts)} species")
species_df = species_counts.reset_index()
species_df.columns = ["species", "count"]
st.dataframe(species_df, use_container_width=True, hide_index=True)

# --- 2) Hyperparameters ---
st.markdown("### 2. Set Training Options")
epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=1e-4, format="%.6f")
batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
model_tag = st.text_input("Optional Model Tag", value="")

# --- helpers for rendering ---
def _per_class_rows(report_dict: dict, labels: list[str]) -> pd.DataFrame:
    rows = []
    for cls in labels:
        m = report_dict.get(cls, {})
        rows.append({
            "Species": cls,
            "Precision": round(m.get("precision", 0.0), 3),
            "Recall": round(m.get("recall", 0.0), 3),
            "F1": round(m.get("f1-score", 0.0), 3),
            "Support": int(m.get("support", 0)),
        })
    if "macro avg" in report_dict:
        rows.append({"Species": "—", "Precision": None, "Recall": None, "F1": None, "Support": None})
        rows.append({
            "Species": "Macro Avg",
            "Precision": round(report_dict["macro avg"]["precision"], 3),
            "Recall": round(report_dict["macro avg"]["recall"], 3),
            "F1": round(report_dict["macro avg"]["f1-score"], 3),
            "Support": int(report_dict["macro avg"]["support"]),
        })
    if "weighted avg" in report_dict:
        rows.append({
            "Species": "Weighted Avg",
            "Precision": round(report_dict["weighted avg"]["precision"], 3),
            "Recall": round(report_dict["weighted avg"]["recall"], 3),
            "F1": round(report_dict["weighted avg"]["f1-score"], 3),
            "Support": int(report_dict["weighted avg"]["support"]),
        })
    return pd.DataFrame(rows)

def _fmt_top5(t5):
    return " | ".join([f"{lbl} ({p:.2f})" for lbl, p in t5])

# --- 3) Train + evaluate ---
if st.button("Train and Evaluate Model", type="primary"):
    with st.spinner("Training in progress..."):
        result = train_and_evaluate_speciesnet(
            df=df,
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            tag=model_tag.strip()
        )

    # Header + IDs
    st.success(f"Training complete! Model saved to `{result['model_path']}`")
    model_run_id = result.get("model_run_id")
    if model_run_id:
        st.info(f"Run ID: {model_run_id}")

    # Topline metrics
    col1, col2 = st.columns(2)
    col1.metric("Top-1 Accuracy", f"{result['val_acc']*100:.2f}%")
    col2.metric("Top-5 Accuracy", f"{result['top5_acc']*100:.2f}%")

    # Confusion matrix
    labels = result["labels"]
    cm = np.array(result["confusion_matrix"], dtype=np.int32)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    st.markdown("### Confusion Matrix (Validation)")
    fig_cm = px.imshow(
        cm_df, text_auto=True, aspect="equal",
        labels=dict(x="Predicted", y="True", color="Count"),
    )
    fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=700)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class table
    st.markdown("### Per-Class Metrics")
    rep_df = _per_class_rows(result["classification_report"], labels)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

    # Top-5 inspection
    st.markdown("### Top-5 Inspection")
    preds_df = result["predictions"].copy()
    if "top5" not in preds_df.columns:
        preds_df["top5"] = ""
    if "match" in preds_df.columns:
        preds_df["Correct"] = preds_df["match"].map(lambda x: "✅" if x else "❌")
    elif "Correct" not in preds_df.columns:
        preds_df["Correct"] = ""

    tab1, tab2 = st.tabs(["Misses (Top-1 incorrect)", "All Validation Samples"])
    with tab1:
        miss_df = preds_df[preds_df["Correct"] == "❌"]
        st.dataframe(miss_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Misses CSV",
            data=miss_df.to_csv(index=False).encode("utf-8"),
            file_name="speciesnet_misses.csv",
            mime="text/csv",
        )
    with tab2:
        st.dataframe(preds_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download All Predictions CSV",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="speciesnet_predictions.csv",
            mime="text/csv",
        )

    # Optional: flag suspiciously high accuracy with small val set
    total_support = rep_df["Support"].dropna().sum()
    if result["val_acc"] > 0.98 and total_support and total_support < 300:
        st.warning(
            "Validation accuracy is extremely high for a small validation set. "
            "Consider stronger augmentations or k-fold cross-validation to verify generalization."
        )

# --- 4) Recent runs from DB (optional) ---
st.markdown("### Recent Runs")
with SessionLocal() as db:
    try:
        recent = db.execute(text("""
            SELECT model_run_id, model_name, model_version, tag,
                   epochs, lr, batch_size, num_classes, num_train, num_val,
                   top1_accuracy, top5_accuracy, model_path, started_at, finished_at
            FROM wildlife.model_run
            ORDER BY started_at DESC
            LIMIT 20
        """)).mappings().all()
        if recent:
            runs_df = pd.DataFrame(recent)
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No runs yet. Train a model to see it here.")
    except Exception:
        st.caption("Run history unavailable (tables may not exist yet).")
