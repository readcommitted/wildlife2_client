# train_speciesnet_ui.py — Streamlit UI for Training SpeciesNet (Train/Test wording)

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
        SELECT
            ih.image_id,
            ih.jpeg_path,
            ih.capture_date,       -- used for group-by-day split
            il.label_value AS common_name
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

# --- 2) Set Training Options ---
st.markdown("### 2. Set Training Options")

colA, colB, colC = st.columns(3)
with colA:
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
with colB:
    lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=1e-4, format="%.6f")
with colC:
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)

colS1, colS2, colS3 = st.columns(3)
with colS1:
    val_size = st.number_input(
        "Validation size (fraction)",
        min_value=0.05, max_value=0.90, value=0.30, step=0.05, format="%.2f",
        help="e.g., 0.30 = 70/30 split"
    )
with colS2:
    test_size = st.number_input(
        "Test size (fraction, optional)",
        min_value=0.00, max_value=0.90, value=0.00, step=0.05, format="%.2f",
        help="Set >0 for a separate *test* holdout (e.g., 0.20 → 60/20/20). "
             "If left at 0, the validation split will be displayed as Test."
    )
with colS3:
    use_groups = st.checkbox(
        "Group by capture day (prevent leakage)",
        value=True,
        help="Keeps all images from the same capture date (YYYY‑MM‑DD) in the same split"
    )

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
    if val_size + test_size >= 0.95:
        st.warning("Validation + Test must be < 0.95 so there’s enough data to train.")
        st.stop()

    with st.spinner("Training in progress..."):
        result = train_and_evaluate_speciesnet(
            df=df,  # includes capture_date
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            tag=model_tag.strip(),
            val_size=float(val_size),
            test_size=float(test_size) if test_size > 0 else None,
            use_capture_date_groups=bool(use_groups),
            debug_split_checks=True
        )

    # Header + IDs
    st.success(f"Training complete! Model saved to `{result['model_path']}`")
    model_run_id = result.get("model_run_id")
    if model_run_id:
        st.info(f"Run ID: {model_run_id}")

    # ---- Decide which split to show as "Test" ----
    has_true_test = ("test_acc" in result)
    test_acc = result.get("test_acc", result["val_acc"])
    test_top5 = result.get("test_top5_acc", result["top5_acc"])
    test_labels = result.get("test_labels", result["labels"])
    test_cm = np.array(result.get("test_confusion_matrix", result["confusion_matrix"]), dtype=np.int32)
    test_report = result.get("test_classification_report", result["classification_report"])

    # For counts, compute held-out sample count from the report's support
    heldout_count = 0
    try:
        # Sum supports across all classes (ignore macro/weighted rows)
        heldout_count = int(sum(v["support"] for k, v in test_report.items() if isinstance(v, dict) and "support" in v and k not in {"accuracy", "macro avg", "weighted avg", "micro avg"}))
    except Exception:
        pass

    # Topline metrics (TEST)
    st.markdown("### Test Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top‑1 Accuracy (Test)", f"{test_acc*100:.2f}%")
    c2.metric("Top‑5 Accuracy (Test)", f"{test_top5*100:.2f}%")
    c3.metric("Held‑out Samples", f"{heldout_count}" if heldout_count else "—")


    # Additional headline metrics (F1)
    try:
        macro_f1 = float(result.get("test_macro_f1", test_report.get("macro avg", {}).get("f1-score", 0.0)))
        weighted_f1 = float(result.get("test_weighted_f1", test_report.get("weighted avg", {}).get("f1-score", 0.0)))
        micro_f1 = float(result.get("test_micro_f1", test_acc))
    except Exception:
        macro_f1 = weighted_f1 = test_acc
        micro_f1 = test_acc

    c4, c5, c6 = st.columns(3)
    c4.metric("Macro F1 (Test)", f"{macro_f1*100:.2f}%")
    c5.metric("Weighted F1 (Test)", f"{weighted_f1*100:.2f}%")
    c6.metric("Micro F1 (Test)", f"{micro_f1*100:.2f}%")
    # Training curves
    hist = result.get("history", {})
    if hist:
        hist_df = pd.DataFrame(hist)
        st.markdown("### Training Curves")
        st.plotly_chart(px.line(hist_df, x="epoch", y=["train_loss", "val_loss"], title="Loss"),
                        use_container_width=True)
        st.plotly_chart(px.line(hist_df, x="epoch", y=["train_acc", "val_acc"], title="Accuracy"),
                        use_container_width=True)

    # Confusion matrix (TEST)
    st.markdown("### Confusion Matrix (Test)")
    cm_df = pd.DataFrame(test_cm, index=test_labels, columns=test_labels)
    fig_cm = px.imshow(
        cm_df, text_auto=True, aspect="equal",
        labels=dict(x="Predicted", y="True", color="Count"),
    )
    fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=700)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class table (TEST)
    st.markdown("### Per‑Class Metrics (Test)")
    rep_df = _per_class_rows(test_report, test_labels)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

    # Per‑sample inspection:
    # We only have detailed per-sample predictions for the validation split the trainer returns.
    # If we DO NOT have a true test split, those "predictions" correspond to the displayed Test (val-as-test).
    # If we DO have a true test, show the validation inspection separately.
    if not has_true_test:
        st.markdown("### Top‑5 Inspection (Test)")
        preds_df = result["predictions"].copy()
        if "top5" not in preds_df.columns:
            preds_df["top5"] = ""
        if "match" in preds_df.columns:
            preds_df["Correct"] = preds_df["match"].map(lambda x: "✅" if x else "❌")
        elif "Correct" not in preds_df.columns:
            preds_df["Correct"] = ""

        tab1, tab2 = st.tabs(["Misses (Top‑1 incorrect)", "All Test Samples"])
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
    else:
        with st.expander("Validation (dev) diagnostics"):
            # Headline (Validation)
            v1, v2 = st.columns(2)
            v1.metric("Top‑1 (Validation)", f"{result['val_acc']*100:.2f}%")
            v2.metric("Top‑5 (Validation)", f"{result['top5_acc']*100:.2f}%")

            # Confusion (Validation)
            v_labels = result["labels"]
            v_cm = np.array(result["confusion_matrix"], dtype=np.int32)
            v_cm_df = pd.DataFrame(v_cm, index=v_labels, columns=v_labels)
            fig_v = px.imshow(v_cm_df, text_auto=True, aspect="equal",
                              labels=dict(x="Predicted", y="True", color="Count"))
            fig_v.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
            st.plotly_chart(fig_v, use_container_width=True)

            # Per‑class (Validation)
            v_rep_df = _per_class_rows(result["classification_report"], v_labels)
            st.dataframe(v_rep_df, use_container_width=True, hide_index=True)

            # Per‑sample (Validation)
            st.markdown("Validation Top‑5 Inspection")
            preds_df = result["predictions"].copy()
            if "top5" not in preds_df.columns:
                preds_df["top5"] = ""
            if "match" in preds_df.columns:
                preds_df["Correct"] = preds_df["match"].map(lambda x: "✅" if x else "❌")
            elif "Correct" not in preds_df.columns:
                preds_df["Correct"] = ""
            st.dataframe(preds_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Validation Predictions CSV",
                data=preds_df.to_csv(index=False).encode("utf-8"),
                file_name="speciesnet_val_predictions.csv",
                mime="text/csv",
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
