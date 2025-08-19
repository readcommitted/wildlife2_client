"""
analysis.py — Wildlife Species Embedding Analysis Tools
-------------------------------------------------------

This module provides visualization and evaluation tools for:

* Comparing CLIP-predicted species against user-provided labels
* Visualizing species embeddings with UMAP projections
* Analyzing SpeciesNet detection accuracy and confidence
* Exploring species similarity within ecological regions

Uses:
- CLIP (ViT-B/32) for image/text embeddings
- PostgreSQL (via SQLAlchemy) for image/label data
- Plotly and Matplotlib for visualizations

"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ.setdefault("NUMBA_NUM_THREADS", "4")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
_umap_lock = threading.Lock()
import umap
import ast
from matplotlib import pyplot as plt
from sqlalchemy import text
from db.db import SessionLocal
from db.image_model import ImageEmbedding, ImageLabel, ImageHeader, ImageExif, ImageFeature
from db.species_model import SpeciesFlattened
import json
from db.image_model import ImageLog



@st.cache_data(show_spinner=False)
def run_umap_cached(X: np.ndarray, seed: int | None, neighbors: int = 15):
    # Single entry at a time (Streamlit may run multiple sessions)
    with _umap_lock:
        reducer = umap.UMAP(
            n_neighbors=neighbors,
            n_components=2,
            metric="cosine",
            random_state=None if seed is None else seed,
            low_memory=True,
            n_jobs=None if seed is None else 1,  # seed -> single-thread; None -> let it parallelize
        )
        return reducer.fit_transform(X)


def similarity_to_confidence(similarity: float, min_val=0.15, max_val=0.45) -> float:
    """
    Map cosine similarity to a 0–100 confidence percentage.
    Assumes similarity is in range [min_val, max_val].
    """
    clipped = max(min(similarity, max_val), min_val)
    scaled = (clipped - min_val) / (max_val - min_val)
    return round(scaled * 100, 1)


def run_clip_label_vs_prediction_analysis():
    with SessionLocal() as session:
        rows = session.query(
            ImageHeader.image_id,
            ImageEmbedding.common_name,
            ImageEmbedding.score,
            ImageLabel.label_value,
            ImageExif.size_class,
            ImageFeature.colors,
            ImageLog.log_json
        ).join(
            ImageEmbedding, ImageHeader.image_id == ImageEmbedding.image_id
        ).join(
            ImageLabel, ImageHeader.image_id == ImageLabel.image_id
        ).join(
            ImageExif, ImageHeader.image_id == ImageExif.image_id
        ).join(
            ImageFeature, ImageHeader.image_id == ImageFeature.image_id
        ).outerjoin(  # logs might be missing
            ImageLog, ImageHeader.image_id == ImageLog.image_id
        ).filter(
            ImageLabel.label_type == 'user'
        ).all()


    results = []
    for image_id, predicted, score, true_label, size_class, colors, log_json in rows:
        agreement = (
            predicted and true_label and
            predicted.lower().strip() == true_label.lower().strip()
        )

        # Parse top candidates from rerank log
        top_candidates = []

        if log_json:
            try:
                log = log_json if isinstance(log_json, dict) else json.loads(log_json)

                # Try in order: rerank → rerank.top_candidates → llm_decision
                candidates = (
                        log.get("rerank", {}).get("top_candidates_reranked")
                        or log.get("rerank", {}).get("top_candidates")
                        or log.get("llm_decision", {}).get("top_candidates")
                        or []
                )

                for entry in candidates[:5]:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("common_name", "Unknown")
                    val = entry.get("combined_score", 0.0)
                    try:
                        score_str = f"{round(float(val), 4)}"
                    except (ValueError, TypeError):
                        score_str = "?"
                    top_candidates.append(f"{name} ({score_str})")

            except Exception:
                top_candidates.append("parse error")

        results.append({
            "image_id": image_id,
            "label": true_label,
            "predicted": predicted,
            "confidence score": f"{similarity_to_confidence(score, min_val=0.15, max_val=0.45)}%" if score is not None else None,
            "match": "✅" if agreement else "❌"
            #"size": size_class,
            #"colors": colors,
            #"top candidates (cosine similarity)": ", ".join(top_candidates)
        })

    df = pd.DataFrame(results).sort_values(by="image_id", ascending=False)

    st.dataframe(df[[
        "image_id",
        "label",
        "predicted",
        "confidence score",
        "match"
        #"size",
        #"colors",
        #"top candidates (cosine similarity)"
    ]])

    correct = sum(r["match"] == "✅" for r in results)
    total = len(results)
    st.markdown(f"### Accuracy: {correct} / {total} = **{correct / total:.2%}**")

    import plotly.express as px

    # --- Bar Chart: Correct vs Incorrect Predictions per Species ---
    chart_data = []
    for row in results:
        if not row["label"]:  # skip if label is missing
            continue
        chart_data.append({
            "Species": row["label"],
            "Result": "Correct" if row["match"] == "✅" else "Incorrect"
        })

    chart_df = pd.DataFrame(chart_data)

    if not chart_df.empty:
        chart_counts = chart_df.value_counts().reset_index(name="count")
        fig = px.bar(
            chart_counts,
            x="Species",
            y="count",
            color="Result",
            title="Prediction Outcomes by Species",
            barmode="stack",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def run_multimodel_analysis():
    with SessionLocal() as session:
        rows = session.query(
            ImageHeader.image_id,
            SpeciesFlattened.common_name,
            ImageHeader.species_confidence,
            ImageHeader.species_detection_method,
            ImageLabel.label_value,
            ImageExif.size_class,
            ImageFeature.colors,
            ImageLog.log_json
        ).join(
            ImageEmbedding, ImageHeader.image_id == ImageEmbedding.image_id
        ).join(
            ImageLabel, ImageHeader.image_id == ImageLabel.image_id
        ).join(
            ImageExif, ImageHeader.image_id == ImageExif.image_id
        ).join(
            ImageFeature, ImageHeader.image_id == ImageFeature.image_id
        ).join(
            SpeciesFlattened, ImageHeader.species_id == SpeciesFlattened.species_id
        ).outerjoin(  # logs might be missing
            ImageLog, ImageHeader.image_id == ImageLog.image_id
        ).filter(
            ImageLabel.label_type == 'user'
        ).all()


    results = []
    for image_id, predicted, species_confidence, species_detection_method, true_label, size_class, colors, log_json in rows:
        agreement = (
            predicted and true_label and
            predicted.lower().strip() == true_label.lower().strip()
        )


        results.append({
            "image_id": image_id,
            "label": true_label,
            "predicted": predicted,
            "confidence score": species_confidence,
            "match": "✅" if agreement else "❌",
            "species detection method" : species_detection_method
            #"size": size_class,
            #"colors": colors
        })

    df = pd.DataFrame(results).sort_values(by="image_id", ascending=False)

    st.dataframe(df[[
        "image_id",
        "label",
        "predicted",
        "confidence score",
        "match",
        "species detection method"
        #"size",
        #"colors"
    ]])

    correct = sum(r["match"] == "✅" for r in results)
    total = len(results)
    st.markdown(f"### Accuracy: {correct} / {total} = **{correct / total:.2%}**")

    import plotly.express as px

    # --- Bar Chart: Correct vs Incorrect Predictions per Species ---
    chart_data = []
    for row in results:
        if not row["label"]:  # skip if label is missing
            continue
        chart_data.append({
            "Species": row["label"],
            "Result": "Correct" if row["match"] == "✅" else "Incorrect"
        })

    chart_df = pd.DataFrame(chart_data)

    if not chart_df.empty:
        chart_counts = chart_df.value_counts().reset_index(name="count")
        fig = px.bar(
            chart_counts,
            x="Species",
            y="count",
            color="Result",
            title="Prediction Outcomes by Species",
            barmode="stack",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def display_umap_species_projection():
    """UMAP 2D projection of species and image embeddings with drift detection"""

    st.subheader("UMAP Species Embedding Visualization")

    with SessionLocal() as session:
        # --- Controls ---
        st.markdown("#### About this view")
        with st.expander("What is this?"):
            st.markdown("""
            Visualizes species embeddings from the CLIP model:

            - **Blue**: Canonical species vectors  
            - **Gold**: Canonical species used in image labels  
            - **Green**: Image predictions matching user label  
            - **Red**: Image predictions differing from user label  
            """)

        class_filter = st.selectbox("Filter by Class", ["mammals", "birds"])
        show_overlay = st.checkbox("Overlay image-level embeddings", value=True)
        min_images = st.slider("Minimum images per ecoregion", 5, 200, 20, step=5)
        max_regions = st.slider("Max ecoregions", 1, 12, 4, step=1)

        # --- 1) Choose ecoregions based on current image distribution ---
        # Requires a DB function public.get_ecoregion_by_coords(lat, lon) -> (eco_code text)
        # If you already store ecoregion_code on images, swap this CTE to read that column instead.
        regions = session.execute(text("""
            WITH imgs AS (
                SELECT ih.image_id, ih.latitude AS lat, ih.longitude AS lon
                FROM wildlife.image_header ih
                JOIN wildlife.image_embedding ie ON ie.image_id = ih.image_id
                WHERE ie.common_name IS NOT NULL
                  AND ih.latitude IS NOT NULL AND ih.longitude IS NOT NULL
            ),
            eco AS (
                SELECT g.eco_code AS ecoregion_code, COUNT(*) AS image_count
                FROM imgs i
                JOIN LATERAL public.get_ecoregion_by_coords(i.lat, i.lon) AS g ON TRUE
                GROUP BY g.eco_code
                HAVING COUNT(*) >= :min_images
            ),
            ranked AS (
                SELECT e.ecoregion_code, e.image_count, rf.ecoregion_name
                FROM eco e
                JOIN wildlife.regions_flattened rf ON rf.ecoregion_code = e.ecoregion_code
                ORDER BY e.image_count DESC
            )
            SELECT ecoregion_code, ecoregion_name, image_count
            FROM ranked
        """), {"min_images": min_images, "max_regions": max_regions}).fetchall()

        if not regions:
            st.warning("No ecoregions met the minimum image threshold. Lower the slider or ingest more images.")
            st.stop()

        ECOREGION_CODES = [r[0] for r in regions]
        ECOREGION_NAME = ", ".join([f"{r[1]} ({r[0]}, n={r[2]})" for r in regions])

        # --- Species canonical vectors in those regions ---
        species_data = session.execute(text("""
            SELECT DISTINCT e.common_name, e.category, e.image_embedding
            FROM wildlife.species_embedding e
            JOIN wildlife.species_by_region r ON lower(e.common_name) = lower(r.common_name)
            WHERE r.ecoregion_code = ANY(:regions)
              AND e.category = :cls
              AND e.image_embedding IS NOT NULL
        """), {"regions": ECOREGION_CODES, "cls": class_filter}).fetchall()

        if not species_data:
            st.warning(f"No species embeddings found for '{class_filter}' in {ECOREGION_NAME}")
            st.stop()

        species_names, species_vecs = [], []
        for name, _, emb in species_data:
            emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
            species_names.append(name)
            species_vecs.append(emb)
        species_vecs = np.array(species_vecs)

        # Species that appear in user labels (for the gold highlight)
        labeled_species = {
            r[0] for r in session.query(ImageLabel.label_value.distinct())
                               .filter(ImageLabel.label_type == 'user').all()
        }

        # Species that appear in image predictions (for the overlay filter)
        image_species = [
            r[0] for r in session.query(ImageEmbedding.common_name.distinct())
                                 .filter(ImageEmbedding.common_name.in_(species_names)).all()
        ]

        label_filter = st.selectbox("Filter Image Embeddings by Species", ["All"] + image_species)

        # Prepare vectors and labels (canonical layer)
        all_vecs = list(species_vecs)
        all_labels = ["canonical species vectors"] * len(species_vecs)
        all_names = species_names.copy()
        all_paths = [None] * len(species_vecs)
        all_ids = [None] * len(species_vecs)
        all_true = [None] * len(species_vecs)
        all_distances = [None] * len(species_vecs)

        # --- 2) Overlay image embeddings with TRUE distance (not 'score') ---
        if show_overlay:
            # Compute cosine distance in SQL using pgvector: (species_emb <-> image_emb) AS distance
            overlay_rows = session.execute(text("""
                SELECT
                    ie.image_id,
                    ie.image_embedding,
                    ie.common_name AS predicted,
                    ih.jpeg_path,
                    (se.image_embedding <-> ie.image_embedding) AS distance,
                    il.label_value AS true_label
                FROM wildlife.image_embedding ie
                JOIN wildlife.image_header ih ON ih.image_id = ie.image_id
                JOIN wildlife.image_label  il ON il.image_id = ie.image_id AND il.label_type = 'user'
                JOIN wildlife.species_embedding se ON lower(se.common_name) = lower(ie.common_name)
                WHERE ie.common_name = ANY(:species_list)
                /** Optional: restrict overlay images to selected ecoregions **/
                AND EXISTS (
                    SELECT 1
                    FROM LATERAL public.get_ecoregion_by_coords(ih.latitude, ih.longitude) g
                    WHERE g.eco_code = ANY(:regions)
                )
                /** Optional filter by a specific predicted species **/
                AND (:only_species IS NULL OR ie.common_name = :only_species)
            """), {
                "species_list": species_names,
                "regions": ECOREGION_CODES,
                "only_species": None if label_filter == "All" else label_filter
            }).fetchall()

            for img_id, emb, pred, path, distance, true_label in overlay_rows:
                emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
                all_vecs.append(emb)
                all_labels.append("match" if (true_label and pred == true_label) else "drift")
                all_names.append(pred)
                all_paths.append(path)
                all_ids.append(img_id)
                all_true.append(true_label)
                all_distances.append(float(distance) if distance is not None else None)

        vec_array = np.array(all_vecs, dtype=np.float32)
        umap_2d = run_umap_cached(vec_array, seed=42, neighbors=10)
        #reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=42)
        #umap_2d = reducer.fit_transform(np.array(all_vecs))


        df_plot = pd.DataFrame({
            "x": umap_2d[:, 0],
            "y": umap_2d[:, 1],
            "type": all_labels,
            "name": all_names,
            "path": all_paths,
            "image_id": all_ids,
            "label_species": all_true,
            "distance": all_distances,  # <-- distance, not score
        })

        canonical_df = df_plot[df_plot["type"] == "canonical species vectors"]
        gold_df = canonical_df[canonical_df["name"].isin(labeled_species)]
        blue_df = canonical_df[~canonical_df["name"].isin(labeled_species)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=blue_df["x"], y=blue_df["y"], mode="markers",
            name="canonical species", marker=dict(color="blue", size=10),
            hovertext=blue_df["name"]
        ))
        if not gold_df.empty:
            fig.add_trace(go.Scatter(
                x=gold_df["x"], y=gold_df["y"], mode="markers+text",
                name="labeled species (gold)",
                marker=dict(color="gold", size=12, line=dict(width=1, color="black")),
                text=gold_df["name"], textposition="top center"
            ))

        for label, color in [("match", "green"), ("drift", "red")]:
            df_sub = df_plot[df_plot["type"] == label]
            if not df_sub.empty:
                fig.add_trace(go.Scatter(
                    x=df_sub["x"], y=df_sub["y"], mode="markers",
                    name=label, marker=dict(color=color, size=10),
                    hovertext=df_sub["name"]
                ))

        fig.update_layout(height=700, margin=dict(l=40, r=40, t=40, b=40), hovermode="closest")
        st.markdown(f"**Ecoregions:** {ECOREGION_NAME}")
        st.plotly_chart(fig, use_container_width=True)

        drift_df = df_plot[df_plot["type"] == "drift"]
        if not drift_df.empty:
            st.markdown("### ❌ Drift Outliers")
            st.dataframe(
                drift_df[["image_id", "label_species", "name", "distance"]]
                    .rename(columns={"name": "predicted_species"}),
                use_container_width=True
            )
        else:
            st.markdown("No drift detected among image embeddings.")



def clip_region_comparison(limit=100):
    """Compare a species to others within its ecoregion using CLIP distance"""
    ECOREGION_CODE = "NA0528"
    ECOREGION_NAME = "South Central Rockies forests"

    with SessionLocal() as session:
        species_data = session.execute(text("""
            SELECT common_name, class_name
            FROM wildlife.species_by_region
            WHERE ecoregion_code = :code
            ORDER BY common_name
        """), {"code": ECOREGION_CODE}).fetchall()

    species_names = [s[0] for s in species_data]
    species_classes = {s[0]: s[1] for s in species_data}
    target_species = st.selectbox("Target Species", species_names)

    if st.button("Run Regional Comparison"):
        with SessionLocal() as session:
            vecs = session.query(ImageEmbedding.image_embedding) \
                .join(ImageHeader, ImageEmbedding.image_id == ImageHeader.image_id) \
                .join(ImageLabel, ImageHeader.image_id == ImageLabel.image_id) \
                .filter(ImageLabel.label_value == target_species,
                        ImageLabel.label_type == 'user').all()

            if not vecs:
                st.warning(f"No user-labeled embeddings found for '{target_species}'")
                st.stop()

            base_vec = np.mean([np.array(v) for (v,) in vecs], axis=0)

            species_embs = session.execute(text("""
                SELECT common_name, image_embedding
                FROM wildlife.species_embedding
                WHERE common_name = ANY(:species)
            """), {"species": species_names}).fetchall()

            distances = []
            for name, emb in species_embs:
                if name == target_species:
                    continue
                emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
                dist = 1 - np.dot(base_vec, emb) / (np.linalg.norm(base_vec) * np.linalg.norm(emb))
                distances.append((name, dist, species_classes.get(name, "Unknown")))

        if distances:
            df = pd.DataFrame(distances, columns=["Species", "Cosine Distance", "Class"]).sort_values("Cosine Distance")

            class_colors = {
                "Mammalia": "darkblue",  # brighter colors for dark background
                "Aves": "deepskyblue",
                "Reptilia": "orange",
                "Amphibia": "violet",
                "Insecta": "lightgray"
            }

            fig, ax = plt.subplots(figsize=(10, 5))

            # Dark theme adjustments
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            colors = df["Class"].map(class_colors).fillna("white")
            ax.scatter(df["Species"], df["Cosine Distance"], c=colors, alpha=0.85)

            ax.set_xticks([])  # Hide x-tick clutter
            ax.set_ylabel(f"Cosine Distance to '{target_species}'")
            ax.set_title(f"CLIP Similarity: '{target_species}' vs. Other Species in {ECOREGION_NAME}")

            st.pyplot(fig)
            st.markdown("### 🐾 Closest Species")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("No species embeddings found for this region.")

