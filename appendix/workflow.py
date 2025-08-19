"""
workflow.py — System Workflow & Process Overview
-------------------------------------------------

This Streamlit module provides a detailed, interactive breakdown of the core
Wildlife Vision System workflow. It presents each major processing step
as an expandable section for easy understanding and onboarding.

Covers:
1. Species Embedding Preloading
2. Image Ingestion Pipeline
3. Intelligent Species Identification
4. Manual & Automated Annotation
5. Similarity Search via Vector Embeddings
6. Analytical Reporting & Insights

This visual workflow serves as:
- A technical reference for system architecture
- A training tool for new users or contributors
- A roadmap for understanding how data flows through the system

Dependencies:
- Streamlit for interactive UI

"""

import streamlit as st

# --- Species Embedding Preloading ---
with st.expander("1️⃣ Preload Species Embeddings (Offline)"):
    st.markdown("""
- **Trigger:** Manual or scheduled (ad-hoc).
- **Process:**
  1. Scrape species names (mammals, birds, reptiles) from Wikipedia.
  2. Download representative images for each species.
  3. Generate image embeddings using the local CLIP model.
  4. Store species names, image paths, and embeddings in the `species_embeddings` table.
- **Purpose:** Pre-generated reference vectors enable immediate recognition even for unseen species.
    """)

# --- Image Ingestion ---
with st.expander("2️⃣ Image Ingestion (Real-time or Batch)"):
    st.markdown("""
- **Trigger:** New images uploaded to the raw folder.
- **Process:**
  1. Extract metadata (EXIF) and capture creation date.
  2. Organize images in data lake directories (RAW, JPG, Staged).
  3. Convert RAW to JPG for consistency.
  4. Generate image embeddings using the local CLIP model.
  5. Store metadata and embeddings in the `images` table.
- **Purpose:** Standardize and enrich image data for downstream tasks.
    """)

# --- Species Identification ---
with st.expander("3️⃣ Species Identification (Dynamic, Intelligent)"):
    st.markdown("""
- **Trigger:** Immediately after ingestion.
- **Process:**
  1. Generate the image embedding with CLIP.
  2. Perform a semantic search against `species_embeddings` via pgvector.
     - If high-confidence match, assign that species.
     - If no high match, fall back to CLIP zero-shot classification.
  3. Update the `images` table with the identified species.
- **Purpose:** Combine fast vector lookup with intelligent fallback classification.
    """)

# --- Annotation and Tagging ---
with st.expander("4️⃣ Annotation and Tagging (Manual & Automated)"):
    st.markdown("""
- **Trigger:** After species identification or on demand.
- **Process:**
  1. Allow users to add/edit location, behavior, and custom tags.
  2. Suggest annotations by semantic similarity of existing embeddings.
  3. Store final annotations in the database.
- **Purpose:** Enhance data quality and collect labeled examples for model retraining.
    """)

# --- Similarity Search ---
with st.expander("5️⃣ Similarity Search (Advanced Analysis)"):
    st.markdown("""
- **Trigger:** User-initiated from Tools or Analysis section.
- **Process:**
  1. Generate embedding for the input image.
  2. Search the `images` table for top-N similar embeddings via pgvector.
  3. Display results with species, metadata, and paths.
- **Purpose:** Enable reverse image search and visual discovery.
    """)

# --- Analysis and Reporting ---
with st.expander("6️⃣ Analysis and Reporting (Insights)"):
    st.markdown("""
- **Trigger:** User-initiated from Tools or Dashboard.
- **Process:**
  1. Visualize species distribution and frequency.
  2. Analyze model confidence and identify gaps.
  3. Explore trends by season, location, or behavior.
- **Purpose:** Provide actionable insights and guide continuous improvement.
    """)

# --- Final Note ---
st.write("---")
st.info("This workflow is modular and scalable: expand `species_embeddings` for new regions/taxa, ingest any new images, and leverage semantic search for fast retrieval.")
