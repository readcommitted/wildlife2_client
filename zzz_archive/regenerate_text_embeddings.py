"""
regenerate_text_embeddings.py — Semantic Text Embedding Regeneration
---------------------------------------------------------------------

Streamlit utility to regenerate 1536-dimensional OpenAI text embeddings for all images
using stored metadata (species label, location, behavior, tags).

These embeddings power the semantic search functionality within the Wildlife Vision System.

Features:
✅ Estimates token usage and approximate API cost
✅ Generates consistent text prompts from validated image metadata
✅ Stores new text_embedding vectors in the PostgreSQL database

Requirements:
- OpenAI API Key in `.streamlit/secrets.toml`
- pgvector extension in PostgreSQL for vector storage
- SQLAlchemy for database interaction

"""

import streamlit as st
from sqlalchemy import select
from db.db import SessionLocal
from db.image_model import ImageEmbedding, ImageHeader, ImageLabel
from tools.openai_utils import get_embedding

# --- Embedding Model Parameters ---
EMBEDDING_MODEL = "text-embedding-3-small"
TOKENS_PER_PROMPT_EST = 100
COST_PER_1K_TOKENS = 0.0001  # Adjust if using larger OpenAI models

# --- Estimate API Cost ---
with SessionLocal() as session:
    image_ids_to_update = session.execute(
        select(ImageEmbedding.image_id)
    ).scalars().all()

num_embeddings = len(image_ids_to_update)
estimated_tokens = num_embeddings * TOKENS_PER_PROMPT_EST
estimated_cost = (estimated_tokens / 1000) * COST_PER_1K_TOKENS

st.markdown(f"🔢 **Embeddings to regenerate:** `{num_embeddings}`")
st.markdown(f"🧠 **Model:** `{EMBEDDING_MODEL}` (via OpenAI API)")
st.markdown(f"💰 **Estimated cost:** `${estimated_cost:.4f}` for ~{estimated_tokens:,} tokens")


def build_text_embedding(image_id, session):
    """
    Constructs a structured prompt from image metadata and generates a text embedding.

    Args:
        image_id (int): Image record ID
        session (Session): Active SQLAlchemy session

    Updates:
        Stores the 1536-D text_embedding in the `image_embedding` table
    """
    image = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
    if not image:
        return

    species_label = session.query(ImageLabel.label_value).filter_by(
        image_id=image_id, label_type="user"
    ).scalar()

    location = image.location_description or ""
    behavior = image.behavior_notes or ""
    tags = ", ".join(image.tags or [])

    prompt = f"{species_label}. {location}. {behavior}. Tags: {tags}".strip()
    embedding = get_embedding(prompt)

    session.query(ImageEmbedding).filter_by(image_id=image_id).update({
        "text_embedding": embedding
    })


# --- Trigger Regeneration ---
if st.button("🔁 Regenerate All Text Embeddings (OpenAI)"):
    st.info("🔄 Starting regeneration...")

    updated = 0
    failed = []

    with SessionLocal() as session:
        for image_id in image_ids_to_update:
            try:
                build_text_embedding(image_id, session)
                image_name = session.query(ImageHeader.image_name).filter_by(image_id=image_id).scalar()
                st.markdown(f"✅ Regenerated text embedding for image: `{image_name}`")
                updated += 1
            except Exception as e:
                failed.append((image_id, str(e)))

        session.commit()

    st.success(f"🎉 Regenerated {updated} text embeddings using `{EMBEDDING_MODEL}`.")
    if failed:
        st.warning(f"⚠️ {len(failed)} failures encountered:")
        for image_id, error in failed[:10]:
            st.text(f"{image_id}: {error}")
