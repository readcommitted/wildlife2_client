"""
regenerate_image_embeddings.py — Embedding Generation & CLIP Re-Ranking
------------------------------------------------------------------------

This module provides utilities to regenerate and enhance image embeddings within the Wildlife Vision System.

Key Features:
* Regenerates 512-D CLIP image embeddings
* Computes similarity scores using PostgreSQL vector functions
* Re-ranks species candidates with CLIP text-image comparison
* Generates image_text_embeddings for improved search and validation
* Supports batch updates for both wildlife images and species reference data

Requirements:
- PyTorch and CLIP for embeddings and similarity comparison
- pgvector extension in PostgreSQL
- SQLAlchemy for database access
- Streamlit for UI feedback

"""

import streamlit as st
import open_clip
from sqlalchemy import select, update, text
from db.db import SessionLocal
from db.image_model import ImageEmbedding, ImageHeader
from db.species_model import SpeciesEmbedding
from tools.embedding_utils import generate_openclip_image_embedding
import torch
import torch.nn.functional as F
import datetime


model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# --- You want ONE global model/tokenizer per process ---
_model, _, _preprocess = None, None, None
_tokenizer = None


def get_openclip_model():
    global _model, _preprocess, _tokenizer
    if _model is None or _preprocess is None or _tokenizer is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        _tokenizer = open_clip.get_tokenizer(model_name)
        _model = _model.to(device)
        _model.eval()
    return _model, _preprocess, _tokenizer


def regenerate_image_embeddings_and_score(image_ids: list[int] = None, only_current_batch: bool = False):
    """
    Regenerates image embeddings and computes updated similarity scores for wildlife images.

    Args:
        image_ids (list[int], optional): Specific image IDs to update.
        only_current_batch (bool, optional): Restrict to images flagged as current_batch.

    Updates:
        - image_embedding (512-D vector)
        - common_name (best species match)
        - score (similarity score)
        - image_text_embedding (512-D text embedding for search)
    """
    model, preprocess, _ = get_openclip_model()
    updated = 0
    with SessionLocal() as session:
        query = (
            session.query(ImageEmbedding, ImageHeader)
            .join(ImageHeader, ImageEmbedding.image_id == ImageHeader.image_id)
        )
        if image_ids:
            query = query.filter(ImageEmbedding.image_id.in_(image_ids))
        elif only_current_batch:
            query = query.filter(ImageHeader.current_batch.is_(True))
        else:
            query = query.filter(ImageHeader.species_id == -1)

        rows = query.all()

        for embedding_row, header_row in rows:
            image_id = embedding_row.image_id
            image_path = header_row.jpeg_path
            lat, lon = header_row.latitude, header_row.longitude

            try:
                embedding = generate_openclip_image_embedding(image_path)
                embedding_row.image_embedding = embedding

                if embedding is None or lat is None or lon is None:
                    continue

                sql = text("""
                    SELECT species, common_name, image_path, distance, location_boosted, final_score
                    FROM wildlife.usf_rank_species_candidates(
                        (:lat)::double precision,
                        (:lon)::double precision,
                        (:embedding)::vector,
                        :category,
                        :top_n
                    )
                """)

                top5 = session.execute(sql, {
                    "lat": lat, "lon": lon,
                    "embedding": embedding.tolist(),
                    "category": "unknown",
                    "top_n": 5
                }).fetchall()

                if not top5:
                    continue

                top5_common_names = [row[1] for row in top5]
                prompts = [f"a photo of a {name.lower()}" for name in top5_common_names]

                with torch.no_grad():
                    text_tokens = open_clip.tokenize(prompts).to(device)
                    text_embs = F.normalize(model.encode_text(text_tokens), dim=-1)
                    image_tensor = F.normalize(torch.tensor(embedding).to(device).unsqueeze(0), dim=-1)
                    sims = (image_tensor @ text_embs.T).squeeze(0).cpu().numpy()

                re_ranked = [
                    (row[1], 0.6 * (1 - row[3]) + 0.4 * sims[i]) for i, row in enumerate(top5)
                ]
                re_ranked.sort(key=lambda x: x[1], reverse=True)
                best_match, score = re_ranked[0]

                text_prompt = f"a photo of a {best_match.lower()}"
                with torch.no_grad():
                    text_vec = F.normalize(model.encode_text(open_clip.tokenize([text_prompt]).to(device)), dim=-1).squeeze(0).cpu().numpy().tolist()

                embedding_row.common_name = best_match
                embedding_row.score = float(score)
                ##embedding_row.text_embedding = text_vec
                embedding_row.embedding_date = datetime.datetime.utcnow()

                updated += 1

            except Exception as e:
                st.warning(f"❌ Failed to update image_id {image_id}: {e}")

        session.commit()
        st.success(f"* Updated {updated} image embeddings with scores and text embeddings.")


def regenerate_species_embeddings():
    """
    Regenerates image embeddings for all species in `species_embedding` table.

    Updates:
        - image_embedding field based on stored image_path
    """
    updated = 0
    with SessionLocal() as session:
        rows = session.execute(select(SpeciesEmbedding.id, SpeciesEmbedding.image_path)).fetchall()

        for id_value, image_path in rows:
            try:
                embedding = generate_openclip_image_embedding(image_path)
                session.execute(
                    update(SpeciesEmbedding)
                    .where(SpeciesEmbedding.id == id_value)
                    .values({"image_embedding": embedding})
                )
                updated += 1
            except Exception as e:
                st.warning(f"❌ Failed to process species {id_value}: {e}")

        session.commit()
        st.success(f"* Regenerated {updated} species embeddings.")


def update_image_embedding_common_names(session, image_records):
    """
    Internal helper to update missing common_name, score, and image_text_embedding fields.

    Args:
        session: SQLAlchemy session object
        image_records: List of ImageEmbedding rows to update
    """
    now = datetime.datetime.utcnow()

    for embedding_row in image_records:
        image_id = embedding_row.image_id
        embedding = embedding_row.image_embedding

        header = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
        if not header:
            continue

        lat, lon = header.latitude, header.longitude
        if embedding is None or len(embedding) == 0 or lat is None or lon is None:
            continue

        sql = text("""
            SELECT species, common_name, image_path, distance, location_boosted, final_score
            FROM wildlife.usf_rank_species_candidates(
                (:lat)::double precision,
                (:lon)::double precision,
                (:embedding)::vector,
                :category,
                :top_n
            )
        """)
        top5 = session.execute(sql, {
            "lat": lat, "lon": lon,
            "embedding": embedding.tolist(),
            "category": "unknown",
            "top_n": 5
        }).fetchall()

        if not top5:
            continue

        common_names = [row[1] for row in top5]
        prompts = [f"a photo of a {name.lower()}" for name in common_names]

        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(device)
            text_embs = F.normalize(model.encode_text(text_tokens), dim=-1)
            image_tensor = F.normalize(torch.tensor(embedding).to(device).unsqueeze(0), dim=-1)
            sims = (image_tensor @ text_embs.T).squeeze(0).cpu().numpy()

        re_ranked = [
            (row[1], 0.6 * (1 - row[3]) + 0.4 * sims[i]) for i, row in enumerate(top5)
        ]
        re_ranked.sort(key=lambda x: x[1], reverse=True)
        best_common_name, best_score = re_ranked[0]

        with torch.no_grad():
            text_vec = F.normalize(model.encode_text(open_clip.tokenize([f"a photo of a {best_common_name.lower()}"]).to(device)), dim=-1).squeeze(0).cpu().numpy().tolist()

        embedding_row.common_name = best_common_name
        embedding_row.score = float(best_score)
        #embedding_row.image_text_embedding = text_vec
        embedding_row.embedding_date = now

    session.commit()


def update_missing_common_names():
    """
    Searches for image embeddings with missing common_name fields and updates them.

    Updates:
        - common_name
        - score
        - image_text_embedding
    """
    with SessionLocal() as session:
        rows = session.query(ImageEmbedding).join(ImageHeader).filter(
            (ImageEmbedding.common_name == None) | (ImageEmbedding.common_name == '')
        ).all()

        if not rows:
            st.info("No missing common names to update.")
            return

        update_image_embedding_common_names(session, rows)
        st.success(f"Updated {len(rows)} common_name, score, and image_text_embedding fields.")
