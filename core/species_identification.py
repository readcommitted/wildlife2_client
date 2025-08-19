import datetime
from sqlalchemy import text
from db.db import SessionLocal
from db.image_model import ImageEmbedding, ImageHeader
from db.species_model import SpeciesEmbedding
import numpy as np

from zzz_archive.embedding_logger import build_embedding_log


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def species_identification(
    image_ids=None,
    only_current_batch=False,
    top_n=5,
    category='unknown',
    image_weight=0.6,
    text_weight=0.4
):
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
            lat, lon = header_row.latitude, header_row.longitude
            embedding = embedding_row.image_embedding

            if embedding is None or lat is None or lon is None:
                continue

            print(f"\n=== Processing image_id={image_id}, lat={lat}, lon={lon} ===")

            # --- 1. Get top-N candidates using DB function ---
            sql = text("""
                SELECT species, common_name, image_path, distance, eco_code
                FROM wildlife.usf_rank_species_candidates(
                    (:lat)::double precision,
                    (:lon)::double precision,
                    (:embedding)::vector,
                    :category,
                    :top_n
                )
            """)
            try:
                top_candidates = session.execute(sql, {
                    "lat": lat,
                    "lon": lon,
                    "embedding": embedding if isinstance(embedding, list) else embedding.tolist(),
                    "category": category,
                    "top_n": top_n
                }).fetchall()
            except Exception as e:
                print(f"❌ DB candidate selection failed for image_id {image_id}: {e}")
                continue

            if not top_candidates:
                print("No top candidates found.")
                continue

            print("Top candidates from DB:")
            for i, row in enumerate(top_candidates):
                print(f"  {i+1}. {row[1]} (species: {row[0]}) - Distance: {row[3]:.4f} - EcoRegion: {row[4]}")

            # --- 2. For each candidate, get its text embedding from species_embedding ---
            candidates = []
            for row in top_candidates:
                species = row[0]
                common_name = row[1]
                eco_code = row[4]
                db_row = session.query(SpeciesEmbedding).filter_by(species=species).first()
                if not db_row or db_row.text_embedding is None:
                    continue
                img_sim = 1 - row[3]  # Convert distance to similarity
                text_sim = cosine_similarity(embedding, db_row.text_embedding)  # This works if both are OpenCLIP 1024d
                combined_score = image_weight * img_sim + text_weight * text_sim
                candidates.append((common_name, combined_score, eco_code, img_sim, text_sim))

            if not candidates:
                print("  No candidates with valid text embedding found.")
                continue

            # --- 3. Compute probability scores with softmax ---
            scores = [c[1] for c in candidates]
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)

            # --- 4. Print candidates with scores and probabilities as percentages ---
            print("\nFinal Ranking with Probabilities (image+text CLIP similarity):")
            for i, (common_name, combined_score, eco_code, img_sim, text_sim) in enumerate(candidates):
                prob_pct = 100 * probs[i]
                print(f"  {common_name:25} | EcoRegion: {eco_code:8} | "
                      f"Img Sim: {img_sim:.3f} | Text Sim: {text_sim:.3f} | "
                      f"Combined: {combined_score:.4f} | Probability: {prob_pct:.1f}%")

            # --- 5. Select best match by probability ---
            best_idx = np.argmax(probs)
            best_common_name, best_score, best_eco_code, _, _ = candidates[best_idx]

            print(f"\n*** Best Match: {best_common_name} (score={best_score:.4f}, probability={100*probs[best_idx]:.1f}%) ***")

            # --- 6. Update the image embedding row ---
            embedding_row.common_name = best_common_name
            embedding_row.score = float(best_score)
            embedding_row.embedding_date = datetime.datetime.utcnow()
            embedding_row.embedding_method = "ViT-H-14 \\ laion2b_s32b_b79k"
            updated += 1

            # --- 7. Log output to postgres ---
            # Build top_candidates_raw from top_candidates
            top_candidates_raw = []
            for i, row in enumerate(top_candidates):
                top_candidates_raw.append({
                    "rank": i + 1,
                    "common_name": row[1],
                    "species": row[0],
                    "distance": float(row[3]),
                    "ecoregion": row[4]
                })

            # Build top_candidates_reranked from candidates
            top_candidates_reranked = []
            for i, (common_name, combined_score, eco_code, img_sim, text_sim) in enumerate(candidates):
                top_candidates_reranked.append({
                    "rank": i + 1,
                    "common_name": common_name,
                    "ecoregion": eco_code,
                    "img_similarity": float(img_sim),
                    "text_similarity": float(text_sim),
                    "combined_score": float(combined_score),
                    "probability": float(probs[i])
                })

            # Build best_match
            best_match = {
                "common_name": best_common_name,
                "ecoregion": best_eco_code,
                "combined_score": float(best_score),
                "probability": float(probs[best_idx])
            }

            # Now build and log
            log_entry = build_embedding_log(
                image_id=image_id,
                lat=lat,
                lon=lon,
                top_candidates_raw=top_candidates_raw,
                top_candidates_reranked=top_candidates_reranked,
                best_match=best_match
            )

        session.query(ImageHeader).filter_by(image_id=image_id).update({"embedding_log": log_entry})

        session.commit()
        print(f"\n✅ Updated {updated} image embeddings with species predictions (CLIP image+text re-ranking).\n")
