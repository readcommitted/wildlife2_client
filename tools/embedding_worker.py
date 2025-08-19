"""
embedding_worker.py — Generate Missing OpenCLIP Embeddings
-----------------------------------------------------------------

This script identifies species records in the `SpeciesEmbedding` table that
are missing image or text embeddings, and computes them in parallel using
OpenCLIP utilities.

It uses a thread pool to efficiently handle large-scale embedding backfills.

Features:
* Detects records with missing `image_embedding` or `text_embedding`
* Generates embeddings using `generate_openclip_image_embedding()` and `generate_openclip_text_embedding()`
* Builds natural language prompts from species metadata
* Uses multithreading (ThreadPoolExecutor) for performance
* Commits updates back to the database

Dependencies:
- OpenCLIP utils (via `embedding_utils`)
- SQLAlchemy for DB access
- Python standard library (concurrent.futures, os)
"""

import concurrent.futures
from embedding_utils import generate_openclip_image_embedding, generate_openclip_text_embedding
from db.db import SessionLocal
from db.species_model import SpeciesEmbedding
import os


def update_embeddings_worker(rec_id, image_path, text):
    """
    Worker function to update a single SpeciesEmbedding record.

    Args:
        rec_id (int): Record ID in `species_embedding`
        image_path (str): Local path to the image
        text (str): Text prompt to embed
    """
    session = SessionLocal()
    try:
        rec = session.get(SpeciesEmbedding, rec_id)
        if image_path and os.path.exists(image_path):
            rec.image_embedding = generate_openclip_image_embedding(image_path).tolist()
        if text and text.strip():
            rec.text_embedding = generate_openclip_text_embedding(text).tolist()
        session.commit()
    except Exception as e:
        print(f"Error processing {rec_id}: {e}")
    finally:
        session.close()


def bulk_update_embeddings(max_workers=4):
    """
    Scans for records missing embeddings and processes them in parallel.

    Steps:
    * Find records with null `image_embedding` or `text_embedding`
    * Generate natural language prompts using common name and description
    * Spawn parallel workers to compute and store embeddings

    Args:
        max_workers (int): Number of threads for concurrent processing
    """
    from db.db import SessionLocal
    from db.species_model import SpeciesEmbedding

    session = SessionLocal()
    try:
        records = session.query(SpeciesEmbedding).filter(
            (SpeciesEmbedding.image_embedding.is_(None)) | (SpeciesEmbedding.text_embedding.is_(None))
        ).all()
    except Exception as e:
        print(f"Error fetching records: {e}")
        session.close()
        return
    session.close()

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rec in records:
            # Create short text prompt using species name and image description
            desc = rec.image_description.split('.')[0] if rec.image_description else ''
            if desc:
                text = f"a photo of a {rec.common_name_clean}. {desc}"
            else:
                text = f"a photo of a {rec.common_name_clean}"

            tasks.append(executor.submit(
                update_embeddings_worker,
                rec.id,
                rec.image_path,
                text,
            ))

        # Log progress every 20 completions
        for i, f in enumerate(concurrent.futures.as_completed(tasks), 1):
            if i % 20 == 0:
                print(f"Processed {i}/{len(records)} records")

    print(f"Done! Updated embeddings for {len(records)} records.")


if __name__ == "__main__":
    bulk_update_embeddings(max_workers=4)
