"""
run_vector_search.py — pgvector Semantic Search Utility
-------------------------------------------------------

Performs a vector similarity search using PostgreSQL `pgvector` for semantic matching of image text embeddings.

Features:
* Computes cosine distance between stored embeddings and the provided query vector
* Supports filtering results by species label (case-insensitive)
* Returns matching images sorted by similarity (closest first)
* Filters out records with missing latitude or invalid location data

This powers the natural language semantic search experience within the Wildlife Vision System.

Requirements:
- PostgreSQL with `pgvector` extension installed
- SQLAlchemy session setup (`SessionLocal`)
- Text embeddings stored in `wildlife.image_embedding`

"""

from sqlalchemy import text
from db.db import SessionLocal


def run_vector_search(
    query_vector: list[float],
    limit: int = 50,
    max_distance: float = 1.1,
    label_filter: str = None
):
    """
    Executes a cosine similarity search using pgvector to find images semantically matching the query.

    Args:
        query_vector (list[float]): 512-dimensional embedding to compare.
        limit (int, optional): Maximum number of results to return. Defaults to 50.
        max_distance (float, optional): Distance threshold for filtering weak matches. Defaults to 1.1.
        label_filter (str, optional): Species label to filter results (ILIKE match). Defaults to None.

    Returns:
        list[Row]: Result rows with fields:
            - image_id
            - jpeg_path
            - location_description
            - latitude
            - longitude
            - behavior_notes
            - tags
            - label (user-provided species label)
            - distance (cosine distance to query vector)
    """
    # Format vector as SQL-compatible string for pgvector comparison
    vector_str = f"[{', '.join(f'{x:.6f}' for x in query_vector)}]"

    # Core SQL for similarity search
    sql = f"""
        SELECT i.image_id, i.jpeg_path, i.location_description, i.latitude, i.longitude,
               i.behavior_notes, lbl.label_value AS label,
               e.text_embedding <-> '{vector_str}'::vector AS distance
        FROM wildlife.image_embedding e
        JOIN wildlife.image_header i ON e.image_id = i.image_id
        JOIN wildlife.image_label lbl ON lbl.image_id = i.image_id AND lbl.label_type = 'user'
        WHERE i.latitude IS NOT NULL AND i.latitude != 0
          AND (e.text_embedding <-> '{vector_str}'::vector) < :max_distance
    """

    # Optional label filtering (case-insensitive)
    if label_filter:
        sql += " AND lbl.label_value ILIKE :label_filter"

    sql += " ORDER BY distance ASC LIMIT :limit"

    # Query parameters
    params = {
        "limit": limit,
        "max_distance": max_distance
    }
    if label_filter:
        params["label_filter"] = f"%{label_filter.strip()}%"

    # Execute and return results
    with SessionLocal() as session:
        results = session.execute(text(sql), params).fetchall()

    return results
