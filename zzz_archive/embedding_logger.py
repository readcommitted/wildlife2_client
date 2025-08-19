# embedding_logger.py

def build_embedding_log(image_id, lat, lon, top_candidates_raw, top_candidates_reranked, best_match):
    return {
        "image_id": image_id,
        "lat": lat,
        "lon": lon,
        "top_candidates_raw": top_candidates_raw,
        "top_candidates_reranked": top_candidates_reranked,
        "best_match": best_match
    }
