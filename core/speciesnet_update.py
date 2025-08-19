# speciesnet_update.py
# --------------------
# Update DB from in-memory SpeciesNet results:
# - Inserts a row into wildlife.image_model_result
# - Updates wildlife.image_header (species_id, species_confidence, species_detection_method)
#
# Usage:
#   from speciesnet_update import apply_speciesnet_result
#   preds = run_speciesnet_one(jpg_local, country=..., admin1_region=...)
#   apply_speciesnet_result(image_id, preds)


# core/speciesnet_update.py
from __future__ import annotations
from typing import Any, Optional, Tuple, Dict
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from db.species_model import SpeciesFlattened


def _taxon_to_common_name(taxon: Optional[str]) -> Optional[str]:
    if not taxon or not isinstance(taxon, str):
        return None
    parts = [p.strip() for p in str(taxon).split(";") if p and p.strip()]
    return parts[-1] if parts else None

def _normalize(preds: Any) -> Tuple[Optional[str], Optional[float], Dict]:
    rec = preds[0] if isinstance(preds, list) and preds else (preds if isinstance(preds, dict) else {})
    if not rec:
        return None, None, {}
    pred = rec.get("prediction")
    score = rec.get("prediction_score")
    if not pred:
        cls = rec.get("classifications") or {}
        classes = cls.get("classes") or []
        scores  = cls.get("scores")  or []
        if classes:
            pred  = classes[0]
            score = scores[0] if scores else None
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    return _taxon_to_common_name(pred), score, rec

def apply_speciesnet_result(session, image_id: int, preds: Any, *, model_name: str = "speciesnet") -> Dict:
    common, score, rec = _normalize(preds)
    model_version = (rec or {}).get("model_version")

    # resolve species_id
    species_id = -1
    if common:
        row = (session.query(SpeciesFlattened)
               .filter(SpeciesFlattened.common_name.ilike(f"%{common}%"))
               .first())
        if row:
            species_id = row.species_id

    # prepare statement with JSONB bindparam (no ::jsonb in SQL string!)
    stmt = text("""
        INSERT INTO wildlife.image_model_result
            (image_id, model, model_version, species_id, common_name, prediction_score, result)
        VALUES (:image_id, :model, :model_version, :species_id, :common_name, :score, :result)
        RETURNING image_model_id
    """).bindparams(bindparam("result", type_=JSONB))

    res = session.execute(
        stmt,
        {
            "image_id": image_id,
            "model": model_name,
            "model_version": model_version,
            "species_id": species_id,
            "common_name": common,
            "score": score,
            "result": rec,  # pass the dict directly; SQLAlchemy serializes to JSONB
        },
    )
    image_model_id = res.scalar_one()

    return {
        "image_id": image_id,
        "image_model_id": image_model_id,
        "common_name": common,
        "species_id": species_id,
        "score": score,
        "model_version": model_version,
    }

