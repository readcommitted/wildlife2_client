"""
species_watcher.py — LangGraph-Based Species ID Processor
-----------------------------------------------------------

This script continuously monitors for newly ingested images with missing
species identification and runs the LangGraph species agent pipeline on
each.

It integrates image embeddings, color profiles, and geolocation to make
a final prediction, then logs the decision trace to the `image_log` table.

Features:
* Detects unprocessed images with valid embeddings
* Runs the LangGraph agent pipeline (`run_species_agent_pipeline`)
* Slims and stores LangGraph reasoning steps and final state in `image_log`
* Includes image color profiles as part of the species decision context
* Logs failures with diagnostic payloads

Dependencies:
- NumPy
- SQLAlchemy
- LangGraph species pipeline
"""

# -------------------------------------------------------------------------
# Imports and database models
# -------------------------------------------------------------------------

from datetime import datetime
import time
from db.db import SessionLocal
from db.image_model import ImageHeader, ImageEmbedding, ImageLog, ImageFeature
from core.langgraph_species_agent import run_species_agent_pipeline
import ast
import numpy as np
import json
from config.settings import WATCHER_WAIT, WATCHER_IMAGES
from tools.species_lookup import smart_species_match
from sqlalchemy.dialects.postgresql import JSONB, insert



# -------------------------------------------------------------------------
# JSON + embedding serialization helpers
# -------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _strip_large_fields(obj, keys=("embedding", "image_embedding", "text_embedding", "embedding_vector")):
    """
    Recursively removes large or noisy fields from logs before serialization.
    """
    if isinstance(obj, dict):
        return {k: _strip_large_fields(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_large_fields(v, keys) for v in obj]
    return obj


def _to_candidate(row: dict) -> dict:
    """
    Formats a candidate row with all similarity scores for logging.
    """
    return {
        "common_name": row.get("common_name"),
        "prob": float(row.get("prob", 0.0)),
        "img": float(row.get("img", 0.0)),
        "text": float(row.get("text", 0.0)),
        "color": float(row.get("color", 0.0)),
        "combined": float(row.get("combined", 0.0)),
    }

# -------------------------------------------------------------------------
# Utility functions for slimming LangGraph output for storage
# -------------------------------------------------------------------------

def _round_num(x, ndigits=3):
    if isinstance(x, (float, int)):
        return round(float(x), ndigits)
    if isinstance(x, (np.floating, np.integer)):
        return round(x.item(), ndigits)
    return x

def _truncate(s: str | None, max_len=800):
    if not s or not isinstance(s, str):
        return s
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def _slim_best_match(b: dict, include_sims=False):
    """
    Extracts key fields from the best match result block.
    """
    if not isinstance(b, dict):
        return None
    get = b.get
    combined_val = get("combined_score", get("combined"))
    prob_val = get("probability", get("prob", combined_val))
    out = {
        "common_name": get("common_name"),
        "combined": _round_num(combined_val),
        "prob": _round_num(prob_val),
    }
    if include_sims:
        out["img"]   = _round_num(get("image_similarity", get("img")))
        out["text"]  = _round_num(get("text_similarity", get("text")))
        out["color"] = _round_num(get("color_similarity", get("color")))
    return {k: v for k, v in out.items() if v is not None}


def _slim_candidates(cands: list, top=5, include_sims=False):
    """
    Returns a list of top N slimmed candidate species.
    """
    out = []
    for c in (cands or [])[:top]:
        out.append(_slim_best_match(c, include_sims=include_sims))
    return [c for c in out if c]


def _attach_llm_rationale(out: dict, block: dict, max_len=600):
    """
    Extracts and attaches rationale string to a slimmed output dict.
    """
    rat = None
    if isinstance(block, dict):
        rat = block.get("llm_rationale")
        if not rat and isinstance(block.get("arb_result"), dict):
            rat = block["arb_result"].get("rationale")
    if rat:
        out["llm_rationale"] = _truncate(rat, max_len)
    return out


def _slim_step_block(name: str, block: dict):
    """
    Dispatches per-step logic for LangGraph step block slimming.
    """
    if not isinstance(block, dict):
        return {}

    # --- Step-specific logic ---
    if name == "identify":
        out = {
            "best_match": _slim_best_match(block.get("best_match"), include_sims=True),
            "top_candidates": _slim_candidates(block.get("top_candidates"), include_sims=True),
        }
        return _attach_llm_rationale(out, block)

    if name == "llm_decision":
        return {
            "decision": block.get("decision"),
            "llm_rationale": _truncate(block.get("llm_rationale")),
        }

    if name == "adjust_weights":
        return {
            "weights_used": block.get("weights_used") or {
                "image": _round_num(block.get("image_weight")),
                "text": _round_num(block.get("text_weight")),
                "color": _round_num(block.get("color_weight")),
            }
        }

    if name == "rerank":
        return {
            "weights_used": block.get("weights_used"),
            "best_match": _slim_best_match(block.get("best_match"), include_sims=True),
            "top_candidates": _slim_candidates(block.get("top_candidates"), include_sims=True),
        }

    if name == "commit_embedding":
        name_f = block.get("embed_commit_common_name") or (block.get("best_match") or {}).get("common_name")
        conf_f = block.get("embed_commit_confidence") or (block.get("best_match") or {}).get("probability")
        sid_f  = block.get("embed_commit_species_id") or (block.get("best_match") or {}).get("species_id")
        return {
            "common_name": name_f,
            "species_id": sid_f,
            "confidence": _round_num(conf_f),
        }

    if name == "fetch_speciesnet":
        sn = block.get("sn_best") or {}
        return {
            "species_id": sn.get("species_id"),
            "common_name": sn.get("common_name"),
            "sn_confidence": _round_num(sn.get("sn_confidence")),
            "model_version": sn.get("model_version"),
            "created_at": sn.get("created_at"),
        }

    if name == "compare_vs_speciesnet":
        sn = block.get("sn_best") or {}
        emb = block.get("best_match") or {}
        return {
            "consensus": bool(block.get("consensus")),
            "embedding": emb.get("common_name"),
            "speciesnet": sn.get("common_name"),
        }

    if name == "llm_arbitrate":
        arb = block.get("arb_result") or {}
        out = {k: arb.get(k) for k in ("winner", "confidence_final") if arb.get(k) is not None}
        rat = arb.get("rationale")
        if rat:
            out["llm_rationale"] = _truncate(rat, 600)
        return out

    if name == "update_final":
        return {
            "final_common_name": block.get("final_common_name"),
            "final_species_id": block.get("final_species_id"),
            "final_probability": _round_num(block.get("final_probability")),
            "final_method": block.get("final_method"),
        }

    return {}


def slim_steps_for_logging(steps: list) -> list:
    """
    Slims a LangGraph run stream to store only key step details.
    """
    keep = {
        "identify",
        "llm_decision",
        "adjust_weights",
        "rerank",
        "commit_embedding",
        "fetch_speciesnet",
        "compare_vs_speciesnet",
        "llm_arbitrate",
        "update_final",
    }
    slimmed = []
    for frame in steps or []:
        if not isinstance(frame, dict):
            continue
        for name, block in frame.items():
            if name in keep:
                slim = _slim_step_block(name, block)
                if any(v is not None and v != {} and v != [] for v in slim.values()):
                    slimmed.append({name: slim})
    return slimmed


def slim_final_state(result_state: dict) -> dict:
    """
    Extracts final state summary for logging.
    """
    if not isinstance(result_state, dict):
        return {}
    uf = result_state.get("update_final") or result_state
    out = {
        "final_common_name": uf.get("final_common_name"),
        "final_species_id": uf.get("final_species_id"),
        "final_probability": _round_num(uf.get("final_probability")),
        "final_method": uf.get("final_method"),
    }
    return {k: v for k, v in out.items() if v is not None}


# -----------------------------------------------------------------------------
# Main processing loop: runs LangGraph species ID on pending images
# -----------------------------------------------------------------------------

def process_pending_images(session):
    """
    Detects images that have no species match and processes them using LangGraph.

    Input tables:
    * image_header (source metadata)
    * image_embedding (precomputed embedding)
    * image_feature (color JSONB field, optional)

    Output table:
    * image_log (LangGraph reasoning trace)
    """
    result = (
        session.query(
            ImageHeader.image_id,
            ImageHeader.latitude,
            ImageHeader.longitude,
            ImageEmbedding.image_embedding,
            ImageFeature.colors
        )
        .join(ImageEmbedding, ImageHeader.image_id == ImageEmbedding.image_id)
        .outerjoin(ImageFeature, ImageHeader.image_id == ImageFeature.image_id)
        .filter(
            ImageHeader.metadata_updated.is_(True),
            ImageHeader.species_id == -1,
            ImageHeader.species_detection_method.is_(None),
            ImageEmbedding.image_embedding.isnot(None)
        )
        .order_by(ImageHeader.updated_at.asc())
        .limit(WATCHER_IMAGES)
        .all()
    )

    for row in result:
        image_id = row.image_id
        lat = row.latitude
        lon = row.longitude
        colors = row.colors or {}

        embedding = row.image_embedding
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = ast.literal_eval(embedding)
            elif isinstance(embedding, bytes):
                embedding = json.loads(embedding.decode("utf-8"))
            embedding = np.array(embedding)

        print(f"*** Processing image_id={image_id}")

        try:
            steps, result_state = run_species_agent_pipeline(
                image_id=image_id,
                lat=lat,
                lon=lon,
                embedding=embedding.tolist(),
                image_colors=colors,
                top_n=5
            )

            steps_slim = slim_steps_for_logging(_strip_large_fields(steps))
            final_state_slim = slim_final_state(_strip_large_fields(result_state))

            payload = json.loads(json.dumps(
                {"steps": steps_slim, "final_state": final_state_slim},
                default=_json_default
            ))

            log_entry = {
                "image_id": image_id,
                "log_type": "langgraph",
                "phase": "latest",
                "log_json": payload,
            }

            stmt = insert(ImageLog).values(**log_entry)
            stmt = stmt.on_conflict_do_update(
                index_elements=["image_id", "log_type"],
                set_={"log_json": stmt.excluded.log_json}
            )
            session.execute(stmt)
            session.commit()

            print(f"*** Logged LangGraph run for image_id={image_id} (phase=latest)")

        except Exception as e:
            print(f"❌ Error processing image_id={image_id}: {e}")
            try:
                error_payload = {"steps": _strip_large_fields(locals().get("steps", [])),
                                 "final_state": {"error": str(e)}}
                log_entry = {
                    "image_id": image_id,
                    "log_type": "langgraph",
                    "phase": "latest",
                    "log_json": json.loads(json.dumps(error_payload, default=_json_default)),
                }
                stmt = insert(ImageLog).values(**log_entry)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["image_id", "log_type", "phase"],
                    set_={"log_json": stmt.excluded.log_json}
                )
                session.execute(stmt)
                session.commit()
            except Exception:
                session.rollback()


# -----------------------------------------------------------------------------
# Main loop: poll for updates every 30 seconds
# -----------------------------------------------------------------------------

def main():
    print("Starting species watcher...")
    while True:
        with SessionLocal() as session:
            process_pending_images(session)
        time.sleep(WATCHER_WAIT)


if __name__ == "__main__":
    main()
