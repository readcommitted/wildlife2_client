"""
langgraph_species_agent.py — Multi-Step Species Identification Pipeline
-----------------------------------------------------------------------

This module defines a multi-node LangGraph pipeline for identifying wildlife species
from image embeddings, location, and optional color features.

The pipeline uses:
* Embedding-based species identification via `identify_species_by_embedding`
* Optional reranking using LLM-guided weight adjustments
* SpeciesNet comparison + LLM arbitration
* Multi-modal final decision logic

Stages:
identify → (LLM decision) → rerank? → commit_embedding → fetch SpeciesNet →
compare → (LLM arbitration if needed) → update_final

Key Features:
* Smart weight adjustment based on LLM rationale
* Cosine similarity conversion to confidence %
* Structured comparison logic between models
* Pipeline logs include rationale, weights, and trace

Dependencies:
- LangGraph
- OpenAI SDK
- SQLAlchemy
- NumPy
- Requests
- Wildlife Vision API

Security Notes:
* Uses `OPENAI_API_KEY` from `config.settings`
* External HTTP POST to Wildlife API is bounded by timeout (30s)
* Embedded JSON is safely extracted via regex + balanced parser
* Prompted LLM responses include guardrails and formatting instructions

LangGraph
START
 └── identify (calls /identify-by-embedding)
       ↓
     llm_decision (accept vs rerank)
       ├── accept → commit_embedding
       └── rerank → adjust_weights → rerank → commit_embedding
                        ↓
               fetch_speciesnet
                        ↓
            compare_vs_speciesnet
                        ↓
       ┌───────────────┴───────────────┐
       │                               │
 consensus                          no consensus
   ↓                                     ↓
 update_final                    llm_arbitrate → update_final
   ↓
  END

"""


import re
from datetime import datetime
from sqlalchemy import text
import requests
from typing import List, Callable, Optional, Literal, Dict
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import json
import numpy as np
from openai import OpenAI
import streamlit as st
from sqlalchemy import text
from db.db import SessionLocal
from tools.species_lookup import smart_species_match
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
import textwrap
from config.settings import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- AgentState  ----------
class AgentState(BaseModel):
    image_id: int
    lat: float
    lon: float
    embedding: List[float]
    top_n: int = 5
    image_weight: float = 0.6
    text_weight: float = 0.4
    color_weight: float = 0.0
    color_rerank_attempted: bool = False
    image_colors: Optional[dict] = None
    top_candidates: Optional[List[dict]] = None
    best_match: Optional[dict] = None
    rationale: Optional[str] = None
    rerank_attempted: Optional[bool] = False
    decision: Optional[Literal["accept", "rerank", "arbitrate"]] = None
    llm_rationale: Optional[str] = None


# ---------- API tools ----------
BASE_URL = "https://api.wildlife.readcommitted.com"


def identify_species_by_embedding_tool(body: dict) -> dict:
    r = requests.post(f"{BASE_URL}/species/identify-by-embedding", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def rerank_with_weights_tool(body: dict) -> dict:
    r = requests.post(f"{BASE_URL}/species/rerank-with-weights", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- helpers ----------
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def _extract_weights_from_text(text: str) -> Dict[str, float]:
    if not text:
        return {}
    m = _JSON_BLOCK.search(text)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
        out = {}
        for k in ("image_weight", "text_weight", "color_weight"):
            if k in raw:
                out[k] = float(raw[k])
        return out
    except Exception:
        return {}


def fetch_latest_speciesnet(image_id: int) -> dict | None:
    with SessionLocal() as s:
        row = s.execute(text("""
            SELECT species_id, common_name, prediction_score, model_version, created_at
            FROM wildlife.image_model_result
            WHERE image_id = :image_id AND model = 'speciesnet'
            ORDER BY created_at DESC
            LIMIT 1
        """), {"image_id": image_id}).mappings().first()
    if not row:
        return None
    return {
        "species_id": row["species_id"],
        "common_name": row["common_name"],
        "sn_confidence": float(row["prediction_score"] or 0.0),
        "model_version": row["model_version"],
        "created_at": str(row["created_at"]),
    }



_NAME_NORM = re.compile(r"[^a-z0-9]+")

def _norm_name(x: str) -> str:
    return _NAME_NORM.sub(" ", x.lower()).strip()

def names_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    # simple, fast, and good enough when species_id isn’t present
    return _norm_name(a) == _norm_name(b)



# ---------- nodes ----------
def identify_fn(state: dict) -> dict:
    res = identify_species_by_embedding_tool({
        "image_id": state["image_id"],
        "embedding": state["embedding"],
        "lat": state["lat"],
        "lon": state["lon"],
        "top_n": state.get("top_n", 5),
        "image_weight": state.get("image_weight", 0.6),
        "text_weight": state.get("text_weight", 0.4),
        "color_weight": state.get("color_weight", 0.0),
        "image_colors": state.get("image_colors")
    })

    top_candidates = res.get("top_candidates", [])
    best = res.get("best_match", {})

    best.setdefault("color_similarity", 0.0)
    for c in top_candidates:
        c.setdefault("color_similarity", 0.0)

    # --- Inject SpeciesNet prediction if needed ---
    sn = fetch_latest_speciesnet(state["image_id"])
    state["speciesnet"] = sn or {}

    if sn:
        sn_name = sn.get("common_name")
        if sn_name and isinstance(sn_name, str):
            try:
                mapped = smart_species_match(sn_name)
                if mapped:
                    sn_common = mapped["common_name"]
                    already_present = any(c.get("common_name") == sn_common for c in top_candidates)
                    if not already_present:
                        sn_conf = float(sn.get("sn_confidence") or 0.0)
                        top_candidates.append({
                            "img": None,
                            "prob": sn_conf,
                            "text": None,
                            "color": 0.0,
                            "combined": sn_conf,
                            "common_name": sn_common,
                        })
                        print(f"[SpeciesNet Injected] {sn_common} @ {sn_conf:.2f}")
            except Exception as e:
                print(f"[SpeciesNet Injection Error] {sn_name=} — {e}")

    return {
        **state,
        "top_candidates": top_candidates,
        "best_match": best,
        "rationale": res.get("rationale", "")
    }


def decision_fn(state: dict) -> dict:
    cands = state.get("top_candidates") or []
    best = state.get("best_match") or {}
    tool_rat = state.get("rationale", "")

    # format candidates safely
    def fmt(c):
        return (
            f"- {c.get('common_name','?')}: "
            f"image={c.get('image_similarity',0.0):.3f}, "
            f"text={c.get('text_similarity',0.0):.3f}, "
            f"combined={c.get('combined_score',0.0):.3f}"
        )

    formatted_candidates = "\n".join(fmt(c) for c in cands)

    prompt = f"""
You are reviewing the output of a wildlife species identification model.
The model ranked candidates using weighted image/text/color similarity. Your job is to decide if the selected "best match" is reasonable, or if reranking is needed.

Rules:

The goal is to choose the most likely species based on combined score (weighted image, text, and color similarity).
Only choose "rerank" if:
The current best match is not the clear winner, and
Adjusting weights could realistically change the top species.
This includes any of:
If a candidate has no img or text similarity but includes a combined score, this means it came from the SpeciesNet model. Use this to guide confidence or reranking decisions.
A non-best candidate has text_similarity ≥ best.text_similarity + 0.05 (text advantage).
Best.combined_score < another candidate’s combined_score and best.text_similarity − next_best_text ≤ 0.02 (combined inconsistency).
Another candidate has color_similarity ≥ 0.60 and (candidate.color_similarity − next_best_color) ≥ 0.06 and combined_score difference ≤ 0.07 (color distinctiveness).
Do not rerank if:
The current best match already has the highest combined score by a meaningful margin (≥ 0.05 absolute difference from the next candidate), or
No modality shows a significant gap that could plausibly flip the top species.
If reranking is needed, explain why and propose new weights based on the strongest signal:
Baseline (balanced): {{"image_weight":0.45,"text_weight":0.45,"color_weight":0.10}}
Text advantage: {{"image_weight":0.30,"text_weight":0.70,"color_weight":0.00}}
Image advantage: {{"image_weight":0.60,"text_weight":0.40,"color_weight":0.00}}
Color distinctiveness: {{"image_weight":0.40,"text_weight":0.30,"color_weight":0.30}} (cap color ≤ 0.30)
Never let color dominate when best.image_similarity < 0.25 and best.text_similarity − next_best_text ≥ 0.05.
If reranking is not needed, return "decision": "accept" and explain that the current best match is already optimal.
First, reply with one word: "accept" or "rerank".
Then, on the next line, briefly explain your reasoning (≤ 25 words).
On the third line, output the JSON weights exactly.
Top Candidates:
{formatted_candidates}
Best Match: {best.get('common_name','?')}
Text similarity: {float(best.get('text_similarity') or 0):.3f}
Image similarity: {float(best.get('image_similarity') or 0):.3f}
Color similarity: {float(best.get('color_similarity') or 0):.3f}
Combined score: {float(best.get('combined_score') or 0):.3f}


Tool rationale:
{tool_rat}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = (resp.choices[0].message.content or "").strip()
    lines = raw.splitlines()
    decision_line = (lines[0] if lines else "").strip().lower()
    llm_rationale = " ".join(lines[1:]).strip()
    decision = "rerank" if "rerank" in decision_line else "accept"

    return {**state, "decision": decision, "llm_rationale": llm_rationale}



def adjust_weights(state: dict) -> dict:
    iw = state.get("image_weight", 0.6)
    tw = state.get("text_weight", 0.4)
    cw = state.get("color_weight", 0.0)

    new_w = _extract_weights_from_text(state.get("llm_rationale", ""))
    iw = float(new_w.get("image_weight", iw))
    tw = float(new_w.get("text_weight", tw))
    cw = float(new_w.get("color_weight", cw))

    s = iw + tw + cw
    if s > 0:
        iw, tw, cw = iw/s, tw/s, cw/s

    print(f"[adjust_weights] image={iw:.2f} text={tw:.2f} color={cw:.2f}")
    return {
        **state,
        "image_weight": iw,
        "text_weight": tw,
        "color_weight": cw,
        "weights_used": {"image": iw, "text": tw, "color": cw},
    }


def rerank_fn(state: dict) -> dict:
    print("[rerank] state keys:", list(state.keys()))
    cands_in = state.get("top_candidates") or []
    if not cands_in:
        msg = "No top_candidates present at rerank step"
        print("[rerank] ❌", msg)
        return {
            **state,
            "rerank_attempted": True,
            "rationale": msg,
            "weights_used": {
                "image": state.get("image_weight", 0.6),
                "text": state.get("text_weight", 0.4),
                "color": state.get("color_weight", 0.0),
            },
        }

    iw = state.get("image_weight", 0.6)
    tw = state.get("text_weight", 0.4)
    cw = state.get("color_weight", 0.0)
    print(f"[rerank] using weights image={iw:.2f} text={tw:.2f} color={cw:.2f}")

    # Call server-side rerank (now returns color_similarity & color dicts)
    res = rerank_with_weights_tool({
        "top_candidates": cands_in,
        "image_weight": iw,
        "text_weight": tw,
        "color_weight": cw,
    })
    cands = res.get("top_candidates", [])

    # Ensure defaults so recompute never breaks
    for c in cands:
        c["color_similarity"] = c.get("color_similarity", 0.0)
        c["image_colors"] = c.get("image_colors") or {}
        c["species_colors"] = c.get("species_colors") or {}

    # Recompute combined score with all weights
    scores = []
    for c in cands:
        col = c["color_similarity"]
        c["combined_score"] = (
            iw * c.get("image_similarity", 0.0)
            + tw * c.get("text_similarity", 0.0)
            + cw * col
        )
        scores.append(c["combined_score"])

    # Recompute probabilities via softmax
    if scores:
        m = max(scores)
        exp = [np.exp(s - m) for s in scores]
        total = float(np.sum(exp)) or 1.0
        for c, p in zip(cands, exp):
            c["probability"] = float(p / total)

    best = max(cands, key=lambda x: x.get("probability", 0.0), default=None)

    return {
        **state,
        "top_candidates": cands,
        "best_match": best,
        "rationale": f"Reranked using weights (image: {iw:.2f}, text: {tw:.2f}, color: {cw:.2f}).",
        "rerank_attempted": True,
        "weights_used": {"image": iw, "text": tw, "color": cw},
    }


def fetch_speciesnet_fn(state: dict) -> dict:
    """Pull latest SpeciesNet result from image_model_result for this image."""
    sn = fetch_latest_speciesnet(state["image_id"])
    print(f"[fetch_speciesnet_fn] {sn}")
    return {**state, "sn_best": sn}

def get_embed_best(state: dict) -> dict:
    bm = (
        state.get("llm_decision", {}).get("best_match")
        or state.get("rerank", {}).get("best_match")
        or state.get("identify", {}).get("best_match")
        or state.get("best_match")
        or {}
    )
    # backfill from finals if missing
    if not bm.get("common_name") and state.get("final_common_name"):
        bm["common_name"] = state["final_common_name"]
    if bm.get("species_id") is None and state.get("final_species_id") is not None:
        bm["species_id"] = state["final_species_id"]
    # normalize score keys for downstream
    bm["combined_score"] = (
        bm.get("combined_score")
        or bm.get("combined")
        or bm.get("probability")
        or 0.0
    )
    return bm


def compare_vs_speciesnet_fn(state: dict) -> dict:
    embed_best = get_embed_best(state)
    sn_best    = state.get("sn_best") or {}

    sid_equal = (
        embed_best.get("species_id") is not None and
        sn_best.get("species_id") is not None and
        int(embed_best["species_id"]) == int(sn_best["species_id"])
    )
    name_equal = names_match(embed_best.get("common_name"), sn_best.get("common_name"))
    same = bool(sid_equal or name_equal)

    arb_input = {
        "embedding": {
            "species_id": embed_best.get("species_id"),
            "common_name": embed_best.get("common_name"),
            "combined_score": float(embed_best.get("combined_score") or 0.0),
            "text_similarity": float(embed_best.get("text_similarity", 0.0)),
            "image_similarity": float(embed_best.get("image_similarity", 0.0)),
            "color_similarity": float(embed_best.get("color_similarity", 0.0)),
        },
        "speciesnet": sn_best,
        "location": {"lat": state["lat"], "lon": state["lon"]},
    }
    print(f"[compare_vs_speciesnet_fn] {arb_input}")
    return {**state, "consensus": same, "arb_input": arb_input}


def _extract_first_json_obj(text: str):
    """Extract the first top-level JSON object from arbitrary text."""
    # 1) Prefer fenced ```json blocks if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2) Fallback: scan for the first balanced {...} while respecting strings/escapes
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
    return None

def llm_arbitrate_fn(state: dict) -> dict:
    """If no consensus, ask LLM to choose embedding vs speciesnet, with a final confidence."""
    if state.get("consensus"):
        print("consensus reached")
        return state

    print("llm_arbitrate_fn running")
    p = state["arb_input"]
    loc = p.get("location") or {}
    lat = loc.get("lat")
    lon = loc.get("lon")
    emb = p.get("embedding") or {}
    sn  = p.get("speciesnet") or {}

    emb_view = {
        "common_name": emb.get("common_name"),
        "combined_score": float(emb.get("combined_score") or 0.0),
    }
    sn_view = {
        "common_name": sn.get("common_name"),
        "species_id": sn.get("species_id"),
        "sn_confidence": float(sn.get("sn_confidence") or 0.0),
    }

    emb_str = json.dumps(emb_view, ensure_ascii=False)
    sn_str  = json.dumps(sn_view,  ensure_ascii=False)

    header = (
        f"Arbitrate wildlife identification between two picks.\n\n"
        f"Location: lat={lat}, lon={lon}\n\n"
        f"Embedding pick:\n{emb_str}\n\n"
        f"SpeciesNet pick:\n{sn_str}\n\n"
    )

    rubric = textwrap.dedent("""
        Decide the final winner with this rubric:

        1) Specificity
           - Prefer species-level over family/genus/umbrella labels.
           - Treat SpeciesNet as LOW-SPECIFICITY if any is true:
             • species_id <= 0
             • common_name includes terms: "family", "genus", "unknown", "unidentified", "sp."
           - LOW-SPECIFICITY predictions can NEVER win unless rule (4) says otherwise.

        2) Confidence gates (hard floors)
           - SpeciesNet may win only if sn_confidence ≥ 0.90 AND NOT low-specificity.
           - Embedding may win only if combined_score ≥ 0.24.

        3) Strength comparison (when both pass gates)
           - strength_embedding = combined_score
           - strength_sn = sn_confidence
           - Switch to SpeciesNet only if (strength_sn - strength_embedding) ≥ 0.10; otherwise prefer Embedding unless rule (4).

        4) Rare override (very strong SN)
           - If sn_confidence ≥ 0.92 and embedding combined_score < 0.20, SpeciesNet may win even if rule (1) marks it low-specificity. Explain why.

        5) Geographic plausibility
           - Break near-ties by plausibility for the given lat/lon.

        Return STRICT JSON only (no extra text):
        {
          "winner": "embedding" | "speciesnet",
          "confidence_final": 0.0,
          "rationale": "short reason",
          "checks": {
            "sn_low_specificity": false,
            "sn_confidence": 0.0,
            "embedding_combined": 0.0,
            "margin_used": 0.0
          }
        }
    """).strip()

    prompt = header + rubric

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        # If your SDK supports it, this enforces pure JSON:
        # response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (resp.choices[0].message.content or "").strip()

    j = _extract_first_json_obj(raw)
    if not j:
        # Safe fallback to embedding
        j = {
            "winner": "embedding",
            "confidence_final": float(emb_view["combined_score"] or 0.5),
            "rationale": "Fallback due to parsing.",
            "checks": {
                "sn_low_specificity": False,
                "sn_confidence": sn_view["sn_confidence"],
                "embedding_combined": emb_view["combined_score"],
                "margin_used": sn_view["sn_confidence"] - emb_view["combined_score"],
            },
        }

    state["arb_prompt"] = prompt
    state["arb_result"] = j
    return state


def similarity_to_confidence(similarity: float, min_val=0.15, max_val=0.45) -> float:
    """
    Map cosine similarity to a 0–100 confidence percentage.
    Assumes similarity in [min_val, max_val].
    """
    clipped = max(min(similarity, max_val), min_val)
    scaled = (clipped - min_val) / (max_val - min_val)
    return round(scaled * 100.0, 1)

def to_conf_pct(score: float, *, normalized_max_is_one: bool = False,
                min_val=0.15, max_val=0.45) -> float:
    """
    Helper: if your score is already normalized to [0,1], just scale to 0–100.
    Otherwise use banded cosine → 0–100 mapping above.
    """
    if normalized_max_is_one:
        return round(max(0.0, min(1.0, float(score))) * 100.0, 1)
    return similarity_to_confidence(float(score), min_val=min_val, max_val=max_val)


# --- MID-PIPELINE COMMIT (embedding → image_embedding + image_header) ---
def commit_embedding_fn(state: dict) -> dict:
    """
    Persist the EMBEDDING pick (post-rerank) to:
      - wildlife.image_embedding (common_name, score)
      - wildlife.image_header   (species_id, species_confidence, species_detection_method='embedding')
    Then continue to SpeciesNet compare. Final commit may overwrite header later.
    """
    image_id = state["image_id"]

    # most recent winner in priority order
    best_match = (
        state.get("llm_decision", {}).get("best_match")
        or state.get("rerank", {}).get("best_match")
        or state.get("identify", {}).get("best_match")
        or state.get("best_match")
        or {}
    )

    common_name = best_match.get("common_name") or state.get("final_common_name")
    combined = (
        best_match.get("combined")
        or best_match.get("combined_score")
        or best_match.get("probability")
        or 0.0
    )

    print(f"[commit_embedding] image_id={image_id} "
          f"common_name={common_name!r} combined={combined:.3f} "
          f"has_top_best_match={'best_match' in state}")

    if not common_name:
        # Nothing to commit yet; continue pipeline
        state["embedding_committed"] = False
        state["embedding_committed_at"] = None
        return state

    with SessionLocal() as s:
        # Resolve species_id (None is OK; header will store NULL)
        try:
            sid = None
            match = smart_species_match(common_name, s)
            # Handle different possible return shapes gracefully
            if isinstance(match, dict):
                sid = match.get("species_id") or match.get("id")
            elif isinstance(match, (list, tuple)) and match:
                sid = match[0] if isinstance(match[0], int) else None
            elif isinstance(match, int):
                sid = match
        except Exception as e:
            print(f"[commit_embedding] smart_species_match error: {e}")
            sid = None

        print(f"[commit_embedding] resolved species_id={sid}")

        conf_pct = to_conf_pct(combined, normalized_max_is_one=False)

        # 1) Update image_embedding opinion (unchanged fields)
        s.execute(text("""
            UPDATE wildlife.image_embedding
               SET common_name = :common_name,
                   score       = :score
             WHERE image_id    = :image_id
        """), {"image_id": image_id, "common_name": common_name, "score": float(combined)})

        # 2) Interim header (now with %)
        s.execute(text("""
            UPDATE wildlife.image_header
               SET species_id               = COALESCE(:sid, species_id),
                   species_confidence       = :conf_pct,
                   species_detection_method = 'embedding',
                   updated_at               = now()
             WHERE image_id                 = :image_id
        """), {"image_id": image_id, "sid": sid, "conf_pct": conf_pct})

        s.commit()

    if isinstance(best_match, dict):
        best_match["species_id"] = sid
        state["best_match"] = best_match  # keep the top-level mirror in sync

    # also reflect finals
    state["final_method"] = "embedding"
    state["final_common_name"] = common_name
    state["final_probability"] = float(combined)
    state["final_species_id"] = sid
    state["embedding_committed"] = True

    return state


def _score_from_embed(embed: dict) -> float:
    return float(
        embed.get("combined_score")
        or embed.get("combined")
        or embed.get("probability")
        or 0.0
    )


def update_final_fn(state: dict) -> dict:
    """
    Persist the FINAL winner to wildlife.image_header and expose final_* in state.
    - If consensus: use shared pick; confidence = max(embedding, speciesnet) + small boost (<=1.0).
    - Else: use arb_result.winner to select from embedding or speciesnet.
    """
    image_id  = state["image_id"]
    embed     = state.get("best_match") or {}
    sn        = state.get("sn_best") or {}
    consensus = bool(state.get("consensus"))
    arb       = state.get("arb_result") or {}

    if consensus:
        emb_conf_raw = _score_from_embed(embed)                 # 0–1-ish (cosine-derived)
        sn_conf_raw  = float(sn.get("sn_confidence") or 0.0)    # 0–1
        use_sn       = sn_conf_raw >= emb_conf_raw
        chosen_name  = embed.get("common_name") or sn.get("common_name")
        chosen_id    = embed.get("species_id")  or sn.get("species_id")
        final_p      = min(1.0, max(emb_conf_raw, sn_conf_raw) + 0.05)
        method       = "consensus"
        # Map to % based on the source that won
        final_conf_pct = round(final_p * 100.0, 1) if use_sn else similarity_to_confidence(final_p)
    else:
        winner = (arb.get("winner") or "embedding").lower()
        if winner == "speciesnet" and sn:
            chosen_name   = sn.get("common_name")
            chosen_id     = sn.get("species_id")
            final_p       = float(arb.get("confidence_final") or sn.get("sn_confidence") or 0.8)
            method        = "speciesnet"
            final_conf_pct = round(max(0.0, min(1.0, final_p)) * 100.0, 1)  # SN is 0–1 → %
        else:
            chosen_name   = embed.get("common_name")
            chosen_id     = embed.get("species_id")
            final_p       = float(arb.get("confidence_final") or _score_from_embed(embed) or 0.8)
            method        = "embedding"
            final_p       = max(0.0, min(1.0, final_p))
            final_conf_pct = similarity_to_confidence(final_p)              # cosine band → %

    # Reflect finals into state BEFORE writing
    state["final_common_name"]    = chosen_name
    state["final_species_id"]     = chosen_id
    state["final_probability"]    = final_p            # keep raw 0–1 in-state if you need it
    state["final_confidence_pct"] = final_conf_pct     # human/UI + DB
    state["final_method"]         = method

    print(f"[final_write] image_id={image_id} name={chosen_name!r} sid={chosen_id} "
          f"raw={final_p:.3f} pct={final_conf_pct:.1f} method={method}")

    # Write % to DB; COALESCE avoids clobbering species_id with NULL
    with SessionLocal() as s:
        s.execute(text("""
            UPDATE wildlife.image_header
               SET species_id               = COALESCE(:sid, species_id),
                   species_confidence       = :conf_pct,
                   species_detection_method = :method,
                   updated_at               = now()
             WHERE image_id                 = :image_id
        """), {
            "image_id": image_id,
            "sid": chosen_id,
            "conf_pct": final_conf_pct,   # <-- use percent, not raw
            "method": method,
        })
        s.commit()

    return state


def run_species_agent_pipeline(
    image_id: int,
    lat: float,
    lon: float,
    embedding: List[float],
    image_colors: dict,
    top_n: int = 5,
):
    """
       identify → (maybe) rerank → COMMIT EMBEDDING → fetch SN → compare → (maybe) arbitrate → COMMIT FINAL → END
       Returns: (steps, final_state)
       """
    state_graph = StateGraph(dict)

    # existing nodes
    state_graph.add_node("identify", identify_fn)
    state_graph.add_node("llm_decision", decision_fn)  # accept vs rerank
    state_graph.add_node("adjust_weights", adjust_weights)
    state_graph.add_node("rerank", rerank_fn)

    # NEW mid-commit
    state_graph.add_node("commit_embedding", commit_embedding_fn)

    # arbitration phase
    state_graph.add_node("fetch_speciesnet", fetch_speciesnet_fn)
    state_graph.add_node("compare_vs_speciesnet", compare_vs_speciesnet_fn)
    state_graph.add_node("llm_arbitrate", llm_arbitrate_fn)
    state_graph.add_node("update_final", update_final_fn)

    # entry
    state_graph.set_entry_point("identify")

    # identify → decide
    state_graph.add_edge("identify", "llm_decision")

    # decision → either commit immediately or rerank then commit
    state_graph.add_conditional_edges(
        "llm_decision",
        lambda s: s.get("decision"),
        {"accept": "commit_embedding", "rerank": "adjust_weights"},
    )
    state_graph.add_edge("adjust_weights", "rerank")
    state_graph.add_edge("rerank", "commit_embedding")

    # after embedding commit, proceed to SN/compare/arbitrate/final
    state_graph.add_edge("commit_embedding", "fetch_speciesnet")
    state_graph.add_edge("fetch_speciesnet", "compare_vs_speciesnet")
    state_graph.add_conditional_edges(
        "compare_vs_speciesnet",
        lambda s: "accept" if s.get("consensus") else "arbitrate",
        {"accept": "update_final", "arbitrate": "llm_arbitrate"},
    )
    state_graph.add_edge("llm_arbitrate", "update_final")
    state_graph.add_edge("update_final", END)

    compiled = state_graph.compile()

    # Prepare the initial state with image/location data and default weights
    initial_state = {
        "image_id": image_id,
        "lat": lat,
        "lon": lon,
        "embedding": embedding,
        "image_colors": image_colors,
        "top_n": top_n,
        "image_weight": 0.6,
        "text_weight": 0.4,
        "color_weight": 0.0,
    }

    try:
        # Execute the pipeline, streaming intermediate states for trace logging
        steps = list(compiled.stream(initial_state))

        # Get the last state snapshot regardless of LangGraph's stream format
        last = steps[-1] if steps else {}
        final_state = (
            last.get("state", last) if isinstance(last, dict) else last
        )

        # Return both the full step trace and the final state
        return steps, final_state
    except Exception as e:
        # Log pipeline failure for this image
        print(f"❌ Error running LangGraph for image_id={image_id}: {e}")

        # Return empty steps and an error object for downstream compatibility
        return [], {"error": str(e)}

