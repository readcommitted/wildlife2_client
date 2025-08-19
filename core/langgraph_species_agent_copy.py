
import requests
from typing import List, Callable, Optional, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import json
import numpy as np
from openai import OpenAI
import streamlit as st
from db.db import SessionLocal
from sqlalchemy import text


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------------
# 🧠 Agent State
# ------------------------
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
    top_candidates: Optional[List[dict]] = None
    best_match: Optional[dict] = None
    rationale: Optional[str] = None
    rerank_attempted: Optional[bool] = False
    decision: Optional[Literal["accept", "rerank"]] = None
    llm_rationale: Optional[str] = None

# ------------------------
# 🔧 Tool Functions (Direct Requests)
# ------------------------
BASE_URL = "https://api.wildlife.readcommitted.com"


def identify_species_by_embedding_tool(body: dict) -> dict:
    response = requests.post(f"{BASE_URL}/species/identify-by-embedding", json=body)
    response.raise_for_status()
    return response.json()


def rerank_with_weights_tool(body: dict) -> dict:
    response = requests.post(f"{BASE_URL}/species/rerank-with-weights", json=body)
    response.raise_for_status()
    return response.json()


# ------------------------
# 🤖 LLM Reasoning
# ------------------------
def decision_fn(state: AgentState) -> AgentState:
    formatted_candidates = "\n".join(
        f"- {c['common_name']}: image={c['image_similarity']:.3f}, text={c['text_similarity']:.3f}, combined={c['combined_score']:.3f}"
        for c in state.top_candidates
    )

    prompt = f"""
You are reviewing the output of a wildlife species identification model.

The model ranked candidates using weighted image/text/color similarity. Your job is to decide if the selected "best match" is reasonable, or if reranking is needed.

You should suggest "rerank" if:
- A non-best candidate has significantly higher text similarity (≥ 0.05 higher)
- The current best match has mediocre text similarity and a lower combined score than another candidate
- Another candidate has significantly better color similarity and similar overall scores
If color appears to be a significant factor in visual similarity, you may adjust the image_weight, text_weight, and color_weight to influence the final decision.
Respond with a JSON including image_weight, text_weight, and optionally color_weight (e.g., 0.4, 0.3, 0.3)
First, reply with one word: "accept" or "rerank".

Then, on the following line, briefly explain your reasoning (1-2 sentences).

Top Candidates:
{formatted_candidates}

Best Match: {state.best_match['common_name']}  
Text similarity: {state.best_match['text_similarity']:.3f}  
Image similarity: {state.best_match['image_similarity']:.3f}  
Color similarity: {state.best_match.get('color_similarity', 0.0):.3f}  
Combined score: {state.best_match['combined_score']:.3f}  

Tool rationale:  
{state.rationale}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response.choices[0].message.content.strip()
    lines = raw_response.splitlines()
    decision_line = lines[0].strip().lower()
    rationale = " ".join(lines[1:]).strip()
    decision = "rerank" if "rerank" in decision_line else "accept"

    return state.copy(update={
        "decision": decision,
        "llm_rationale": rationale
    })


# ------------------------
# Graph Nodes
# ------------------------
def identify_fn(state: AgentState) -> AgentState:
    import requests
    import time


    for attempt in range(2):
        try:
            body = {
                "image_id": state.image_id,
                "embedding": list(map(float, state.embedding)),
                "lat": state.lat,
                "lon": state.lon,
                "top_n": state.top_n,
                "image_weight": state.image_weight,
                "text_weight": state.text_weight,
                "color_weight": 0.0
            }
            response = requests.post("https://api.wildlife.readcommitted.com/species/identify-by-embedding", json=body)
            response.raise_for_status()
            result = response.json()
            return state.copy(update=result)
        except requests.RequestException as e:
            print(f"❌ API error on attempt {attempt + 1}: {e}")
            if attempt == 0:
                time.sleep(1.5)  # short backoff
            else:
                raise


def check_color_rerank_needed(state: AgentState) -> Literal["accept", "rerank_color"]:
    if state.color_rerank_attempted:
        return "accept"

    best = state.best_match
    best_color_sim = best.get("color_similarity", 0.0)

    for candidate in state.top_candidates:
        if candidate["common_name"] == best["common_name"]:
            continue
        other_color_sim = candidate.get("color_similarity", 0.0)
        if other_color_sim - best_color_sim >= 0.25 and abs(candidate["combined_score"] - best["combined_score"]) < 0.05:
            return "rerank_color"

    return "accept"


def color_rerank_fn(state: AgentState) -> AgentState:
    body = {
        "top_candidates": state.top_candidates,
        "image_weight": 0.3,
        "text_weight": 0.3,
        "color_weight": 0.4
    }

    result = rerank_with_weights_tool(body)
    return state.copy(update=result, color_rerank_attempted=True)


def adjust_weights(state: AgentState) -> AgentState:
    return state.copy(update={"image_weight": 0.3, "text_weight": 0.7, "rerank_attempted": True})


def rerank_fn(state: AgentState) -> AgentState:
    body = {
        "top_candidates": state.top_candidates,
        "image_weight": state.image_weight,
        "text_weight": state.text_weight,
        "color_weight": state.color_weight if state.color_weight is not None else 0.0,
    }
    result = rerank_with_weights_tool(body)
    return state.copy(update=result)   # ✅ MUST return a state/dict



# ------------------------
#  Main Runner
# ------------------------
def run_species_agent_pipeline(
    image_id: int,
    lat: float,
    lon: float,
    embedding: List[float],
    top_n: int = 5
) -> tuple:

    # --- Define LangGraph ---
    state_graph = StateGraph(AgentState)

    # --- Nodes ---
    state_graph.add_node("identify", identify_fn)
    state_graph.add_node("llm_decision", decision_fn)
    state_graph.add_node("adjust_weights", adjust_weights)
    state_graph.add_node("rerank", rerank_fn)
    state_graph.add_node("color_rerank", color_rerank_fn)

    # --- Entry Point ---
    state_graph.set_entry_point("identify")

    # --- Main Flow ---
    state_graph.add_edge("identify", "llm_decision")

    state_graph.add_conditional_edges("llm_decision", lambda s: s.decision, {
        "accept": END,
        "rerank": "adjust_weights"
    })

    state_graph.add_edge("adjust_weights", "rerank")

    # ✅ ADD this instead
    state_graph.add_conditional_edges(
        "rerank",
        check_color_rerank_needed,  # returns "accept" | "rerank_color"
        {
            "accept": END,
            "rerank_color": "color_rerank",
        },
    )
    state_graph.add_edge("color_rerank", END)

    # --- Compile ---
    compiled_graph = state_graph.compile()

    try:
        steps = list(compiled_graph.stream(
            AgentState(
                image_id=image_id,
                lat=lat,
                lon=lon,
                embedding=embedding,
                top_n=top_n
            )
        ))

        final_state = steps[-1]
        return steps, final_state

    except Exception as e:
        print(f"❌ Error running LangGraph for image_id={image_id}: {e}")
        return {"error": str(e)}