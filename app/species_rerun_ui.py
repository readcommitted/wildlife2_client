import streamlit as st
from core.langgraph_species_agent import run_species_agent_pipeline
from db.db import SessionLocal
from sqlalchemy import text
import ast
import numpy as np
import json


from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

run_button = st.button("Re-run Missed Identifications")

session = SessionLocal()

# Previously accepted results with low probability
query = """
SELECT 
    e.image_id,
    e.image_name,
    e.latitude,
    e.longitude,
    i.image_embedding,
    i.common_name AS predicted,
    l.label_value AS actual,
    e.species_confidence
FROM wildlife.image_header e
JOIN wildlife.image_label l ON e.image_id = l.image_id
JOIN wildlife.image_embedding i ON e.image_id = i.image_id
WHERE l.label_type = 'user'
  AND i.common_name <> l.label_value
ORDER BY e.image_id
"""
rows = session.execute(text(query)).fetchall()

if run_button:
    for row in rows:
        # --- Prepare embedding ---
        embedding = row.image_embedding
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = ast.literal_eval(embedding)
            elif isinstance(embedding, bytes):
                embedding = json.loads(embedding.decode("utf-8"))
            embedding = np.array(embedding)
        else:
            embedding = None

        try:
            steps, result_state = run_species_agent_pipeline(
                image_id=row.image_id,
                lat=row.latitude,
                lon=row.longitude,
                embedding=embedding.tolist() if embedding is not None else None,
                top_n=5
            )

            # Extract best_match (LLM or rerank)
            best_match = (
                result_state.get("llm_decision", {}).get("best_match") or
                result_state.get("rerank", {}).get("best_match") or {}
            )

            common_name = best_match.get("common_name", "not available")
            tool_rationale = result_state.get("identify", {}).get("rationale", "No tool rationale")
            final_rationale = result_state.get("rerank", {}).get("rationale", "No final rationale")
            llm_rationale = result_state.get("llm_decision", {}).get("llm_rationale", "No LLM rationale")

            # Output summary block only
            st.write(f"Image ID: {row.image_id} | Name: {row.image_name} | "
                     f"Label: {row.actual} | Previous Prediction: {row.predicted}")
            st.write(f"New Prediction: {common_name}")
            st.write(f"Tool Rationale (identify): {tool_rationale}")
            st.write(f"Final Rationale: {final_rationale}")
            st.write(f"LLM Rationale: {llm_rationale}")
            st.markdown("---")

        except Exception as e:
            st.error(f"⚠️ Error during re-evaluation of image {row.image_id}: {e}")
