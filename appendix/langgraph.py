import streamlit as st
from pathlib import Path


# Path to PNG (relative to repo root)
img_path = Path(__file__).parent.parent / "assets" / "langgraph.png"

if img_path.exists():
    st.image(str(img_path), use_container_width=True, caption="LangGraph Multi-Model Species Identification & Arbitration")
else:
    st.error(f"Architecture diagram not found at {img_path}")
