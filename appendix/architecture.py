import streamlit as st
from pathlib import Path


st.title("Wildlife Vision 2 — Architecture")

# Path to PNG (relative to repo root)
img_path = Path(__file__).parent.parent / "assets" / "architecture.png"

if img_path.exists():
    st.image(str(img_path), use_column_width=True, caption="System architecture")
else:
    st.error(f"Architecture diagram not found at {img_path}")
