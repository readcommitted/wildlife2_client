"""
semantic_search_ui.py — AI-Powered Semantic Wildlife Search Interface
-----------------------------------------------------------------------

This Streamlit module provides a natural language search tool for exploring wildlife
image data using semantic similarity powered by OpenAI embeddings.

Features:
- Accepts flexible, conversational prompts (e.g., "bears foraging near river")
- Optionally filters results by known species label
- Performs vector similarity search against precomputed text embeddings
- Displays top species and location summaries
- Provides AI-generated insights based on sightings, behaviors
- Interactive map of search results using Folium
- Option to preview sample images related to the query

Designed to enhance discovery of relevant wildlife images, behavioral patterns,
and ecological context from large datasets.

Dependencies:
- Streamlit for UI
- Folium for map visualization
- OpenAI for embedding generation and AI summaries
- PostgreSQL via SQLAlchemy
- Pandas for result handling

"""
import streamlit as st
from streamlit_folium import st_folium
from db.species_model import SpeciesFlattened
from tools.openai_utils import get_embedding
from db.db import SessionLocal
import pandas as pd
import folium
from openai import OpenAI
from tools.run_vector_search import run_vector_search
from tools.openai_utils import summarize_wildlife_search
from core.exception import custom_exception_hook
import sys


sys.excepthook = custom_exception_hook

from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

try:
    # --- OpenAI Client Setup ---
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # --- Session State Defaults ---
    for key, default in {
        "search_submitted": False,
        "last_prompt": "",
        "show_map": False,
        "show_images": False,
        "summary_prompt": None,
        "summary_response": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Smart Species Filter (Dropdown from Known Species) ---
    with SessionLocal() as session:
        species_options = session.query(SpeciesFlattened.common_name).order_by(SpeciesFlattened.common_name).all()
    species_list = [s[0] for s in species_options if s[0]]

    species_filter = st.selectbox(
        "Species Label (smart search)",
        options=[""] + species_list,
        index=0,
        help="Optionally filter results by a known species label"
    )
    label_filter = species_filter if species_filter else None

    # --- Natural Language Search Prompt ---
    user_prompt = st.text_input("Semantic Search Prompt", placeholder="e.g. near river, in winter, close-up")

    # --- Sidebar Search Options ---
    with st.sidebar:
        st.header("Search Options")
        result_limit = st.slider("Max Results", min_value=10, max_value=200, value=50, step=10)
        similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.5, 0.8, 0.05)

        st.markdown("🛠️ Developer Settings")
        dev_mode = st.checkbox("Enable Developer Mode", value=False)

        if st.button("🔄 Reset Search"):
            st.session_state.search_submitted = False
            st.session_state.last_prompt = ""
            st.session_state.summary_prompt = None
            st.session_state.summary_response = None

    # --- Submit Search ---
    if st.button("Submit Search"):
        st.session_state.search_submitted = True
        combined_prompt = f"{label_filter} {user_prompt.strip()}" if label_filter else user_prompt.strip()
        st.session_state.last_prompt = combined_prompt
        st.session_state.summary_prompt = None
        st.session_state.summary_response = None

    # --- Run Semantic Search if Submitted ---
    if st.session_state.search_submitted and st.session_state.last_prompt:

        with st.spinner("🔎 Searching..."):
            query_vector = get_embedding(st.session_state.last_prompt)
            results = run_vector_search(
                query_vector,
                limit=result_limit,
                max_distance=similarity_threshold,
                label_filter=label_filter
            )

        # Developer Debug View
        if dev_mode:
            st.markdown("### 🛠️ Developer Mode")
            st.code(st.session_state.last_prompt, language="markdown")
            if results:
                for row in results[:5]:
                    label = row.get("label", "Unknown") if isinstance(row, dict) else row[6]
                    score = row.get("distance", 0.0) if isinstance(row, dict) else row[8]
                    st.write(f"- {label} (distance: {score:.4f})")

        if not results:
            st.warning("No matching results found.")
        else:
            # Format results into DataFrame
            df = pd.DataFrame(results, columns=[
                "image_id", "jpeg_path", "location", "lat", "lon",
                "behavior", "label", "distance"
            ])

            st.markdown(f"### Found {len(df)} matching entries for: *{st.session_state.last_prompt}*")

            # --- Quick Species and Location Summary ---
            top_species = df['label'].value_counts().head(3).index.tolist()
            top_locations = df['location'].value_counts().head(3).index.tolist()

            st.markdown(f"**Top Species:** {', '.join(top_species)}")
            st.markdown(f"**Top Locations:** {', '.join(top_locations)}")

            # --- AI-Generated Summary Insight ---
            st.markdown("#### 🧠 Summary Insight")

            if st.session_state.summary_response is None:
                sample_notes = df['behavior'].dropna().unique().tolist()

                notes_summary = "; ".join(sample_notes[:5])

                summary_prompt = (
                    f"The user searched for '{st.session_state.last_prompt}'"
                    f"{' and filtered by label: ' + label_filter if label_filter else ''}. "
                    f"Based on sightings of {', '.join(top_species)} in {', '.join(top_locations)}, "
                    f"with common behaviors such as: {notes_summary} "
                    "summarize seasonal patterns, movement behavior, or habitat preferences for these sightings."
                )

                st.session_state.summary_prompt = summary_prompt
                st.session_state.summary_response = summarize_wildlife_search(summary_prompt)

            st.info(st.session_state.summary_response)

            # --- Toggle Map Button ---
            if st.button("📍 Toggle Map"):
                st.session_state.show_map = not st.session_state.show_map

            # --- Display Map ---
            if st.session_state.show_map:
                st.markdown("#### 📍 Result Map")
                m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=9)
                for _, row in df.iterrows():
                    popup = f"{row['label']}<br>{row['location']}<br>{row['behavior'] or ''}"
                    folium.Marker([row['lat'], row['lon']], popup=popup).add_to(m)
                st_folium(m, width=800, height=300)
                st.markdown("---")

            # --- Toggle Images Button ---
            if st.button("Toggle Images"):
                st.session_state.show_images = not st.session_state.show_images

            from tools.spaces import generate_signed_url

            # --- Image Gallery Preview ---
            if st.session_state.show_images:
                st.markdown("Images")
                cols = st.columns(5)

                for idx, row in enumerate(df.head(10).itertuples()):
                    with cols[idx % 5]:
                        try:
                            # Use Spaces-signed URL for the image path
                            signed_url = generate_signed_url(row.jpeg_path)
                            st.image(signed_url, caption=f"{row.label or 'Unknown'}", use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Failed to load image: {row.jpeg_path}")


except Exception as e:
    # Manually invoke the custom exception handler
    short_message = custom_exception_hook(type(e), e, e.__traceback__)
    st.error(f"{short_message}: An error occurred, please contact the developer.")
    st.stop()
