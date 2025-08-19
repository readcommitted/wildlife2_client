"""
about.py — Project Overview & Visual Showcase
---------------------------------------------

This Streamlit module introduces the Wildlife Image Processing & Semantic Search System.
It provides a project overview and displays representative wildlife examples along with
predicted classification details.

Features:
- Presents a project summary highlighting AI-enhanced image processing and semantic search
- Displays example images with species predictions and confidence scores
- Uses Streamlit columns for responsive layout

Displayed examples:
- Brown Bear with confidence 0.738
- American Bison with confidence 0.9952
- Owl with confidence 0.322

Dependencies:
- Streamlit for UI rendering
- time for controlled display timing

"""

import time
import streamlit as st
from tools.spaces import generate_signed_url


@st.cache_data(ttl=3600)
def get_cached_signed_url(remote_path: str) -> str:
    return generate_signed_url(remote_path)


# --- Page Title and Project Description ---
st.title("Wildlife Image Processing & Semantic Search System")
st.write("")
st.write("This project presents the development of a modular, AI-enhanced system for processing, classifying, "
         "and retrieving wildlife images and videos. It integrates traditional computer vision techniques with "
         "advanced semantic understanding powered by Large Language Models. The platform supports manual and "
         "AI-assisted annotation, stores visual metadata and embeddings, and enables intuitive natural language "
         "queries to discover relevant visual content. By enabling contextual insights and advanced search "
         "capabilities, the system transforms how wildlife media can be explored and utilized.")
st.write("")

# --- Layout for Displaying Example Images ---
col1, col2, col3 = st.columns([1, 1, 1])

# Optional delay for smoother load experience
time.sleep(3)

# --- Example 1: Brown Bear ---
with col1:
    st.image(get_cached_signed_url("images/610-2.jpg"))
    st.markdown("""
            <style>
            img {
                pointer-events: none;
                user-select: none;
            }
            </style>
        <div style='text-align: center;'>
            <strong style='font-size: 2.0em;'>Brown Bear</strong><br>
            <strong>Confidence: 0.738</strong><br>
            <em>mammalia, carnivora, ursidae, ursus, arctos, brown bear</em>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Example 2: American Bison ---
with col2:
    st.image(get_cached_signed_url("images/bison_river.jpg"))
    st.markdown("""
            <style>
            img {
                pointer-events: none;
                user-select: none;
            }
            </style>
        <div style='text-align: center;'>
            <strong style='font-size: 2.0em;'>American Bison</strong><br>
            <strong>Confidence: 0.9952</strong><br>
            <em>mammalia, cetartiodactyla, bovidae, bison, bison, None, american bison</em>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Example 3: Owl ---
with col3:
    st.image(get_cached_signed_url("images/owl.jpg"))
    st.markdown("""
        <style>
        img {
            pointer-events: none;
            user-select: none;
        }
        </style>
        <div style='text-align: center;'>
            <strong style='font-size: 2.0em;'>Great Horned Owl</strong><br>
            <strong>Confidence: 0.322</strong><br>
            <em>aves, strigiformes, great horned owl</em>
        </div>
        """,
        unsafe_allow_html=True
    )



