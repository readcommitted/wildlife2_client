"""
adhoc_scraper.py — Ad-Hoc Wikipedia Species Scraper & Embedder
---------------------------------------------------------------

This Streamlit UI allows users to scrape species names and images from Wikipedia,
generate CLIP image embeddings, and insert records into the species_embedding table.

Supports:
- Scraping from individual species pages or Wikipedia list pages
- Fetching representative species images via the Wikipedia API
- Generating CLIP embeddings using the project's ingest pipeline
- Automatically updating species metadata via stored procedures

Intended for:
- Rapid addition of new species to the Wildlife Vision System
- Filling missing embeddings for species already present

Dependencies:
- Streamlit UI
- SQLAlchemy ORM
- Wikipedia API (read-only)
- Project image embedding tools

"""

import streamlit as st
from db.db import SessionLocal
from db.species_model import SpeciesEmbedding, SpeciesFlattened
from tools.species_embeddings import (
    fetch_species_image,
    scrape_species,
    extract_species_name_from_species_page,
)
import requests
from sqlalchemy import text
from tools.embedding_utils import generate_openclip_image_embedding, generate_openclip_text_embedding
from config.settings import WIKI_API_URL
import os


def scrape_species_description(title):
    """Fetches the lead/summary section of a Wikipedia page by title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title
    }
    url = "https://en.wikipedia.org/w/api.php"
    response = requests.get(url, params=params)
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")

def get_canonical_species_dict():
    session = SessionLocal()
    records = session.query(SpeciesFlattened).order_by(SpeciesFlattened.common_name).all()
    session.close()
    species_dict = {}
    for rec in records:
        display = f"{rec.common_name} ({rec.scientific_name})"
        species_dict[display] = rec
    return species_dict

# --- Streamlit UI ---
url = st.text_input("Wikipedia species or list URL", placeholder="https://en.wikipedia.org/wiki/Great_grey_owl")
category = st.selectbox("Category", ["birds", "mammals"])
user_description = st.text_area("Optional: Provide a custom description to use if Wikipedia description is missing")

# --- Canonical species smart search ---
species_dict = get_canonical_species_dict()
species_display = list(species_dict.keys())

# Only show smart search for individual page
show_smart_search = url and not ("list_of_" in url.lower() or "list" in url.lower())
selected_display = None
if show_smart_search:
    selected_display = st.selectbox("Link to canonical species (optional, improves accuracy):", ["[None]"] + species_display)

if st.button("Scrape and Process"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        session = SessionLocal()
        species_set = set()

        # 1. Parse URL for names
        if "list_of_" in url.lower() or "list" in url.lower():
            st.info("Detected Wikipedia list page")
            species_set = scrape_species(url)
        else:
            st.info("Detected individual species page")
            single_name = extract_species_name_from_species_page(url)
            species_set = [single_name] if single_name else []

        total = len(species_set)
        prog = st.progress(0)
        log = st.empty()
        success = fail = skip = 0

        for idx, name in enumerate(sorted(species_set), start=1):
            st.write(f"Processing species {idx}/{total}: {name}")
            fail_reason = ""
            # If smart search selected, override name/scientific with canonical
            if selected_display and selected_display != "[None]":
                canonical_rec = species_dict[selected_display]
                common_name = canonical_rec.common_name
                scientific_name = canonical_rec.scientific_name
            else:
                # Use scraped name as best guess; try fuzzy match if you wish
                common_name = name.split(",")[0].strip()
                scientific_name = name.split(",")[1].strip() if "," in name else None

            # --- Use .first() instead of .all() and match on canonical name
            rec = session.query(SpeciesEmbedding).filter_by(common_name=common_name).first()

            # --- Skip if already embedded
            #if rec and rec.image_embedding is not None and len(rec.image_embedding) > 0:
            #    skip += 1
            #    continue

            # Build Wikipedia API URL
            params = {
                "action": "query",
                "titles": common_name,
                "prop": "pageimages",
                "format": "json",
                "piprop": "original|thumbnail",
                "pithumbsize": 500
            }
            api_url = requests.Request("GET", WIKI_API_URL, params=params).prepare().url

            if not rec:
                rec = SpeciesEmbedding(
                    common_name=common_name,
                    scientific_name_clean=scientific_name,
                    category=category,
                    api_url=api_url,
                )
                session.add(rec)
                session.commit()

            # --- Scrape Description ---
            description = scrape_species_description(common_name)
            if not description and user_description:
                description = user_description
                st.info("Using user description")
            if description:
                rec.description = description
                # Generate text embedding for description
                try:
                    desc_embedding = generate_openclip_text_embedding(description)
                    rec.text_embedding = desc_embedding
                    rec.image_description = description
                    session.commit()
                except Exception as e:
                    st.warning(f"Error embedding description for {common_name}: {e}")

            # --- Determine image path ---
            if rec.image_path and os.path.exists(rec.image_path):
                img_path = rec.image_path
            else:
                # --- Fetch image and embed ---
                img_path = fetch_species_image(common_name, category, api_url)
                if img_path:
                    rec.image_path = img_path  # update DB with new path
                    session.commit()

            # --- Fetch image and embed ---
            if not img_path or not os.path.exists(img_path):
                rec.status = 'no_image'
                session.commit()
                fail += 1
                st.error(f"No image available for {common_name}.")
                continue

            try:
                vec = generate_openclip_image_embedding(img_path)
                rec.image_path = img_path
                rec.image_embedding = vec
                rec.status = 'embedded'
                session.commit()
                success += 1
            except Exception as e:
                rec.status = 'error'
                fail_reason = f"Embedding failed: {e}"
                st.warning(f"Error embedding image for {common_name}: {e}")
                session.commit()
                fail += 1
                st.error(f"[{idx}/{total}] Failed: {fail_reason}")
                continue

            # --- Scrape or refresh description (always try to keep this updated) ---
            description = scrape_species_description(common_name)
            if description:
                rec.description = description
                session.commit()  # Save the description right away

                # --- Embed the description only if missing or empty ---
                if not rec.description_embedding or (
                        hasattr(rec.description_embedding, "__len__") and len(rec.description_embedding) == 0):
                    try:
                        desc_embedding = generate_openclip_text_embedding(description)
                        rec.description_embedding = desc_embedding
                        session.commit()
                        st.info(f"Text embedding created for {common_name}")
                    except Exception as e:
                        st.warning(f"Error embedding description for {common_name}: {e}")
            else:
                st.warning(f"No description found for {common_name}")

            prog.progress(idx / total)
            log.text(f"{idx}/{total} ✔️{success} ❌{fail} ⏭️{skip}")

        # --- Final Metadata Sync ---


        session.close()
        st.success(f"Done. Success: {success}, Failures: {fail}, Skipped: {skip}")
