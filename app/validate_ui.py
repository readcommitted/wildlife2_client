"""
validate_ui.py — Image Metadata Validation & Embedding Pipeline
-------------------------------------------------------------

This Streamlit module allows users to validate and enrich image metadata
after classification. It provides an interactive UI to:

- View images pending metadata updates
- Manually assign species labels, behavior notes
- Select known locations from a smart searchable dropdown
- Update geolocation fields based on location lookup
- Generate a semantic text embedding using OpenAI for each image
- Commit all updates to the PostgreSQL database

The embeddings support semantic search and downstream analysis.

Dependencies:
- Streamlit
- SQLAlchemy
- OpenAI client API (v1)
- PIL for image display

"""

import streamlit as st
from pathlib import Path
from PIL import Image
from db.db import SessionLocal
from db.image_model import ImageHeader, ImageLabel, ImageEmbedding
import time
from db.species_model import SpeciesFlattened
from db.location_model import LocationLookup
from tools.openai_utils import get_embedding
from datetime import datetime

from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

# --- Expandable Process Overview ---
with st.expander("Show/Hide Validation Process", expanded=False):
    st.header("Validation Process Overview")
    st.write(
        """
        This process allows updating species labels, behaviors, and locations after classification.

        **Steps:**
        1. View pending images with incomplete metadata
        2. Manually assign species labels, behaviors
        3. Select known locations from dropdown (auto-fills coordinates)
        4. Generate AI-based text embeddings for semantic search
        5. Commit all updates to the database
        """
    )


# --- normalize SpeciesNet output to a stable shape ---
def normalize_speciesnet_output(preds):
    """
    Returns: {"label": str|None, "confidence": float|None, "candidates": List[{"label": str, "confidence": float, ...}], "raw": preds}
    Works with either:
      - dicts like {"prediction": {...}, "classifications": {...}, ...}
      - lists like [{"common_name": "...", "confidence": 0.92}, ...]
    """
    label = None
    conf = None
    candidates = []

    if isinstance(preds, dict):
        # Ensemble-style single best prediction
        if isinstance(preds.get("prediction"), dict):
            pred = preds["prediction"]
            label = pred.get("label") or pred.get("class") or pred.get("common_name")
            conf = float(pred.get("score") or pred.get("confidence") or 0)
        # Classifications list style
        if isinstance(preds.get("classifications"), dict):
            cls = preds["classifications"]
            classes = cls.get("classes") or []
            scores = cls.get("scores") or []
            for c, s in zip(classes, scores):
                candidates.append({"label": c, "confidence": float(s)})
            if label is None and candidates:
                label, conf = candidates[0]["label"], candidates[0]["confidence"]
        # Detector-only fallback (rarely used for label)
        if not candidates and "detections" in preds and isinstance(preds["detections"], list):
            for d in preds["detections"]:
                candidates.append({"label": d.get("label"), "confidence": float(d.get("conf", 0.0))})
            if label is None and candidates:
                label, conf = candidates[0]["label"], candidates[0]["confidence"]

    elif isinstance(preds, list):
        # List of candidate dicts like {"common_name": "...", "confidence": 0.xx, ...}
        for item in preds:
            if isinstance(item, dict):
                c_label = item.get("label") or item.get("class") or item.get("common_name")
                c_conf  = float(item.get("confidence") or item.get("score") or 0)
                if c_label:
                    candidates.append({"label": c_label, "confidence": c_conf, **{k:v for k,v in item.items() if k not in ["label","class","common_name","confidence","score"]}})
        if candidates:
            label, conf = candidates[0]["label"], candidates[0]["confidence"]

    return {"label": label, "confidence": conf, "candidates": candidates, "raw": preds}


def get_season_from_date(date_obj: datetime) -> str:
    """
    Returns the season (spring, summer, fall, winter) based on month of date.
    """
    month = date_obj.month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return "winter"


def build_text_embedding(image_id, session):
    """
    Constructs and stores a semantic text embedding based on:
    - species label
    - location description
    - behavior notes
    """
    image = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
    if not image:
        return

    species_label = session.query(ImageLabel.label_value).filter_by(image_id=image_id, label_type="user").scalar()
    location = image.location_description or ""
    behavior = image.behavior_notes or ""

    prompt = f"{species_label}. {location}. {behavior}"
    embedding = get_embedding(prompt)

    session.query(ImageEmbedding).filter_by(image_id=image_id).update({"text_embedding": embedding})


def smart_species_match(label_value, session):
    """
    Attempts to resolve a species label to a known species_id using fuzzy match.
    Returns species_id if matched, otherwise -1.
    """
    if not label_value:
        return None
    result = session.query(SpeciesFlattened).filter(
        SpeciesFlattened.common_name.ilike(f"%{label_value}%")
    ).first()
    return result.species_id if result else -1


def show_processed_images():
    """
    Streamlit UI to:
    - Display images pending metadata update
    - Provide species label, behavior, tag, location entry
    - Allow batch updates to selected images
    - Trigger text embedding generation
    """
    session = SessionLocal()
    image_records = (
        session.query(ImageHeader)
        .filter(ImageHeader.metadata_updated == False)
        .order_by(ImageHeader.capture_date.asc())
        .all()
    )

    species_options = session.query(SpeciesFlattened.common_name).order_by(SpeciesFlattened.common_name).all()
    locations = session.query(LocationLookup).order_by(LocationLookup.location_description).all()
    species_list = [s[0] for s in species_options if s[0]]

    if not image_records:
        st.info("No images pending metadata update.")
        return

    # Session State Initialization
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = set()
    if "page" not in st.session_state:
        st.session_state.page = 0
    if "per_page" not in st.session_state:
        st.session_state.per_page = 25
    if "metadata_fields" not in st.session_state:
        st.session_state.metadata_fields = {
            "species_label": "",
            "behavior_notes": "",
            "park": "",
            "place": ""
        }

    start = st.session_state.page * st.session_state.per_page
    end = start + st.session_state.per_page
    page_records = image_records[start:end]

    # Metadata Entry Form
    with st.form("validation_form", clear_on_submit=False):
        species_label = st.selectbox(
            "Species Label (smart search)",
            options=[""] + species_list,
            index=0,
            help="Start typing to search for a species by common name."
        )
        behavior_notes = st.text_input("Behavior Notes", value=st.session_state.metadata_fields["behavior_notes"])

        location_options = [l.location_description for l in locations if l.location_description]
        selected_location = st.selectbox(
            "Location Description (Smart Search)",
            [""] + location_options,
            help="Choose a known location from the list"
        )
        # At the very top of your page render:
        status_placeholder = st.empty()

        submit = st.form_submit_button("✅ Update Selected Images")

    # Pagination Controls
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⏮️ Previous", disabled=st.session_state.page == 0):
            st.session_state.page -= 1
    with col2:
        if st.button("Next ⏭️", disabled=end >= len(image_records)):
            st.session_state.page += 1

    # Display Settings
    st.markdown("## Select images to update")
    col1, col2 = st.columns(2)
    with col1:
        thumb_width = 150

    from config.settings import JPG_DIR

    # Image Grid
    for record in page_records:
        jpeg_key = record.jpeg_path

        try:
            # Use local file path
            local_jpg_path = JPG_DIR / jpeg_key.replace("jpg/", "")
            img = Image.open(local_jpg_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load local image: {local_jpg_path} ({e})")
            continue

        species = session.query(SpeciesFlattened).filter_by(species_id=record.species_id).first()
        common_name = species.common_name if species else "Unknown"

        st.markdown("""
            <style>
            .left-aligned-container {
                max-width: 600px;
                margin-left: 0;
                margin-right: auto;
                padding: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="left-aligned-container">', unsafe_allow_html=True)
            cols = st.columns([0.2, .5, 2])  # image, text, checkbox
            with cols[0]:
                st.image(img, width=thumb_width)

            with cols[1]:
                st.markdown(f"**{record.image_name}**  \nSpecies: *{common_name}*  \nLocation: *{record.location_description or 'N/A'}*")

            with cols[2]:
                default_selected = record.image_id in st.session_state.selected_images
                selected = st.checkbox("Select", value=default_selected, key=f"check_{record.image_id}")
                if selected:
                    st.session_state.selected_images.add(record.image_id)
                else:
                    st.session_state.selected_images.discard(record.image_id)

    # --- Form Submission Handler ---
    if submit:
        try:
            matched_species_id = smart_species_match(species_label.strip(), session) if species_label.strip() else -1
            loc = session.query(LocationLookup).filter_by(location_description=selected_location).first() if selected_location else None

            for image_id in st.session_state.selected_images:
                if species_label.strip():
                    session.add(ImageLabel(
                        image_id=image_id,
                        label_type="user",
                        label_value=species_label.strip(),
                        label_source="user",
                        confidence=1.0,
                        species_id=matched_species_id
                    ))

                image = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()

                # Add season tag
                if image.capture_date:
                    season = get_season_from_date(image.capture_date)
                    season_tag = f"season: {season}"
                    if behavior_notes:
                        if f"Observed in {season}" not in behavior_notes:
                            behavior_notes += f" Observed in {season}."
                    else:
                        behavior_notes = f"Observed in {season}."

                update_fields = {
                    ImageHeader.metadata_updated: True,
                    ImageHeader.behavior_notes: behavior_notes or None,

                }

                if loc:
                    update_fields.update({
                        ImageHeader.location_description: loc.location_description,
                        ImageHeader.park: loc.park,
                        ImageHeader.place: loc.place,
                        ImageHeader.state_code: loc.state,
                        ImageHeader.country_code: loc.country,
                        ImageHeader.county: loc.county
                    })
                    if (not image.latitude or image.latitude == 0) and (not image.longitude or image.longitude == 0):
                        update_fields.update({
                            ImageHeader.latitude: loc.latitude,
                            ImageHeader.longitude: loc.longitude
                        })

                session.query(ImageHeader).filter(ImageHeader.image_id == image_id).update(update_fields)
                build_text_embedding(image_id, session)

            session.commit()
            status_placeholder.success(f"✅ Updated {len(st.session_state.selected_images)} image(s)")

            # --- Phase 2: SpeciesNet run + persist results ---
            from core.speciesnet_api import run_speciesnet_one
            from core.speciesnet_update import apply_speciesnet_result
            from config.settings import JPG_DIR

            target_ids = list(st.session_state.selected_images)
            sn_results = {}

            # 1) Compute predictions
            for image_id in target_ids:
                img = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
                if not img or not img.jpeg_path:
                    st.warning(f"[SN] Image {image_id}: no jpeg_path")
                    continue

                jpg_local = (JPG_DIR / img.jpeg_path.replace("jpg/", "")).resolve()
                if not jpg_local.exists():
                    st.warning(f"[SN] missing file for {image_id}: {jpg_local}")
                    continue

                try:
                    sn_out = run_speciesnet_one(jpg_local, country=None, admin1_region=None)
                    sn_results[image_id] = sn_out

                    # quick display
                    rec = sn_out[0] if isinstance(sn_out, list) and sn_out else (
                        sn_out if isinstance(sn_out, dict) else {})
                    pred = rec.get("prediction")
                    score = rec.get("prediction_score")
                    if not pred:
                        cls = rec.get("classifications") or {}
                        if cls.get("classes"):
                            pred = cls["classes"][0]
                            score = cls["scores"][0] if cls.get("scores") else None
                    name = pred.split(";")[-1] if isinstance(pred, str) else None
                    try:
                        score = float(score) if score is not None else None
                    except Exception:
                        pass
                    st.info(
                        f"🧠 [{image_id}] SpeciesNet → {name or '—'} ({f'{score:.3f}' if score is not None else '—'})")
                except Exception as e:
                    st.warning(f"[SN] {image_id} failed: {e}")

            # 2) Persist all results (same session; single commit)
            persisted = 0
            try:
                for image_id, preds in sn_results.items():
                    summary = apply_speciesnet_result(session, image_id, preds)  # <-- CORRECT CALL
                    status_placeholder.success(f"wrote image_model_result id={summary['image_model_id']} for image {image_id}")
                    persisted += 1

                session.commit()
                status_placeholder.success(f"Saved SpeciesNet results for {persisted} image(s).")
            except Exception as e:
                session.rollback()
                st.error(f"Failed to save SpeciesNet results: {e}")

            # cleanup + rerun
            st.session_state.selected_images.clear()
            st.session_state.metadata_fields = {"species_label": "", "behavior_notes": "", "park": "", "place": ""}
            time.sleep(1)
            st.rerun()



        except Exception as e:
            session.rollback()
            st.error(f"Database update failed: {e}")
        finally:
            session.close()


# Run the validation interface
show_processed_images()
