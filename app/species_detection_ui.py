"""
species_detection_ui.py — Species Identification with SpeciesNet
-----------------------------------------------------------------

This Streamlit module runs the SpeciesNet model to classify wildlife images
and provides tools to view detection results and model confidence.

Features:
- Runs SpeciesNet model on validated images (with location metadata)
- Displays a real-time progress log during model execution
- Summarizes detected species counts
- Visualizes confidence score distributions with interactive charts

Process Overview:
1. Select eligible images from `stage_processed` where `metadata_updated = TRUE`
2. Run SpeciesNet model (subprocess call to `speciesnet.scripts.run_model`)
3. Save predictions to JSON output
4. Match predictions to species in `species_flattened` table
5. Update `image_header` with species details and confidence scores

Dependencies:
- Streamlit for interactive UI
- Matplotlib for confidence histograms
- Pandas for data handling
- SQLAlchemy for database access
- SpeciesNet for species detection

"""

import streamlit as st
from matplotlib import pyplot as plt
from zzz_archive.speciesnet import run_speciesnet
import pandas as pd
from sqlalchemy.orm import Session
from db.db import SessionLocal
from db.image_model import ImageHeader
from db.species_model import SpeciesFlattened


from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

# --- Expandable Process Overview ---
with st.expander("Show/Hide SpeciesNet Process Overview", expanded=False):
    st.header("Species Identification with SpeciesNet")
    st.write(
        """
        This process runs the **SpeciesNet** model to identify species from JPEG images that have been validated with location metadata.

        **Steps:**
        1. **Select Eligible Images**: Uses JPEGs from the `stage_processed` directory where metadata has been manually validated (`metadata_updated = TRUE`).
        2. **Run SpeciesNet Model**:
           - Calls `speciesnet.scripts.run_model` as a subprocess.
           - Passes region/country information (currently hardcoded to Wyoming, USA).
           - Generates species predictions and saves them in `speciesnet_results.json`.
        3. **Update Database Records**:
           - Parses predictions from the JSON output.
           - Matches species by common name using the internal `species_flattened` table.
           - Updates `image_header` with:
             - `species_id` or fallback
             - `speciesnet_raw` (full prediction string)
             - `species_confidence`
             - `species_update_method = 'speciesnet'`
        """
    )

st.markdown("This will run SpeciesNet on all processed images with updated metadata.")

# --- Progress Log Setup ---
progress_log = st.empty()

# Initialize or reuse session state for log content
if "progress_log_text" not in st.session_state:
    st.session_state.progress_log_text = ""

# --- Action Buttons ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    clear_clicked = st.button("Clear Output")
with col2:
    species_clicked = st.button("Run SpeciesNet")
with col3:
    results_clicked = st.button("View Results")

# --- Clear Log Button ---
if clear_clicked:
    progress_log.empty()

# --- Run SpeciesNet Button ---
if species_clicked:
    def stream_update(msg):
        """
        Live-updates the progress log in the UI during SpeciesNet execution.
        """
        st.session_state.progress_log_text += msg + "\n"
        progress_log.text_area("📜 Progress Log", st.session_state.progress_log_text, height=300)

    run_speciesnet(update_status=stream_update)

# --- View Results Button ---
if results_clicked:
    progress_log.empty()
    session: Session = SessionLocal()

    # Query images updated by SpeciesNet in current batch
    records = (
        session.query(ImageHeader, SpeciesFlattened.common_name)
        .join(SpeciesFlattened, ImageHeader.species_id == SpeciesFlattened.species_id)
        .filter("speciesnet" == ImageHeader.species_detection_method,
                True == ImageHeader.current_batch)
        .all()
    )
    session.close()

    # Prepare results DataFrame
    data = [
        {"species": common_name, "confidence": record.species_confidence}
        for record, common_name in records
    ]
    df = pd.DataFrame(data)

    # --- Species Count Bar Chart ---
    st.subheader("🔢 Species Detection Count")
    species_counts = df["species"].value_counts()
    st.bar_chart(species_counts)

    # --- Confidence Score Distribution ---
    bins = [0, 60, 70, 80, 90, 100]
    labels = ["<60%", "60–69%", "70–79%", "80–89%", "90–100%"]
    df["confidence_pct"] = df["confidence"] * 100
    df["confidence_group"] = pd.cut(df["confidence_pct"], bins=bins, labels=labels, include_lowest=True)

    confidence_counts = df["confidence_group"].value_counts().reindex(labels[::-1])

    st.subheader("📊 Confidence Score Distribution")
    fig, ax = plt.subplots()
    confidence_counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Image Count")
    ax.set_ylabel("Confidence Range")
    ax.invert_yaxis()  # Higher confidence group at the top
    st.pyplot(fig)
