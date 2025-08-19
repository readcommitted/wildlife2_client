"""
ingest.py — Image Ingestion & Preprocessing UI
------------------------------------------------

This Streamlit module handles the ingestion and preprocessing of RAW wildlife images
prior to classification and analysis. It provides an interactive UI to:

- Monitor staged RAW images pending processing
- Execute the ingestion pipeline with visual progress updates
- Clear previous results and reset the staging environment
- Display an expandable process overview for transparency

Ingestion Pipeline Includes:
1. EXIF Metadata extraction from RAW (NEF) files
2. Capture date parsing and directory organization
3. RAW to JPEG conversion using rawpy and OpenCV
4. Saving processed files to structured data lake directories
5. Metadata insertion into the PostgreSQL database

Species Identification Pipeline Includes:
- Running the SpeciesNet model on processed images
- Storing predictions in JSON and updating the database

Dependencies:
- Streamlit for UI
- rawpy, OpenCV for image processing (handled in core.ingest)
- exiftool (external) for metadata extraction
- PostgreSQL backend
- SpeciesNet model for classification


"""

import streamlit as st
from tools.spaces import list_objects, delete_file
import core.ingest
import streamlit as st
from pathlib import Path
from config.settings import LOCAL, STAGE_DIR

from config.settings import APP_MODE
if APP_MODE.lower() == "demo":
    st.title("Demo")
    st.error("🔒 Not available in the demo.")
    st.stop()

def list_stage_nefs():
    """Return a list of NEF items (Paths if LOCAL, keys if CLOUD)."""
    if LOCAL:
        if not STAGE_DIR.exists():
            return []
        return [p for p in STAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".nef"]
    else:
        try:
            return [k for k in list_objects("stage/") if k.lower().endswith(".nef")]
        except Exception as e:
            st.error(f"Spaces error listing 'stage/': {e}")
            return []


# --- Expandable Process Overview UI ---
with st.expander("Show/Hide Ingestion Process Overview", expanded=False):
    st.header("1. Ingestion Process")
    st.write(
        """
        This process is responsible for processing RAW image files (NEF format) and preparing them for analysis.

        **Steps:**
        1. **Extract Metadata**: Uses `exiftool` to extract EXIF metadata from the RAW files.
        2. **Extract Capture Date**: Uses the metadata to get the image capture date.
        3. **Create Data Lake Directories**: Organizes images by date in a structured folder format.
        4. **Convert RAW to JPEG**:
           - Uses `rawpy` for RAW image reading.
           - Uses `OpenCV` for JPEG conversion and saving.
        5. **Move Files to Data Lake**: Saves the processed JPEG and RAW files in their respective directories.
        6. **Insert Metadata into Database**: Saves image metadata (including EXIF) to the PostgreSQL database.
        """
    )

    st.header("2. Species Identification Process")
    st.write(
        """
        This process is dedicated to running the SpeciesNet model on the processed images (JPEG).

        **Steps:**
        1. **Load Processed Images**: Reads all JPEG images from the `stage_processed` directory.
        2. **Run SpeciesNet Model**:
           - Uses the `speciesnet.scripts.run_model` script.
           - The model analyzes the images and generates species predictions.
        3. **Update Results**:
           - Saves predictions to a JSON file (`speciesnet_results.json`).
           - Updates species information in the PostgreSQL database.
        """
    )

    st.header("3. Models Used")
    st.write(
        """
        - **SpeciesNet Model**: Custom model for species identification.
        - **EXIF Metadata Extraction**: Uses `exiftool` for metadata extraction from RAW images.
        - **Image Processing**: Uses `rawpy` for RAW file decoding and `OpenCV` for image processing.
        """
    )


# --- Progress Log Helper ---
def update_status(message):
    """
    Append a status message to the progress log and display it in a scrollable, auto-scrolling container.
    """
    import streamlit as st

    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []

    st.session_state.progress_log.append(message)

    full_log_html = "<br>".join(st.session_state.progress_log)

    st.session_state.progress_area.markdown(
        f"""
        <div style='height: 350px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; font-family: monospace;' id='logbox'>
            {full_log_html}
        </div>
        <script>
            var logBox = document.getElementById('logbox');
            if (logBox) {{
                logBox.scrollTop = logBox.scrollHeight;
            }}
        </script>
        """,
        unsafe_allow_html=True
    )



if LOCAL:
    st.caption(f"LOCAL mode — STAGE_DIR: `{STAGE_DIR}`")
else:
    st.caption("CLOUD mode — staging prefix: `stage/`")


# --- Check RAW files in stage ---
stage_items = list_stage_nefs()

if stage_items:
    count = len(stage_items)
    st.info(f"{count} RAW image{'s' if count != 1 else ''} ready for processing.")
    process_clicked = st.button("Process Images")
else:
    st.warning("No RAW images found in staging.")
    process_clicked = False


# --- Set Up Empty Placeholder for Log ---
st.session_state.progress_area = st.empty()

if process_clicked:
    # Reset previous logs
    st.session_state.progress_log = []
    st.session_state.progress_text = "Starting ingestion..."

    # Show header for progress updates
    st.session_state.progress_area.text(st.session_state.progress_text)

    # --- Run Ingestion ---
    core.ingest.process_raw_images(update_status)

    st.success("✅ Images processed and metadata written to database.")
