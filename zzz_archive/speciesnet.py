"""
speciesnet.py — SpeciesNet Execution & Post-Processing Pipeline
---------------------------------------------------------------

This module handles running the SpeciesNet model on staged images,
processing results, and updating the database with species predictions
and refreshed embeddings.

Key Features:
- Invokes SpeciesNet as an external subprocess
- Passes country/state location metadata to improve predictions
- Monitors real-time output, stripping ANSI color codes for display
- Updates database records based on prediction results
- Regenerates image embeddings for newly processed images

Intended for:
- Batch species detection after ingestion and validation of new images

Dependencies:
- Subprocess for external command execution
- SQLAlchemy for database queries
- species_update for applying predictions
- regenerate_image_embeddings for post-processing

"""

import subprocess
import re
from core.species_update import update_species_from_predictions
from db.db import SessionLocal
from db.image_model import ImageHeader
from config.settings import STAGE_PROCESSED_DIR, PREDICTIONS_JSON


def run_speciesnet(update_status=None):
    """
    Runs the SpeciesNet model on staged JPEG images, monitors output,
    and updates database records with predictions.

    Args:
        update_status (function, optional): Function to display real-time status messages.
    """
    jpeg_files = list(STAGE_PROCESSED_DIR.glob("*.png"))
    if not jpeg_files:
        if update_status:
            update_status("No processed images found for SpeciesNet.")
        return

    if update_status:
        update_status(f"Running SpeciesNet on {len(jpeg_files)} images...")

    # --- Determine region for location-aware predictions ---
    session = SessionLocal()
    region = (
        session.query(
            ImageHeader.country_code,
            ImageHeader.state_code
        )
        .filter(
            ImageHeader.current_batch == True,
            ImageHeader.country_code.isnot(None),
            ImageHeader.state_code.isnot(None)
        )
        .distinct()
        .first()
    )
    session.close()

    # Fall back to default region if unset
    country = region.country_code if region and region.country_code else "USA"
    state = region.state_code if region and region.state_code else "WY"

    cmd = [
        "python", "-u", "-m", "speciesnet.scripts.run_model",
        "--folders", str(STAGE_PROCESSED_DIR),
        "--predictions_json", str(PREDICTIONS_JSON),
        "--country", country,
        "--admin1_region", state
    ]

    # Clean ANSI color codes from output for display in UI
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

    # --- Run SpeciesNet as Subprocess ---
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            clean_line = ansi_escape.sub('', line).strip()
            if update_status:
                update_status(clean_line)


    # --- Post-Processing ---
    if process.returncode == 0:
        if update_status:
            update_status("SpeciesNet completed successfully.")

        update_species_from_predictions(PREDICTIONS_JSON)

    else:
        if update_status:
            update_status("SpeciesNet failed. Check the output above for details.")
