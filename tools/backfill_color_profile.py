"""
species_color_backfill.py — Species Color Profile Generator
------------------------------------------------------------

This script computes the dominant color profile for each species using
their representative images stored in the `SpeciesEmbedding` table. It
clusters image pixels using k-means and maps cluster centers to a
predefined color palette.

The results are written to the `SpeciesColorProfile` table, storing
the top 3 named colors and their relative percentages.

Features:
* Uses OpenCV + k-means to extract color clusters
* Maps RGB clusters to nearest named palette colors
* Loads species images from `SpeciesEmbedding.image_path`
* Writes results to `SpeciesColorProfile` with upsert logic

Dependencies:
- OpenCV (cv2)
- NumPy
- scikit-learn
- SQLAlchemy
"""

import cv2
import os
import json
import numpy as np
from collections import Counter
from sqlalchemy.dialects.postgresql import insert
from db.species_model import SpeciesEmbedding, SpeciesColorProfile
from db.db import SessionLocal
from core.ingest import load_palette_from_db, rgb_to_palette_color


def extract_top_colors(image_path, palette, top_k=3):
    """
    Extracts the top `k` dominant named colors from an image.

    Args:
        image_path (str): Path to the image file.
        palette (dict): Loaded palette used to map RGB to named colors.
        top_k (int): Number of top colors to extract (default = 3).

    Returns:
        List of (color_name, proportion) tuples, ordered by prominence.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten image pixels to a 2D array
    pixels = image.reshape(-1, 3)

    # Apply KMeans clustering to find top_k color clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=top_k, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    label_counts = Counter(labels)
    total = sum(label_counts.values())

    # Map cluster centers to named palette colors
    top_colors = []
    for i, center in enumerate(centers):
        r, g, b = map(int, center)
        pct = label_counts[i] / total
        name = rgb_to_palette_color(r, g, b, palette)
        top_colors.append((name, round(pct, 5)))

    return top_colors


def upsert_species_color_profile(session, common_name, color_data):
    """
    Inserts or updates a row in the SpeciesColorProfile table.

    Args:
        session (Session): SQLAlchemy DB session.
        common_name (str): Common name of the species.
        color_data (list): List of (color_name, percentage) tuples.
    """
    values = {
        "common_name": common_name,
        "color_0": color_data[0][0] if len(color_data) > 0 else None,
        "color_0_pct": color_data[0][1] if len(color_data) > 0 else None,
        "color_1": color_data[1][0] if len(color_data) > 1 else None,
        "color_1_pct": color_data[1][1] if len(color_data) > 1 else None,
        "color_2": color_data[2][0] if len(color_data) > 2 else None,
        "color_2_pct": color_data[2][1] if len(color_data) > 2 else None,
    }

    # Upsert color profile using common_name as the unique key
    stmt = insert(SpeciesColorProfile).values(**values).on_conflict_do_update(
        index_elements=['common_name'],
        set_={
            "color_0": values["color_0"],
            "color_0_pct": values["color_0_pct"],
            "color_1": values["color_1"],
            "color_1_pct": values["color_1_pct"],
            "color_2": values["color_2"],
            "color_2_pct": values["color_2_pct"],
        }
    )
    session.execute(stmt)


def backfill_species_color_profile():
    """
    Main function to iterate over species and generate color profiles.

    Loads the color palette, reads all entries in `SpeciesEmbedding`,
    extracts colors from the associated image, and stores the top 3
    named colors in `SpeciesColorProfile`.
    """
    session = SessionLocal()
    palette = load_palette_from_db()

    species_rows = session.query(SpeciesEmbedding).all()

    for row in species_rows:
        try:
            print(f"Processing: {row.common_name}")

            # Skip if the image doesn't exist
            if not os.path.exists(row.image_path):
                print(f"Missing image: {row.image_path}")
                continue

            # Extract top named colors
            color_data = extract_top_colors(row.image_path, palette)

            if not color_data:
                print(f"No color data for {row.common_name}")
                continue

            # Insert or update color profile
            upsert_species_color_profile(session, row.common_name, color_data)

        except Exception as e:
            print(f"❌ Error on {row.common_name}: {e}")

    session.commit()
    session.close()
    print("Backfill complete!")


if __name__ == "__main__":
    backfill_species_color_profile()
