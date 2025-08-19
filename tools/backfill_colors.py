"""
backfill_colors.py — Populate Named Colors from RGB Features
-------------------------------------------------------------

This script processes existing RGB color feature data in the
`wildlife.image_feature` table and populates the `colors` JSONB field
with named color summaries based on a predefined palette.

It maps each of the top 3 dominant RGB colors (stored in
`slim_features`) to a named color using value ranges defined in
`wildlife.color_palette`.

Features:
* Reads raw RGB values from `image_feature.slim_features`
* Matches them to color names using the `color_palette` table
* Aggregates proportions into a summary dictionary (e.g., {"brown": 0.68})
* Updates the `colors` field in `image_feature` for each image

Dependencies:
- SQLAlchemy
- A valid PostgreSQL connection with access to:
  * wildlife.image_feature
  * wildlife.color_palette
"""

import json
from sqlalchemy import text
from db.db import SessionLocal


def load_color_palette(session):
    """
    Loads the named color palette from the database.

    Returns:
        List of dicts with RGB min/max thresholds and color names.
    """
    rows = session.execute(text("""
        SELECT color_name, min_r, max_r, min_g, max_g, min_b, max_b
        FROM wildlife.color_palette
    """)).mappings().fetchall()
    return [dict(row) for row in rows]


def match_color_name(r, g, b, palette):
    """
    Matches an (R, G, B) value to a named color using range matching.

    Returns:
        color_name (str) or "unknown" if no match.
    """
    for color in palette:
        if (color["min_r"] <= r <= color["max_r"] and
            color["min_g"] <= g <= color["max_g"] and
            color["min_b"] <= b <= color["max_b"]):
            return color["color_name"]
    return "unknown"


def update_colors_from_features():
    """
    Main function to populate the `colors` field in `image_feature`.

    Process:
    * Load color palette from `color_palette`
    * Read RGB data from `slim_features` in `image_feature`
    * For each image:
        * Map top 3 RGB clusters to named colors
        * Aggregate their percentages
        * Update the `colors` JSONB field
    """
    session = SessionLocal()
    palette = load_color_palette(session)

    # Query image features with extracted RGB clusters
    rows = session.execute(text("""
        SELECT image_id, slim_features
        FROM wildlife.image_feature
        WHERE slim_features IS NOT NULL
    """)).mappings().fetchall()

    for row in rows:
        image_id = row["image_id"]
        features = row["slim_features"]

        color_summary = {}

        # Process top 3 dominant colors
        for i in range(3):
            r = features.get(f"dom_color_{i}_r")
            g = features.get(f"dom_color_{i}_g")
            b = features.get(f"dom_color_{i}_b")
            pct = features.get(f"dom_color_{i}_pct", 0)

            if None in (r, g, b):
                continue

            color_name = match_color_name(r, g, b, palette)
            if color_name != "unknown":
                color_summary[color_name] = color_summary.get(color_name, 0.0) + pct

        # Skip if no valid named colors found
        if not color_summary:
            continue

        # Write summary to the `colors` JSONB field
        session.execute(text("""
            UPDATE wildlife.image_feature
            SET colors = :colors
            WHERE image_id = :image_id
        """), {
            "image_id": image_id,
            "colors": json.dumps(color_summary)
        })

    session.commit()
    session.close()


if __name__ == "__main__":
    update_colors_from_features()
