"""
generate_species_color_profile.py — Aggregate Species Color Profiles
---------------------------------------------------------------------

This script computes the average color profile for each species by
aggregating dominant color data across all labeled images in the
`image_feature` table. The final profiles are stored in the
`species_color_profile` table.

Features:
* Loads RGB → color name mappings from `color_palette`
* Joins user-labeled images with extracted color features
* Aggregates and averages color contributions across all images
* Stores top 3 named colors and percentages per species
* Uses upsert logic to update existing species records

Also includes a helper function for computing color similarity
between an image and a species profile using cosine similarity.

Dependencies:
- SQLAlchemy
- collections.defaultdict
- NumPy, scikit-learn (for similarity function only)
"""

from collections import defaultdict
from sqlalchemy import text
from db.db import SessionLocal


def load_color_palette(session):
    """
    Load the color palette from the database and return a list of color rules.

    Returns:
        List[dict]: Each item contains color_name + RGB min/max bounds.
    """
    palette_rows = session.execute(text("""
        SELECT color_name, min_r, max_r, min_g, max_g, min_b, max_b
        FROM wildlife.color_palette
    """)).mappings().fetchall()

    palette = []
    for row in palette_rows:
        palette.append({
            "name": row["color_name"],
            "min_r": row["min_r"], "max_r": row["max_r"],
            "min_g": row["min_g"], "max_g": row["max_g"],
            "min_b": row["min_b"], "max_b": row["max_b"]
        })
    return palette


def match_color_name(r, g, b, palette):
    """
    Map an RGB value to a named color using the palette ranges.

    Returns:
        str: Color name or 'unknown'
    """
    for color in palette:
        if (color["min_r"] <= r <= color["max_r"] and
                color["min_g"] <= g <= color["max_g"] and
                color["min_b"] <= b <= color["max_b"]):
            return color["name"]
    return "unknown"


def generate_species_color_profile():
    """
    Main function to compute and store color profiles per species.

    * Aggregates dominant RGB colors from labeled images
    * Averages contributions per species
    * Stores top 3 named colors in `species_color_profile`
    """
    session = SessionLocal()
    palette = load_color_palette(session)

    # Fetch image features joined with user-labeled species
    rows = session.execute(text("""
        SELECT l.label_value AS common_name, f.slim_features
        FROM wildlife.image_feature f
        JOIN wildlife.image_label l ON f.image_id = l.image_id
        WHERE l.label_type = 'user' AND f.slim_features IS NOT NULL
    """)).mappings().fetchall()

    color_totals = defaultdict(lambda: defaultdict(float))  # {species: {color_name: cumulative_pct}}
    species_counts = defaultdict(int)  # {species: image_count}

    for row in rows:
        common_name = row["common_name"]
        features = row["slim_features"]  # Already a Python dict

        # Loop through top 3 dominant colors
        for i in range(3):
            r = features.get(f'dom_color_{i}_r')
            g = features.get(f'dom_color_{i}_g')
            b = features.get(f'dom_color_{i}_b')
            pct = features.get(f'dom_color_{i}_pct', 0)
            if r is not None and g is not None and b is not None:
                color_name = match_color_name(r, g, b, palette)
                color_totals[common_name][color_name] += pct

        species_counts[common_name] += 1

    # Aggregate + store top 3 averaged colors per species
    for common_name, color_dict in color_totals.items():
        count = species_counts[common_name]
        if count == 0:
            continue

        # Normalize by image count
        averaged = {k: v / count for k, v in color_dict.items()}
        top_colors = sorted(averaged.items(), key=lambda x: x[1], reverse=True)[:3]

        # Pad to always include 3 values
        while len(top_colors) < 3:
            top_colors.append(("", 0.0))

        session.execute(text("""
            INSERT INTO wildlife.species_color_profile (
                common_name,
                color_0, color_0_pct,
                color_1, color_1_pct,
                color_2, color_2_pct
            )
            VALUES (:common_name, :c0, :c0p, :c1, :c1p, :c2, :c2p)
            ON CONFLICT (common_name) DO UPDATE SET
                color_0 = EXCLUDED.color_0,
                color_0_pct = EXCLUDED.color_0_pct,
                color_1 = EXCLUDED.color_1,
                color_1_pct = EXCLUDED.color_1_pct,
                color_2 = EXCLUDED.color_2,
                color_2_pct = EXCLUDED.color_2_pct
        """), {
            "common_name": common_name,
            "c0": top_colors[0][0], "c0p": top_colors[0][1],
            "c1": top_colors[1][0], "c1p": top_colors[1][1],
            "c2": top_colors[2][0], "c2p": top_colors[2][1],
        })

    session.commit()
    session.close()


def compute_color_similarity(image_color_dict, species_color_dict, vocab):
    """
    Compute cosine similarity between an image's color vector and species profile.

    Args:
        image_color_dict (dict): Named color → proportion
        species_color_dict (dict): Named color → proportion
        vocab (list): List of all possible color names to use for vector construction

    Returns:
        float: Cosine similarity between the two vectors
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def build_vector(color_dict):
        return np.array([color_dict.get(c, 0.0) for c in vocab])

    vec_image = build_vector(image_color_dict)
    vec_species = build_vector(species_color_dict)
    return float(cosine_similarity([vec_image], [vec_species])[0][0])


if __name__ == "__main__":
    generate_species_color_profile()
