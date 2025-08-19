from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from db.species_model import SpeciesColorProfile

def compute_color_similarity(image_colors: dict, species_colors: dict, vocab: list[str]) -> float:
    """
    Compute cosine similarity between image and species color vectors.

    Args:
        image_colors (dict): e.g., {"gray": 0.42, "black": 0.48}
        species_colors (dict): e.g., {"gray": 0.40, "white": 0.60}
        vocab (list): list of all possible color names (e.g., from species profile)

    Returns:
        float: cosine similarity between 0.0 and 1.0
    """

    def build_vector(color_dict):
        return np.array([color_dict.get(color, 0.0) for color in vocab])

    vec_img = build_vector(image_colors)
    vec_spc = build_vector(species_colors)

    if np.linalg.norm(vec_img) == 0 or np.linalg.norm(vec_spc) == 0:
        return 0.0  # no similarity if either vector is all zeros

    return float(cosine_similarity([vec_img], [vec_spc])[0][0])


def get_color_vocab(session):
    rows = session.execute(text("""
        SELECT DISTINCT color_0 FROM wildlife.species_color_profile
        UNION
        SELECT DISTINCT color_1 FROM wildlife.species_color_profile
        UNION
        SELECT DISTINCT color_2 FROM wildlife.species_color_profile
    """)).fetchall()
    return [row[0] for row in rows if row[0]]


def get_species_color_profile(session, common_name: str) -> dict:
    row = (
        session.query(SpeciesColorProfile)
        .filter(SpeciesColorProfile.common_name == common_name)
        .first()
    )
    if row and row.colors:
        return row.colors
    return {}  # fallback


def get_image_colors(session, image_id):
    row = session.execute(text("""
        SELECT colors FROM wildlife.image_feature
        WHERE image_id = :image_id
    """), {"image_id": image_id}).fetchone()
    return row[0] if row and row[0] else {}