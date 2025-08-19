"""
backfill_colors_profile_colors.py — Normalize Species Color Profiles
-----------------------------------------------------------

This script fills the `colors` JSONB field in the `SpeciesColorProfile` table
using existing discrete color fields (`color_0`, `color_1`, `color_2`, and their
respective percentages).

The `colors` field is structured as:
    {
        "brown": 0.61234,
        "gray": 0.31111,
        "white": 0.07655
    }

Only rows where `colors IS NULL` will be processed and updated.

Features:
* Consolidates flat color fields into a JSONB map
* Skips rows that already have color JSON
* Automatically rounds percentages to 5 decimal places

Dependencies:
- SQLAlchemy session (`SessionLocal`)
- `SpeciesColorProfile` ORM model
"""

from db.db import SessionLocal
from db.species_model import SpeciesColorProfile


def populate_colors_json():
    """
    Converts discrete color fields into a unified JSON field (`colors`)
    for all species where it is currently missing.

    Logic:
    - For each row in SpeciesColorProfile where `colors` is NULL:
        - Create a color → percentage dictionary from non-null fields
        - Assign the dictionary to the `colors` column
        - Print a confirmation message

    Commits all changes in a single transaction.
    """
    session = SessionLocal()

    try:
        # Fetch all species profiles that are missing color JSON
        rows = (
            session.query(SpeciesColorProfile)
            .filter(SpeciesColorProfile.colors.is_(None))
            .all()
        )

        for row in rows:
            color_dict = {}

            # Aggregate non-null color entries
            if row.color_0 and row.color_0_pct is not None:
                color_dict[row.color_0] = round(row.color_0_pct, 5)
            if row.color_1 and row.color_1_pct is not None:
                color_dict[row.color_1] = round(row.color_1_pct, 5)
            if row.color_2 and row.color_2_pct is not None:
                color_dict[row.color_2] = round(row.color_2_pct, 5)

            # Save the JSON dict if it contains any data
            if color_dict:
                row.colors = color_dict
                print(f"✅ Updated colors for {row.common_name}: {color_dict}")

        session.commit()

    finally:
        session.close()


# --- Run it as a script ---
if __name__ == "__main__":
    populate_colors_json()
