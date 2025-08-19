"""
species_lookup.py — Species Matching Utility
---------------------------------------------

Provides intelligent species matching based on common names with:

* Case-insensitive exact matching
* Alias correction for known spelling variations (e.g., "gray" vs. "grey")
* Fuzzy fallback using SQL ILIKE for partial matches

Used during species identification pipelines to map predicted labels
to canonical species IDs in the `species_flattened` table.

Dependencies:
- SQLAlchemy ORM session
- `species_flattened` as authoritative species table

"""

from sqlalchemy import func
from sqlalchemy.orm import Session
from db.species_model import SpeciesFlattened

# Known alias corrections for label standardization
ALIASES = {
    "gray wolf": "grey wolf",
    "grey heron": "gray heron",
    "grey seal": "gray seal",
    "grey falcon": "gray falcon",
    "grey partridge": "gray partridge",
    "grey crowned crane": "gray crowned crane",
    # Extend as needed for more name variations
}


def smart_species_match(label_value: str, session: Session) -> int:
    """
    Attempts to resolve a species label to a species_id in species_flattened.

    Matching Logic:
    1. Normalize label (strip, lowercase, apply alias corrections)
    2. Try exact match (case-insensitive)
    3. Fallback to fuzzy ILIKE partial match

    Args:
        label_value (str): Predicted or user-provided species name
        session (Session): SQLAlchemy session for DB lookup

    Returns:
        int: species_id if found, otherwise -1
    """
    if not label_value:
        print("Empty label_value")
        return -1

    normalized = label_value.strip().lower()
    normalized = ALIASES.get(normalized, normalized)

    # Exact match by lowercased common name
    result = session.query(SpeciesFlattened).filter(
        func.lower(SpeciesFlattened.common_name) == normalized
    ).first()

    if result:
        print(f"Exact match: '{normalized}' → species_id = {result.species_id}")
        return result.species_id

    # Fuzzy partial match using ILIKE
    result = session.query(SpeciesFlattened).filter(
        SpeciesFlattened.common_name.ilike(f"%{normalized}%")
    ).first()

    if result:
        print(f"Fuzzy match: '{normalized}' → species_id = {result.species_id}")
        return result.species_id

    print(f"❌ No match for '{normalized}'")
    return -1
