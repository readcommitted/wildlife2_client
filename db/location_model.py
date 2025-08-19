"""
location_model.py — Location Lookup Table (ORM Model)
------------------------------------------------------

This module defines the SQLAlchemy ORM model for managing known location lookups
used throughout the Wildlife Vision System.

Table:
- `wildlife.location_lookup`

Purpose:
- Provides standardized location descriptions
- Links detailed place, park, state, county, and country hierarchy
- Supplies latitude/longitude for precise geospatial enrichment

Used for:
- Smart search dropdowns during image validation
- Automated geocoding fallback for images missing coordinates
- Consistent tagging of places, parks, and regions

Dependencies:
- SQLAlchemy ORM

"""

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class LocationLookup(Base):
    """
    Table: wildlife.location_lookup

    Stores known locations with structured place hierarchy and geospatial coordinates.

    Fields:
    - location_description: User-friendly display name (unique)
    - place, park, state, county, country: Location hierarchy details
    - latitude, longitude: Decimal GPS coordinates
    """
    __tablename__ = "location_lookup"
    __table_args__ = {"schema": "wildlife"}

    location_lookup_id = Column(Integer, primary_key=True, autoincrement=True)
    location_description = Column(String, unique=True, nullable=False)
    place = Column(String)
    park = Column(String)
    state = Column(String)
    county = Column(String)
    country = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
