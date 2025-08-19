"""
species_model.py — Species Metadata & Embedding ORM Models
-----------------------------------------------------------

This module defines SQLAlchemy ORM models for managing species metadata,
taxonomy, and vector embeddings within the Wildlife Vision System.

Tables:
- `wildlife.species`: Core species metadata (scientific, common names, conservation info)
- `wildlife.species_flattened`: Fully expanded taxonomic hierarchy (for search and display)
- `wildlife.species_embedding`: Precomputed CLIP and text embeddings for semantic search

Features:
- Species-level embeddings with optional image and API source tracking
- Taxonomy flattened for simplified filtering and ecoregion use
- Extinction flag and conservation status indicators

Dependencies:
- SQLAlchemy ORM
- pgvector extension for vector fields

"""

from sqlalchemy import Column, Integer, Text, Boolean, TIMESTAMP, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Species(Base):
    """
    Table: wildlife.species

    Stores core species records including:
    - Scientific and common names
    - Conservation status
    - Habitat notes
    """
    __tablename__ = 'species'
    __table_args__ = {'schema': 'wildlife'}

    species_id = Column(Integer, primary_key=True)
    scientific_name = Column(Text, nullable=False)
    common_name = Column(Text)
    category = Column(Text)  # e.g., 'mammals', 'birds'
    conservation_status = Column(Text)
    habitat = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class SpeciesFlattened(Base):
    """
    Table: wildlife.species_flattened

    Denormalized species hierarchy for simplified filtering, joins, and display:
    - Taxonomic ranks from kingdom to subspecies
    - Scientific and common names
    - Conservation details
    - Extinction status flag
    """
    __tablename__ = 'species_flattened'
    __table_args__ = {'schema': 'wildlife'}

    id = Column(Integer)
    species_id = Column(Integer, primary_key=True)
    kingdom = Column(Text)
    phylum = Column(Text)
    class_name = Column("class", Text)  # Avoids SQL reserved keyword conflict
    order_name = Column(Text)
    family = Column(Text)
    genus = Column(Text)
    species = Column(Text)
    subspecies = Column(Text)
    common_name = Column(Text)
    conservation_code = Column(Text)
    conservation_status = Column(Text)
    scientific_name = Column(Text)
    extinct = Column(Boolean)


class SpeciesEmbedding(Base):
    """
    Table: wildlife.species_embedding

    Stores species-level embeddings and source information:
    - CLIP image and text embeddings
    - Combined hybrid embedding
    - Associated species image path and API URL (if applicable)
    - Taxonomic and conservation context
    """
    __tablename__ = "species_embedding"
    __table_args__ = {"schema": "wildlife"}

    id = Column(Integer, primary_key=True)
    category = Column(Text)  # e.g., 'mammals', 'birds'
    species = Column(Text)
    image_path = Column(Text)
    api_url = Column(Text)
    image_description = Column(Text)
    image_embedding = Column(Vector(1024))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    status = Column(Text)  # e.g., 'complete', 'pending', etc.
    text_embedding = Column(Vector(1024))
    combined_embedding = Column(Vector(1024))
    common_name = Column(Text)
    scientific_name_clean = Column(Text)
    common_name_clean = Column(Text)
    conservation_code = Column(Text)
    conservation_status = Column(Text)
    extinct = Column(Boolean)


class SpeciesColorProfile(Base):
    __tablename__ = "species_color_profile"
    __table_args__ = {"schema": "wildlife"}

    common_name = Column(Text, primary_key=True)

    color_0 = Column(Text)
    color_0_pct = Column(Float)

    color_1 = Column(Text)
    color_1_pct = Column(Float)

    color_2 = Column(Text)
    color_2_pct = Column(Float)

    colors = Column(JSON)