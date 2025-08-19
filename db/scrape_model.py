"""
scrape_model.py — Species Scraping Source Table (ORM Model)
------------------------------------------------------------

This module defines the SQLAlchemy ORM model for managing scraping source URLs
used to populate species data for the Wildlife Vision System.

Table:
- `scrape_source`

Purpose:
- Tracks URLs for species list scraping (e.g., Wikipedia)
- Categorizes sources by region, taxonomic group, and category
- Provides activity flag and timestamps for automation

Used for:
- Controlling which pages the scraping tools process
- Recording last-scraped timestamps
- Enabling modular species dataset expansion

Dependencies:
- SQLAlchemy ORM

"""

from sqlalchemy import (
    Column, Integer, Text, DateTime, Boolean, text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ScrapeSource(Base):
    """
    Table: scrape_source

    Stores metadata for species scraping sources, including:
    - Source URL
    - Species category (e.g., mammals, birds)
    - Region and taxonomic subgroup (optional)
    - Active flag to control scraping jobs
    - Timestamps for automation tracking
    """
    __tablename__ = "scrape_source"

    id = Column(Integer, primary_key=True, autoincrement=True)

    category = Column(Text, nullable=False)  # e.g., 'mammals', 'birds'
    url = Column(Text, nullable=False, unique=True)
    region = Column(Text)  # Optional, e.g., 'North America'
    taxonomic_group = Column(Text)  # Optional sub-group

    active = Column(Boolean, nullable=False, default=True)
    last_scraped_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), onupdate=text("now()"))
