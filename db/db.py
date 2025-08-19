"""
db.py — Database Engine & Session Setup
----------------------------------------

This module initializes the SQLAlchemy database connection and provides
session management for the Wildlife Vision System.

Features:
- Creates database engine using DATABASE_URL from project settings
- Defines `SessionLocal` for transaction management
- Provides `init_db()` to create tables based on ORM models

Intended for:
- Centralized database connection setup
- Reusable session handling across all modules

Dependencies:
- SQLAlchemy for ORM and engine management
- Project settings for environment-based configuration

"""

from sqlalchemy.orm import declarative_base
from config.settings import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# --- ORM Base Class ---
Base = declarative_base()

# --- Database Engine ---

engine = create_engine(
    DATABASE_URL
)


# --- Session Factory ---
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """
    Initializes the database by creating all tables defined in ORM models.
    Safe to run multiple times; only creates tables if they don't exist.
    """
    Base.metadata.create_all(engine)


