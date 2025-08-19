from sqlalchemy import Column, Integer, String, Float
from db.db import Base  # or declarative_base()

class ColorPalette(Base):
    __tablename__ = "color_palette"
    __table_args__ = {"schema": "wildlife"}

    id = Column(Integer, primary_key=True)
    color_name = Column(String, nullable=False)
    min_r = Column(Integer)
    max_r = Column(Integer)
    min_g = Column(Integer)
    max_g = Column(Integer)
    min_b = Column(Integer)
    max_b = Column(Integer)
    min_avg = Column(Float)
    max_avg = Column(Float)
    notes = Column(String)
