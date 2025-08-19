# db/training_model.py
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, MetaData
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# All tables in the 'wildlife' schema by default
metadata = MetaData(schema="wildlife")
Base = declarative_base(metadata=metadata)


class ModelRun(Base):
    __tablename__ = "model_run"

    model_run_id   = Column(Integer, primary_key=True, autoincrement=True)
    model_name     = Column(String(50), nullable=False, default="speciesnet")
    model_version  = Column(String(50), nullable=False, default="resnet18")
    tag            = Column(String(120))
    epochs         = Column(Integer)
    lr             = Column(Float)
    batch_size     = Column(Integer)
    num_classes    = Column(Integer)
    num_train      = Column(Integer)
    num_val        = Column(Integer)
    top1_accuracy  = Column(Float)  # 0..1
    top5_accuracy  = Column(Float)  # 0..1
    confusion_matrix      = Column(JSONB)   # list[list[int]]
    classification_report = Column(JSONB)   # dict from sklearn
    model_path     = Column(Text)
    started_at     = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at    = Column(DateTime(timezone=True))

    results = relationship(
        "ModelResult",
        back_populates="run",
        cascade="all, delete-orphan"
        # lazy="selectin"  # optional: faster eager loading of child rows
    )


class ModelResult(Base):
    __tablename__ = "model_result"

    model_result_id = Column(Integer, primary_key=True, autoincrement=True)
    # FK resolves to wildlife.model_run because Base.metadata has schema="wildlife"
    model_run_id    = Column(
        Integer,
        ForeignKey("model_run.model_run_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    jpeg_path       = Column(Text, nullable=False)
    true_label      = Column(String(120), nullable=False)
    predicted_label = Column(String(120), nullable=False)
    correct         = Column(Boolean, nullable=False)
    top5            = Column(JSONB)  # list[[label, prob], ...]
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    run = relationship("ModelRun", back_populates="results")
