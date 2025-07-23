# src/data/models_db.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analyses = relationship("AnalysisHistory", back_populates="owner")

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    image_filename = Column(String)
    image_url = Column(String, nullable=True)
    risk_level_model = Column(String)
    diagnosis_probable_model = Column(String)
    probability_model = Column(Float)
    llm_recommendation = Column(Text)
    llm_explanation = Column(Text)
    llm_warning = Column(Text)
    details_secondary_model_json = Column(Text, nullable=True)
    
    owner = relationship("User", back_populates="analyses")
