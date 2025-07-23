# src/data/schemas.py
from pydantic import BaseModel, EmailStr # EmailStr para validación de email como username
from typing import Optional, List
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel): # Para decodificar el contenido del token
    username: Optional[str] = None

# --- Analysis History Schemas (para el futuro endpoint de historial) ---
class AnalysisHistoryBase(BaseModel):
    timestamp: datetime
    image_filename: Optional[str] = None
    risk_level_model: Optional[str] = None
    diagnosis_probable_model: Optional[str] = None
    probability_model: Optional[float] = None
    llm_recommendation: Optional[str] = None
    llm_explanation: Optional[str] = None
    llm_warning: Optional[str] = None
    details_secondary_model_json: Optional[str] = None

class AnalysisHistorySchema(AnalysisHistoryBase):
    id: int
    image_url: Optional[str] = None

    class Config:
        from_attributes = True
