# src/crud_users.py
from sqlalchemy.orm import Session
from .data import models_db, schemas
from . import security
from typing import List

def get_user_by_username(db: Session, username: str):
    return db.query(models_db.User).filter(models_db.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    db_user = models_db.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_analysis_history(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models_db.AnalysisHistory]:
    return db.query(models_db.AnalysisHistory)\
             .filter(models_db.AnalysisHistory.user_id == user_id)\
             .order_by(models_db.AnalysisHistory.timestamp.desc())\
             .offset(skip)\
             .limit(limit)\
             .all()

def get_analysis_by_id(db: Session, analysis_id: int):
    """Obtiene un único análisis por su ID."""
    return db.query(models_db.AnalysisHistory).filter(models_db.AnalysisHistory.id == analysis_id).first()

def delete_analysis_by_id(db: Session, analysis_id: int):
    """Elimina un único análisis por su ID."""
    db_analysis = get_analysis_by_id(db, analysis_id)
    if db_analysis:
        db.delete(db_analysis)
        db.commit()
    return db_analysis # Devuelve el objeto borrado para obtener info como la ruta de la imagen
