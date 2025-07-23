# src/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# --- CONFIGURACIÓN DE BASE DE DATOS ADAPTABLE ---

DATABASE_URL_PROD = os.getenv("DATABASE_URL")

# Para desarrollo local, se usa SQLite.
# Construir la ruta al archivo .db en la raíz del proyecto.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio de este archivo (src/)
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR) # Raíz del proyecto (Project_Skincheck/)
SQLALCHEMY_DATABASE_URL_LOCAL = f"sqlite:///{os.path.join(PROJECT_ROOT_DIR, 'skincheck_local.db')}"

# Decidir qué URL de base de datos usar
if DATABASE_URL_PROD:
    print("Variable de entorno DATABASE_URL encontrada. Usando base de datos de producción (PostgreSQL).")
    if DATABASE_URL_PROD.startswith("postgres://"):
        DATABASE_URL_PROD = DATABASE_URL_PROD.replace("postgres://", "postgresql://", 1)
    
    SQLALCHEMY_DATABASE_URL = DATABASE_URL_PROD
else:
    print("Variable de entorno DATABASE_URL NO encontrada. Usando base de datos local (SQLite).")
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL_LOCAL

# --- CREACIÓN DEL MOTOR DE SQLAlchemy ---
connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Crear el motor con la URL y los argumentos correctos
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Función para obtener una sesión de BD (para dependencias de FastAPI)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
