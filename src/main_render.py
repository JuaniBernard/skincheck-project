import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
from PIL import Image
import datetime
import io
import uvicorn
import google.generativeai as genai
import uuid
import json
import cloudinary
import cloudinary.uploader
from jose import JWTError, jwt

from . import database, security, crud_users
from .data import models_db, schemas

# --- Instancia de OAuth2PasswordBearer ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# --- FUNCIÓN DE DEPENDENCIA PARA OBTENER USUARIO ACTUAL ---
async def get_current_user(db: Session = Depends(database.get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = crud_users.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
# -------------------------------------------------------------

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)

# --- Rutas a modelos ---
MODEL1_MELANOMA_PATH = os.path.join(PROJECT_ROOT_DIR, 'models', 'melanoma_efficientnetb0_v1.h5')
MODEL2_MULTICLASS_PATH = os.path.join(PROJECT_ROOT_DIR, 'models', 'model2_best_overall.keras')
# -----------------------------------------------

IMG_SIZE = 224
MELANOMA_THRESHOLD = 0.37
# -----------------------------------------------------------------

# --- Configuración de APIs (Gemini y Cloudinary) ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    print("Modelo Gemini configurado.")
except Exception as e:
    print(f"ERROR configurando Gemini: {e}")
    gemini_model = None

try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    print("Cloudinary configurado exitosamente.")
except Exception as e:
    print(f"ADVERTENCIA: Credenciales de Cloudinary no encontradas. La subida de imágenes fallará.")
# ------------------------------------

# --- Crear tablas en la BD ---
try:
    print("Verificando/Creando tablas en la base de datos...")
    models_db.Base.metadata.create_all(bind=database.engine)
    print("Tablas verificadas/creadas.")
except Exception as e:
    print(f"Error creando tablas: {e}")
# ------------------------------------

app = FastAPI(title="Skin Lesion Risk Assessment API (Cascaded + LLM)")

# --- Carga de Modelos y Variables Globales ---
model1_melanoma = None
model2_multiclass = None


@app.on_event("startup")
async def load_models_on_startup():
    global model1_melanoma, model2_multiclass
    print(f"Cargando Modelo 1 (VGG19) desde {MODEL1_MELANOMA_PATH}...")
    if os.path.exists(MODEL1_MELANOMA_PATH):
        model1_melanoma = tf.keras.models.load_model(MODEL1_MELANOMA_PATH)
        print("Modelo 1 cargado.")
    else:
        print(f"ERROR: Archivo Modelo 1 no encontrado.")

    print(f"Cargando Modelo 2 (VGG16) desde {MODEL2_MULTICLASS_PATH}...")
    if os.path.exists(MODEL2_MULTICLASS_PATH):
        model2_multiclass = tf.keras.models.load_model(MODEL2_MULTICLASS_PATH)
        print("Modelo 2 cargado.")
    else:
        print(f"ERROR: Archivo Modelo 2 no encontrado.")


# --- Funciones de Preprocesamiento ---
def preprocess_image_for_model(image_bytes, model_type: str):
    """Preprocesa una imagen para un tipo de modelo específico (VGG16, VGG19)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img.thumbnail((IMG_SIZE, IMG_SIZE))
        background = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        offset = ((IMG_SIZE - img.width) // 2, (IMG_SIZE - img.height) // 2)
        background.paste(img, offset)
        img_array = tf.keras.preprocessing.image.img_to_array(background)
        img_array = np.expand_dims(img_array, axis=0)

        if model_type == 'vgg19':
            return tf.keras.applications.vgg19.preprocess_input(img_array)
        elif model_type == 'vgg16':
            return tf.keras.applications.vgg16.preprocess_input(img_array)
        else:
            # Fallback o error si el tipo no es soportado
            raise ValueError(f"Tipo de modelo no soportado para preprocesamiento: {model_type}")

    except Exception as e:
        print(f"Error preprocesando imagen para {model_type}: {e}")
        return None


# --- Funciones de LLM (is_skin_image_llm, get_llm_recommendation) ---
async def is_skin_image_llm(image_bytes: bytes) -> bool:
    """
    Usa Gemini para determinar si una imagen es de piel/lesión cutánea.
    """
    global gemini_model
    if gemini_model is None: return True  # Fallback

    try:
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        prompt = "Analiza esta imagen. ¿Es una fotografía de cerca de la piel humana, mostrando un lunar, peca o lesión cutánea? Responde únicamente con 'Sí' o 'No'."

        print("Verificando tipo de imagen con Gemini...")
        response = await gemini_model.generate_content_async([prompt, image_part])
        response_text = response.text.strip().lower()
        print(f"Respuesta de Gemini (filtro): '{response_text}'")
        return "sí" in response_text or "si" in response_text or "yes" in response_text
    except Exception as e:
        print(f"Error llamando a Gemini API para filtro: {e}");
        return True


# --- Función para llamar a Gemini ---
async def get_llm_recommendation(diagnosis: str, risk: str, probability: float):
    global gemini_model
    if gemini_model is None:
        return "Recomendación no disponible (Error en LLM).", "Explicación no disponible."

    prompt = f"""
    Eres un asistente virtual de una aplicación de chequeo de piel llamada Skincheck.
    Un análisis de imagen ha producido los siguientes resultados para un lunar del usuario:
    - Diagnóstico Más Probable: "{diagnosis}"
    - Nivel de Riesgo Estimado por el sistema: "{risk}"
    - Confianza del Modelo en este Diagnóstico: {probability*100:.1f}%

    Basado en esta información:
    1.  Genera una recomendación clara y concisa para el usuario en un tono empático y responsable (1-2 frases).
        Si el riesgo es Alto o Medio, la recomendación principal debe ser consultar a un dermatólogo.
        Si el riesgo es Bajo, sugiere vigilancia y consultar si hay cambios, también recomienda establecer un recordatorio en la aplicación para volver a realizar un análisis (aclarar que sean seis meses si tiene antecedentes familiares de melanoma, o un año si no los tiene).
    2.  Proporciona una breve explicación de lo que significa el "Diagnóstico Más Probable" en términos sencillos para un paciente (2-3 frases).
        No uses jerga médica excesiva. Si el diagnóstico es melanoma o carcinoma basocelular, agrega una frase que defina dicho cáncer y posibles características de la lesión o lunar.
    3.  Siempre finaliza recordando al usuario que esta aplicación es una herramienta de ayuda y no reemplaza una consulta ni un diagnóstico médico profesional.

    Formatea tu respuesta estrictamente así:
    RECOMENDACION: [Tu recomendación aquí]
    EXPLICACION: [Tu explicación aquí]
    ADVERTENCIA: [Tu advertencia aquí]
    """
    try:
        print("Generando respuesta con Gemini...")
        response = await gemini_model.generate_content_async(prompt) # Usar async
        # Extraer el texto y parsear
        text_response = response.text
        print(f"Respuesta cruda de Gemini: {text_response}") # Para depuración

        recommendation = "Error al generar recomendación."
        explanation = "Error al generar explicación."
        warning = "Esta aplicación es una herramienta de ayuda y no reemplaza un diagnóstico médico profesional. Consulta siempre a un dermatólogo." # Fallback

        rec_marker = "RECOMENDACION:"
        exp_marker = "EXPLICACION:"
        war_marker = "ADVERTENCIA:"

        if rec_marker in text_response:
            rec_start = text_response.find(rec_marker) + len(rec_marker)
            exp_start_for_rec = text_response.find(exp_marker, rec_start)
            recommendation = text_response[rec_start:exp_start_for_rec if exp_start_for_rec != -1 else None].strip()

        if exp_marker in text_response:
            exp_start = text_response.find(exp_marker) + len(exp_marker)
            war_start_for_exp = text_response.find(war_marker, exp_start)
            explanation = text_response[exp_start:war_start_for_exp if war_start_for_exp != -1 else None].strip()

        if war_marker in text_response: # Si el LLM incluye la advertencia, usarla, si no, usar el fallback
            war_start = text_response.find(war_marker) + len(war_marker)
            custom_warning = text_response[war_start:].strip()
            if custom_warning: # Si no está vacía
                warning = custom_warning


        return recommendation, explanation, warning

    except Exception as e:
        print(f"Error llamando a Gemini API: {e}")
        return "No se pudo generar una recomendación en este momento.", "No disponible.", "Consulta a un dermatólogo si tienes preocupaciones."
# ---------------------------------------

# --- FUNCIÓN DE DEPENDENCIA PARA OBTENER USUARIO ACTUAL ---
async def get_current_user(db: Session = Depends(database.get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username: str = payload.get("sub")  # claim estándar para el sujeto (username)
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = crud_users.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
# -------------------------------------------------------------

# --- Endpoints de Autenticación (register, token, users/me) ---
# --- Endpoint de Registro ---
@app.post("/auth/register", response_model=schemas.User, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud_users.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    created_user = crud_users.create_user(db=db, user=user)
    # No devolver la contraseña hasheada, User schema ya lo maneja
    return created_user

# --- Endpoint de Login (Token) ---
@app.post("/auth/token", response_model=schemas.Token, tags=["Authentication"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)
):
    user = crud_users.get_user_by_username(db, username=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Endpoint para Obtener Información del Usuario Actual (Protegido) ---
@app.get("/users/me", response_model=schemas.User, tags=["Users"])
async def read_users_me(current_user: models_db.User = Depends(get_current_user)):
    return current_user

@app.post("/predict/cascade", summary="Evalúa riesgo de lesión cutánea y guarda el análisis", tags=["Analysis"])
async def predict_cascade(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db),
        current_user: models_db.User = Depends(get_current_user)
):
    global model1_melanoma, model2_multiclass
    if model1_melanoma is None: raise HTTPException(status_code=503, detail="Modelo 1 no disponible.")
    if model2_multiclass is None: raise HTTPException(status_code=503, detail="Modelo 2 no disponible.")

    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")
    finally:
        await file.close()

    # --- PASO 0: Filtro de Imagen con Gemini Vision (ANTES de subir/procesar) ---
    is_valid_image = await is_skin_image_llm(img_bytes)
    if not is_valid_image:
        raise HTTPException(status_code=400,
                            detail="La imagen no parece ser de piel. Por favor, sube una imagen de un lunar.")

    # --- SUBIR IMAGEN A CLOUDINARY ---
    image_url_cloudinary = None
    original_filename = file.filename if file.filename else "unknown_image.jpg"
    try:
        print("Subiendo imagen a Cloudinary...")
        upload_result = cloudinary.uploader.upload(img_bytes, folder="skincheck_analyses")
        image_url_cloudinary = upload_result.get("secure_url")
        print(f"Imagen subida. URL: {image_url_cloudinary}")
    except Exception as e:
        print(f"Error subiendo imagen a Cloudinary: {e}")
        # Continuar sin URL de imagen si la subida falla

    # --- Preprocesamiento y Predicciones de la Cascada ---
    processed_image_m1 = preprocess_image_for_model(img_bytes, model_type='vgg19')
    if processed_image_m1 is None: raise HTTPException(status_code=400,
                                                       detail="Error al procesar imagen para Modelo 1.")

    try:
        prediction_m1 = model1_melanoma.predict(processed_image_m1)
        melanoma_probability = float(prediction_m1[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción M1: {e}")

    # Lógica de la cascada...
    if melanoma_probability >= MELANOMA_THRESHOLD:
        risk_level = "Alto";
        diagnosis_probable = "Posible Melanoma";
        probability = melanoma_probability
        details_m2 = {"model_used": "Melanoma Binary Classifier (Modelo 1)"}
    else:
        print("Riesgo Melanoma bajo, ejecutando Modelo 2...")
        processed_image_m2 = preprocess_image_for_model(img_bytes, model_type='vgg16')
        if processed_image_m2 is None: raise HTTPException(status_code=400,
                                                           detail="Error al procesar imagen para Modelo 2.")

        try:
            prediction_m2 = model2_multiclass.predict(processed_image_m2)
            probabilities_m2 = prediction_m2[0]
            predicted_class_index = np.argmax(probabilities_m2)
            probability = float(probabilities_m2[predicted_class_index])
            class_map = {0: "Probablemente Benigno", 1: "Posible Carcinoma Basocelular (BCC)",
                         2: "Posible Queratosis Actínica (AKIEC)"}
            risk_map = {0: "Bajo", 1: "Medio", 2: "Medio"}
            diagnosis_probable = class_map.get(predicted_class_index)
            risk_level = risk_map.get(predicted_class_index)
            details_m2 = {"model_used": "Multiclass Classifier (Modelo 2)",
                          "probabilities": {"benign": float(probabilities_m2[0]), "bcc": float(probabilities_m2[1]),
                                            "akiec": float(probabilities_m2[2])}}
        except Exception as e:
            print(f"Error en predicción M2: {e}")
            risk_level = "Bajo";
            diagnosis_probable = "Lesión No Melanoma (análisis secundario falló)"
            probability = melanoma_probability
            details_m2 = {"model_used": "Modelo 1", "error_model2": str(e)}

    # --- LLAMAR A GEMINI PARA RECOMENDACIÓN ---
    llm_recommendation, llm_explanation, llm_warning = await get_llm_recommendation(diagnosis_probable, risk_level,
                                                                                    probability)

    # --- GUARDAR RESULTADO EN BASE DE DATOS ---
    try:
        details_json_str = json.dumps(details_m2) if details_m2 else None
        db_analysis = models_db.AnalysisHistory(
            user_id=current_user.id,
            image_filename=original_filename,
            image_path_local=image_url_cloudinary,  # Guardar la URL de Cloudinary
            risk_level_model=risk_level,
            diagnosis_probable_model=diagnosis_probable,
            probability_model=probability,
            llm_recommendation=llm_recommendation,
            llm_explanation=llm_explanation,
            llm_warning=llm_warning,
            details_secondary_model_json=details_json_str
        )
        db.add(db_analysis);
        db.commit();
        db.refresh(db_analysis)
        print(f"Análisis ID {db_analysis.id} guardado para usuario ID: {current_user.id}")
    except Exception as e:
        print(f"ERROR al guardar análisis en BD: {e}");
        db.rollback()

    # --- RESPUESTA FINAL AL FRONTEND ---
    return {
        "filename": original_filename,
        "content_type": file.content_type,
        "risk_level_model": risk_level,
        "diagnosis_probable_model": diagnosis_probable,
        "probability_model": probability,
        "llm_recommendation": llm_recommendation,
        "llm_explanation": llm_explanation,
        "llm_warning": llm_warning,
        "details_secondary_model": details_m2,
        "image_url": image_url_cloudinary
    }


# --- Endpoint de Historial ---
@app.get("/history", response_model=List[schemas.AnalysisHistorySchema], tags=["History"])
async def read_user_history(
        request: Request, db: Session = Depends(database.get_db),
        current_user: models_db.User = Depends(get_current_user),
        skip: int = 0, limit: int = 100
):
    history_items_db = crud_users.get_user_analysis_history(db=db, user_id=current_user.id, skip=skip, limit=limit)
    response_items = []
    for item_db in history_items_db:
        response_items.append(
            schemas.AnalysisHistorySchema.from_orm(item_db))
    return response_items


# --- Endpoint de Borrar Historial ---
@app.delete("/history/{history_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["History"])
async def delete_analysis(
        history_id: int, db: Session = Depends(database.get_db),
        current_user: models_db.User = Depends(get_current_user)
):
    db_analysis = crud_users.get_analysis_by_id(db=db, analysis_id=history_id)
    if db_analysis is None: raise HTTPException(status_code=404, detail="Análisis no encontrado")
    if db_analysis.user_id != current_user.id: raise HTTPException(status_code=403, detail="No autorizado")

    # Eliminar de Cloudinary si la URL existe
    if db_analysis.image_path_local:
        try:
            public_id = db_analysis.image_path_local.split('/')[-1].split('.')[0]
            print("Imagen (potencialmente) eliminada de Cloudinary. Implementar borrado robusto con public_id.")
        except Exception as e:
            print(f"Error eliminando de Cloudinary: {e}")

    crud_users.delete_analysis_by_id(db=db, analysis_id=history_id)
    return

# --- Ruta Raíz y Ejecución (como lo tenía) ---
@app.get("/")
async def read_root():
    return {"message": "Bienvenido a la API Skincheck (Cascada + LLM)"}

if __name__ == "__main__":
    print("Ejecutando servidor Uvicorn...")
    uvicorn.run("main_render:app", host="127.0.0.1", port=8000, reload=True)
