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
MODEL1_MELANOMA_PATH = os.path.join(PROJECT_ROOT_DIR, 'models', 'model1_efficientnetb0.tflite')
MODEL2_MULTICLASS_PATH = os.path.join(PROJECT_ROOT_DIR, 'models', 'model2_multiclass.tflite')
# -----------------------------------------------

IMG_SIZE = 224
MELANOMA_THRESHOLD = 0.37
# -----------------------------------------------------------------

# --- Configuración de APIs (Gemini y Cloudinary) ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
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

# --- Carga de Intérpretes TFLite y Variables Globales ---
interpreter1_melanoma = None
interpreter2_multiclass = None
input_details1, output_details1 = None, None
input_details2, output_details2 = None, None


@app.on_event("startup")
async def load_models_on_startup():
    global interpreter1_melanoma, interpreter2_multiclass
    global input_details1, output_details1, input_details2, output_details2
    
    print(f"Cargando Modelo 1 (ENB0 TFLite) desde {MODEL1_MELANOMA_PATH}...")
    if os.path.exists(MODEL1_MELANOMA_PATH):
        try:
            interpreter1_melanoma = tf.lite.Interpreter(model_path=MODEL1_MELANOMA_PATH)
            interpreter1_melanoma.allocate_tensors()
            input_details1 = interpreter1_melanoma.get_input_details()
            output_details1 = interpreter1_melanoma.get_output_details()
            print("Intérprete TFLite Modelo 1 cargado.")
        except Exception as e:
            print(f"ERROR al cargar intérprete TFLite Modelo 1: {e}")
    else:
        print(f"ERROR: Archivo Modelo 1 no encontrado.")

    print(f"Cargando Modelo 2 (ENB0 TFLite) desde {MODEL2_MULTICLASS_PATH}...")
    if os.path.exists(MODEL2_MULTICLASS_PATH):
        try:
            interpreter2_multiclass = tf.lite.Interpreter(model_path=MODEL2_MULTICLASS_PATH)
            interpreter2_multiclass.allocate_tensors()
            input_details2 = interpreter2_multiclass.get_input_details()
            output_details2 = interpreter2_multiclass.get_output_details()
            print("Intérprete TFLite Modelo 2 cargado.")
        except Exception as e:
            print(f"ERROR al cargar intérprete TFLite Modelo 2: {e}")
    else:
        print(f"ERROR: Archivo Modelo 2 no encontrado.")
# -----------------------------------------------------------------


# --- Funciones de Preprocesamiento ---
def preprocess_image_for_enb0(image_bytes):
    """Preprocesa una imagen para el modelo EfficientNetB0."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE)) # Resize simple
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32)

        # Usar el preprocesamiento específico de EfficientNet
        processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return processed_img
    except Exception as e:
        print(f"Error preprocesando imagen para EfficientNetB0: {e}")
        return None
# ----------------------------------------------------------------


async def get_llm_validation_and_recommendation(
    image_bytes: bytes,
    diagnosis: str,
    risk: str,
    probability: float
) -> dict:
    """
    Usa Gemini Vision en una sola llamada para validar si la imagen es de piel
    y para generar la recomendación y explicación (basada en características visuales) si es válida.
    Devuelve un diccionario con los resultados.
    """
    global gemini_model
    if gemini_model is None:
        # Fallback si Gemini no está disponible
        return {
            "is_valid_image": True, # Ser permisivo
            "recommendation": "Recomendación no disponible (Error en LLM).",
            "explanation": "Explicación no disponible.",
            "warning": "Esta aplicación es una herramienta de ayuda y no reemplaza un diagnóstico médico profesional."
        }
    
    # --- PROMPT MULTI-TAREA ---
    prompt = f"""
    Actúa como un asistente virtual experto dermatológico de la aplicación Skincheck. Te proporcionaré una imagen y un diagnóstico preliminar. Tu tarea tiene dos partes y debes devolver un JSON.

    PARTE 1 (Validación): Primero, determina si la imagen es una fotografía de cerca de la piel humana mostrando una lesión cutánea (lunar, peca, mancha).

    PARTE 2 (Recomendación y Explicación Visual): Si y solo si la imagen es válida, usa el diagnóstico preliminar de "{diagnosis}" (con un nivel de riesgo "{risk}" y una confianza del modelo {probability*100:.1f}%) para generar lo siguiente:
    1.  Una recomendación clara y concisa para el usuario en un tono empático y responsable (1-2 frases). 
Si el riesgo es Alto o Medio, la recomendación principal debe ser consultar a un dermatólogo.
Si es Bajo, sugiere vigilancia y consultar si hay cambios, también recomienda establecer un recordatorio en la aplicación para volver a realizar un análisis (aclara que sean 6 meses si posee antecedentes familiares de melanoma).
    2.  Una explicación del diagnóstico en términos sencillos (4-6 frases). Para esto, observa la imagen y, basándote en tu conocimiento visual, describe qué características de esta lesión específica (ej. simetría/asimetría, regularidad de los bordes, uniformidad del color, etc.) podrían haber llevado a esta conclusión. Sé educativo y descriptivo sin ser alarmista. Si el diagnóstico es melanoma o carcinoma basocelular, incluye una breve definición de lo que es.
    3.  Una advertencia final estándar de que la aplicación no es un diagnóstico profesional.

    INSTRUCCIONES DE SALIDA:
    Responde ÚNICAMENTE con un objeto JSON.
    - Si la imagen NO es válida, el JSON debe ser: {{"is_valid_image": false, "rejection_reason": "La imagen no parece ser de una lesión cutánea."}}
    - Si la imagen ES válida, el JSON debe ser: {{"is_valid_image": true, "recommendation": "[Tu recomendación aquí]", "explanation": "[Tu explicación aquí]", "warning": "[Tu advertencia aquí]"}}
    """

    try:
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        print("Generando validación y recomendación con Gemini...")
        
        response = await gemini_model.generate_content_async([prompt, image_part])
        
        # Limpiar y parsear la respuesta JSON
        # A veces Gemini envuelve el JSON en ```json ... ```, hay que limpiarlo.
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        print(f"Respuesta cruda de Gemini (limpia): {cleaned_response_text}")
        
        response_json = json.loads(cleaned_response_text)
        return response_json

    except Exception as e:
        print(f"Error llamando o parseando respuesta de Gemini API: {e}")
        # Devolver un error genérico o un fallback
        return {
            "is_valid_image": True, # Ser permisivo en caso de error de LLM
            "recommendation": "No se pudo generar una recomendación en este momento.",
            "explanation": "No disponible.",
            "warning": "Consulta a un dermatólogo si tienes preocupaciones."
        }
# ------------------------------------------------------------------

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
    # ... (Verificación de intérpretes cargados) ...
    if interpreter1_melanoma is None: raise HTTPException(status_code=503, detail="Intérprete Modelo 1 no disponible.")
    if interpreter2_multiclass is None: raise HTTPException(status_code=503, detail="Intérprete Modelo 2 no disponible.")

    try:
        img_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")
    finally:
        await file.close()

    # --- PASO 1 y 2: Preprocesamiento y Predicciones de la Cascada (ANTES del LLM) ---
    processed_image = preprocess_image_for_enb0(img_bytes)
    if processed_image is None:
        raise HTTPException(status_code=400, detail="Error al procesar la imagen.")
    
    # --- Predicción Modelo 1 (ENB0 TFLite) ---
    try:
        interpreter1_melanoma.set_tensor(input_details1[0]['index'], processed_image)
        interpreter1_melanoma.invoke()
        prediction_m1 = interpreter1_melanoma.get_tensor(output_details1[0]['index'])
        melanoma_probability = float(prediction_m1[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción M1 (TFLite): {e}")

    # Lógica de la cascada...
    if melanoma_probability >= MELANOMA_THRESHOLD:
        risk_level = "Alto";
        diagnosis_probable = "Posible Melanoma";
        probability = melanoma_probability
        details_m2 = {"model_used": "Melanoma Binary Classifier (Modelo 1 - ENB0 TFLite)"}
    else:
        print("Riesgo Melanoma bajo, ejecutando Modelo 2...")
        try:
            interpreter2_multiclass.set_tensor(input_details2[0]['index'], processed_image)
            interpreter2_multiclass.invoke()
            prediction_m2 = interpreter2_multiclass.get_tensor(output_details2[0]['index'])
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
            
    # --- PASO 3: Llamada ÚNICA a Gemini para validación y recomendación ---
    llm_response = await get_llm_validation_and_recommendation(
        image_bytes=img_bytes,
        diagnosis=diagnosis_probable,
        risk=risk_level,
        probability=probability
    )
    # -----------------------------------------------------------------------

    # --- Validar la respuesta del LLM ---
    if not llm_response.get("is_valid_image", False):
        raise HTTPException(status_code=400, detail=llm_response.get("rejection_reason", "La imagen fue rechazada por el análisis de IA."))
    
    # Extraer los textos del LLM
    llm_recommendation = llm_response.get("recommendation", "N/A")
    llm_explanation = llm_response.get("explanation", "N/A")
    llm_warning = llm_response.get("warning", "N/A")
    # ------------------------------------

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

    # --- GUARDAR RESULTADO EN BASE DE DATOS ---
    try:
        details_json_str = json.dumps(details_m2) if details_m2 else None
        db_analysis = models_db.AnalysisHistory(
            user_id=current_user.id,
            image_filename=original_filename,
            image_url=image_url_cloudinary,  # Guardar la URL de Cloudinary
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
    if db_analysis.image_url:
        try:
            # Extraer el public_id de la URL. Es todo lo que va después de la última '/'
            # y antes de la extensión '.jpg' o '.png'.
            public_id_with_folder = "/".join(db_analysis.image_url.split('/')[-2:])
            public_id = os.path.splitext(public_id_with_folder)[0]
            print(f"Intentando eliminar de Cloudinary con public_id: {public_id}")
            cloudinary.uploader.destroy(public_id)
            print("Imagen eliminada de Cloudinary.")
        except Exception as e:
            print(f"Error eliminando de Cloudinary (puede que ya no exista): {e}")

    # Borrar de la BD
    crud_users.delete_analysis_by_id(db=db, analysis_id=history_id)
    return

# --- Ruta Raíz y Ejecución (como lo tenía) ---
@app.get("/")
async def read_root():
    return {"message": "Bienvenido a la API Skincheck (Cascada + LLM)"}

if __name__ == "__main__":
    print("Ejecutando servidor Uvicorn...")
    uvicorn.run("main_render:app", host="127.0.0.1", port=8000, reload=True)
