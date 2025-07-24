# Usar una imagen base oficial de Python.
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos primero para aprovechar el caché de Docker.
COPY src/requirements.txt requirements.txt

# Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODO el contenido del directorio actual (la raíz de tu repo local) a /app
# Esto preservará la estructura de carpetas src/, models/, etc.
COPY . .

# Copiar los modelos de IA al contenedor.
# Copiar la carpeta 'models/' del proyecto a la carpeta '/app/models/' del contenedor
COPY ./models /app/models

# Exponer el puerto en el que Uvicorn correrá.
EXPOSE 10000

# Comando para ejecutar la aplicación cuando el contenedor se inicie.
CMD ["uvicorn", "src.main_render:app", "--host", "0.0.0.0", "--port", "10000"]
