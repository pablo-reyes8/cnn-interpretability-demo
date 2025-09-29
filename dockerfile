# Dockerfile
FROM python:3.11.13-slim

# Paquetes del sistema (Pillow, compilación básica para sklearn/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /proyecto_final

# Evitar .pyc y forzar stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PYTHONPATH="/proyecto_final:/proyecto_final/src"


# Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Exponemos puertos típicos
EXPOSE 8000 8501

# Comando por defecto: FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]