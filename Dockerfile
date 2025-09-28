# Multi-stage build para optimizar tamaño de imagen
FROM python:3.11-slim as base

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para OCR avanzado
RUN apt-get update && apt-get install -y \
    # Tesseract y lenguajes
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    tesseract-ocr-cat \
    tesseract-ocr-eus \
    tesseract-ocr-glg \
    # OpenCV y librerías gráficas
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libxss1 \
    # PDF processing
    poppler-utils \
    # OpenCV optimizado
    python3-opencv \
    # Utilidades
    curl \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Crear directorio de trabajo y subdirectorios necesarios
WORKDIR /app
RUN mkdir -p /app/tmp /app/cache /app/logs && chown -R appuser:appuser /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # Verificar instalación de pytesseract
    python -c "import pytesseract; print('Pytesseract OK')" && \
    # Limpiar pip cache
    rm -rf ~/.cache/pip

# Copiar código fuente y archivos de configuración
COPY --chown=appuser:appuser main.py .
# Si tienes archivo .env, copiarlo (opcional)
# COPY --chown=appuser:appuser .env .

# Establecer permisos finales
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/cache /app/logs /app/tmp

# Variables de entorno por defecto (pueden ser sobrescritas)
ENV TESSERACT_CMD=/usr/bin/tesseract \
    TESSERACT_LANGUAGES=spa+eng+cat \
    MAX_FILE_SIZE=10485760 \
    MAX_PAGES_PDF=5 \
    ENABLE_CACHE=true \
    CACHE_DIR=/app/cache \
    ALLOWED_ORIGINS=*

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8001

# Health check optimizado
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Comando de inicio con configuración para producción
# Nota: Ajustar workers según recursos disponibles (1 worker = ~100MB RAM)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2", "--loop", "uvloop", "--access-log"]