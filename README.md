# 🔍 Cakely OCR Service

Servicio avanzado de OCR (Reconocimiento Óptico de Caracteres) especializado en el procesamiento de facturas, tickets y documentos empresariales españoles. Construido con FastAPI y Tesseract OCR.

## 📋 Características

- ✅ **OCR Avanzado** con múltiples calidades de procesamiento
- ✅ **Especialización en Facturas Eléctricas** (Naturgy, Endesa, Iberdrola, EDP)
- ✅ **Procesamiento de PDFs e Imágenes**
- ✅ **Extracción de Datos Estructurados** (importes, fechas, CIF/NIF, etc.)
- ✅ **API RESTful Completa** con documentación automática
- ✅ **Cache Inteligente** para mejor rendimiento
- ✅ **Validación de Documentos** según normativa española
- ✅ **Procesamiento por Lotes** hasta 10 archivos
- ✅ **Endpoints de Debug** para diagnóstico

## 🚀 Inicio Rápido

### Usando Docker

```bash
# Clonar repositorio
git clone https://github.com/sebasblancogonz/ocr.git
cd ocr

# Construir imagen
docker build -t ocr-service .

# Ejecutar contenedor
docker run -p 8001:8001 ocr-service
```

### Variables de Entorno

```env
# Configuración básica
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# CORS y Seguridad
ALLOWED_ORIGINS=*
MAX_FILE_SIZE=50000000  # 50MB

# OCR Settings
TESSERACT_CMD=/usr/bin/tesseract
TESSERACT_LANGUAGES=spa+eng
MAX_PAGES_PDF=5

# Cache (opcional)
ENABLE_CACHE=true
CACHE_DIR=/tmp/ocr_cache
```

## 📚 API Endpoints

### 🏠 Información General

#### `GET /`

Información básica del servicio y endpoints disponibles.

#### `GET /health`

Health check completo con información del sistema.

**Respuesta:**

```json
{
  "status": "healthy",
  "service": "OCR Service",
  "version": "2.0.0",
  "tesseract": {
    "available": true,
    "version": "tesseract 4.1.1",
    "languages": ["spa", "eng", "cat"]
  },
  "limits": {
    "max_file_size": 50000000,
    "max_pdf_pages": 5
  }
}
```

### 🔍 Procesamiento OCR

#### `POST /ocr/process`

Endpoint principal para procesamiento de documentos.

**Parámetros:**

- `file` (required): Archivo a procesar (PDF, JPG, PNG, TIFF, BMP)
- `quality` (optional): Calidad de procesamiento
  - `fast`: Rápido, menor precisión
  - `balanced`: Balance velocidad/precisión (default)
  - `high`: Máxima precisión, más lento
- `extract_items` (optional): Extraer items de línea en facturas (default: false)
- `use_cache` (optional): Usar cache si disponible (default: true)

**Ejemplo:**

```bash
curl -X POST "http://localhost:8001/ocr/process?quality=high" \
  -F "file=@factura.pdf"
```

**Respuesta:**

```json
{
  "success": true,
  "filename": "factura.pdf",
  "content_type": "application/pdf",
  "document_type": "electricity_bill",
  "raw_text": "NATURGY\nFactura Eléctrica...",
  "extracted_data": {
    "document_type": "electricity_bill",
    "supplier": "naturgy",
    "billing_info": {
      "invoice_number": "FE24321476789129",
      "period_start": "2024-01-01",
      "period_end": "2024-01-31"
    },
    "amounts": {
      "total": 89.45,
      "electricity": 65.3,
      "gas": 15.2,
      "vat": 8.95
    },
    "supply_info": {
      "cups": "ES0031408096805001JN0F"
    }
  },
  "metrics": {
    "confidence": 92.5,
    "quality_score": 85.0,
    "processing_time": 2.345,
    "text_length": 1250
  }
}
```

#### `POST /ocr/batch`

Procesamiento por lotes (máximo 10 archivos).

**Parámetros:**

- `files` (required): Array de archivos
- `quality` (optional): Calidad de procesamiento

#### `POST /ocr/enhance`

Procesamiento con múltiples técnicas de mejoramiento para imágenes de baja calidad.

**Parámetros:**

- `file` (required): Imagen a procesar
- `try_all_methods` (optional): Probar todos los métodos (default: true)

**Respuesta:**

```json
{
  "filename": "imagen_borrosa.jpg",
  "methods_tested": ["original", "clahe", "bilateral_filter", "gaussian_blur"],
  "best_result": {
    "method": "clahe",
    "psm": 6,
    "text": "Texto mejorado y legible",
    "score": 85.5
  },
  "recommendations": {
    "low_quality_detected": true,
    "suggested_method": "clahe"
  }
}
```

### ⚡ Especializado en Facturas

#### `POST /ocr/process-utility-bill`

Endpoint especializado para facturas de servicios públicos.

**Parámetros:**

- `file` (required): Factura eléctrica/gas
- `quality` (optional): Recomendado `high`
- `extract_consumption_data` (optional): Extraer datos de consumo (default: true)

**Respuesta:**

```json
{
  "success": true,
  "document_type": "electricity_bill",
  "supplier": "naturgy",
  "validation": {
    "is_valid": true,
    "quality_score": 92.5
  },
  "billing_information": {
    "invoice_number": "FE24321476789129",
    "period_start": "2024-01-01",
    "period_end": "2024-01-31",
    "cups": "ES0031408096805001JN0F"
  },
  "amounts": {
    "total": 89.45,
    "electricity": 65.3,
    "gas": 15.2,
    "vat": 8.95
  },
  "consumption": {
    "kwh": 250,
    "contracted_power_kw": 4.4
  }
}
```

### 🔧 Validación y Análisis

#### `POST /ocr/validate`

Validar si un documento cumple requisitos mínimos para facturación española.

**Respuesta:**

```json
{
  "valid": true,
  "document_type": "invoice",
  "completeness_percentage": 85.0,
  "validation_results": {
    "has_tax_id": true,
    "has_invoice_number": true,
    "has_total": true,
    "has_supplier": true
  },
  "missing_required": [],
  "missing_optional": ["has_iban", "has_items"]
}
```

#### `POST /ocr/analyze`

Análisis rápido sin OCR completo.

**Respuesta:**

```json
{
  "filename": "documento.pdf",
  "type": "pdf",
  "size_mb": 2.5,
  "num_pages": 3,
  "has_text_layer": false,
  "probable_document_type": "electricity_bill",
  "preview": {
    "text": "Vista previa del contenido...",
    "confidence": 78.5
  }
}
```

### 🐛 Debug y Diagnóstico

#### `GET /ocr/test-tesseract`

Verificar que Tesseract funciona correctamente.

#### `POST /ocr/debug`

Debug completo para imágenes problemáticas.

#### `POST /ocr/debug-pdf`

Debug específico para PDFs.

#### `GET /ocr/supported-languages`

Idiomas disponibles para OCR.

#### `DELETE /ocr/cache`

Limpiar cache del sistema.

## 📄 Tipos de Documentos Soportados

### Formatos de Archivo

- **PDF**: Hasta 5 páginas, con texto nativo o escaneado
- **Imágenes**: JPG, PNG, TIFF, BMP
- **Tamaño máximo**: 50MB por defecto

### Tipos de Documentos Detectados

- `invoice`: Facturas generales
- `receipt`: Recibos
- `ticket`: Tickets de compra
- `electricity_bill`: Facturas eléctricas
- `gas_bill`: Facturas de gas
- `unknown`: Documento no identificado

### Proveedores de Energía Soportados

- **Naturgy** (Gas Natural)
- **Endesa**
- **Iberdrola**
- **EDP**
- **Repsol Luz**
- **TotalEnergies**

## ⚙️ Configuración Avanzada

### Calidades de Procesamiento

#### `fast`

- **Uso**: Documentos de alta calidad, procesamiento rápido
- **Tiempo**: ~1-3 segundos
- **Precisión**: 70-80%
- **Preprocesamiento**: Mínimo (solo escala de grises)

#### `balanced` (Recomendado)

- **Uso**: Uso general, balance velocidad/precisión
- **Tiempo**: ~3-8 segundos
- **Precisión**: 80-90%
- **Preprocesamiento**: CLAHE, filtros, binarización adaptativa

#### `high`

- **Uso**: Documentos de baja calidad, máxima precisión
- **Tiempo**: ~8-15 segundos
- **Precisión**: 85-95%
- **Preprocesamiento**: Completo con corrección de rotación

### Configuración de Tesseract

```bash
# Idiomas instalados por defecto
TESSERACT_LANGUAGES=spa+eng

# Idiomas adicionales disponibles
# cat (Catalán), eus (Euskera), glg (Gallego), fra (Francés)
```

### Límites del Sistema

| Parámetro         | Valor por Defecto | Descripción                         |
| ----------------- | ----------------- | ----------------------------------- |
| `MAX_FILE_SIZE`   | 50MB              | Tamaño máximo de archivo            |
| `MAX_PAGES_PDF`   | 5                 | Páginas máximas de PDF              |
| `MAX_BATCH_FILES` | 10                | Archivos en procesamiento por lotes |

## 🔄 Cache del Sistema

El sistema incluye cache automático basado en hash SHA256 del contenido:

- **Duración**: 24 horas
- **Ubicación**: `/tmp/ocr_cache` (configurable)
- **Limpieza**: Endpoint `DELETE /ocr/cache`

## 📊 Métricas y Logging

### Métricas Devueltas

- `confidence`: Confianza del OCR (0-100%)
- `quality_score`: Puntuación de calidad de extracción (0-100%)
- `processing_time`: Tiempo de procesamiento en segundos
- `text_length`: Longitud del texto extraído

### Logs del Sistema

```bash
# Ver logs en tiempo real
docker logs -f container_name

# Ejemplo de log
INFO:main:Iniciando OCR - Imagen: (574, 1024), Modo: RGB, Calidad: high
INFO:main:OCR completado - Config: PSM 6, Texto: 384 chars, Confianza: 92.5
```

## 🔒 Seguridad

### CORS

```env
# Permitir todos los orígenes (desarrollo)
ALLOWED_ORIGINS=*

# Restringir orígenes (producción)
ALLOWED_ORIGINS=https://mi-dominio.com,https://www.mi-dominio.com
```

### Validación de Archivos

- Verificación de tipo MIME
- Límites de tamaño
- Timeout en procesamiento (30s por defecto)

## 🚀 Despliegue en Producción

### Docker Compose (Recomendado)

```yaml
version: "3.8"
services:
  ocr-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
      - ALLOWED_ORIGINS=https://tu-dominio.com
      - MAX_FILE_SIZE=50000000
      - TESSERACT_LANGUAGES=spa+eng
    volumes:
      - ./cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### EasyPanel / VPS

1. **Variables de entorno en EasyPanel:**

```
ENVIRONMENT=production
ALLOWED_ORIGINS=*
MAX_FILE_SIZE=50000000
TESSERACT_LANGUAGES=spa+eng
```

2. **Puerto:** `8001`

3. **Health Check:** `/health`

### Recursos Recomendados

- **RAM**: Mínimo 1GB, recomendado 2GB
- **CPU**: 1-2 cores
- **Storage**: 10GB para cache y logs

## 🐛 Troubleshooting

### Problemas Comunes

#### OCR no extrae texto

```bash
# Verificar Tesseract
GET /ocr/test-tesseract

# Debug imagen específica
POST /ocr/debug
```

#### Baja calidad de extracción

```bash
# Probar diferentes calidades
POST /ocr/process?quality=high

# Usar mejoramiento automático
POST /ocr/enhance
```

#### PDFs no procesados

```bash
# Debug específico para PDF
POST /ocr/debug-pdf
```

### Códigos de Error Comunes

| Código | Error                        | Solución                                      |
| ------ | ---------------------------- | --------------------------------------------- |
| 400    | Tipo de archivo no soportado | Usar PDF, JPG, PNG, TIFF, BMP                 |
| 413    | Archivo demasiado grande     | Reducir tamaño o aumentar MAX_FILE_SIZE       |
| 500    | Error interno OCR            | Verificar logs, usar /ocr/debug               |
| 503    | Tesseract no disponible      | Verificar instalación con /ocr/test-tesseract |

## 📈 Performance Tips

1. **Usar cache**: Habilitar `ENABLE_CACHE=true`
2. **Calidad apropiada**: `fast` para documentos limpios, `high` para problemáticos
3. **Batch processing**: Usar `/ocr/batch` para múltiples archivos
4. **Preprocesamiento**: Usar `/ocr/enhance` para imágenes de baja calidad

## 🤝 Contribuir

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📝 Changelog

### v2.0.0

- ✅ Extracción especializada para facturas eléctricas
- ✅ Múltiples técnicas de mejoramiento de imagen
- ✅ Endpoints de debug y diagnóstico
- ✅ Cache inteligente
- ✅ Validación de documentos españoles

### v1.0.0

- ✅ OCR básico con Tesseract
- ✅ API REST con FastAPI
- ✅ Soporte para PDF e imágenes

## 📄 Licencia

MIT License - ver `LICENSE` para detalles.

## 🆘 Soporte

- **Issues**: [GitHub Issues](https://github.com/sebasblancogonz/ocr/issues)
- **Documentación API**: `http://localhost:8001/docs` (Swagger UI)
- **Redoc**: `http://localhost:8001/redoc`

---

Desarrollado con ❤️ por [sebasblancogonz](https://github.com/sebasblancogonz)
