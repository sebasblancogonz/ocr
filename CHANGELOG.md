# 📝 Changelog

Todos los cambios importantes de este proyecto se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-09-29

### ✨ Añadido

- **Endpoint de mejoramiento** `/ocr/enhance` para imágenes de baja calidad
- **Procesamiento con múltiples técnicas** (CLAHE, filtros bilaterales, etc.)
- **Logging detallado** para debug y monitoreo
- **Manejo robusto de errores** en cada método de procesamiento
- **Documentación completa** (README.md, EXAMPLES.md)
- **Configuración avanzada** con .env.example detallado

### 🔧 Mejorado

- **Preprocesamiento de imágenes** con binarización adaptativa
- **Limpieza de texto** más inteligente con corrección de errores OCR
- **Detección automática de rotación** en imágenes
- **Redimensionado inteligente** para imágenes pequeñas
- **Múltiples configuraciones PSM** para mejor extracción

### 🐛 Corregido

- Error `cv2.medianFilter` → `cv2.medianBlur`
- Manejo de errores individual por método de mejoramiento
- Protección contra fallos en redimensionado de imágenes

## [2.0.0] - 2025-09-28

### ✨ Añadido

- **Extracción especializada** para facturas eléctricas españolas
- **Soporte para proveedores**: Naturgy, Endesa, Iberdrola, EDP, Repsol, TotalEnergies
- **Detección automática de tipo de documento**
- **Validación de documentos** según normativa española
- **Endpoint especializado** `/ocr/process-utility-bill`
- **Cache inteligente** basado en hash SHA256
- **Procesamiento por lotes** hasta 10 archivos
- **Endpoints de debug** para diagnóstico
- **Health check avanzado** con información del sistema
- **Soporte mejorado para PDFs** con extracción de texto nativo

### 🔧 Mejorado

- **Arquitectura modular** con clases especializadas
- **Mejor manejo de errores** con logging detallado
- **Configuración por variables de entorno**
- **Documentación automática** con Swagger/OpenAPI
- **Métricas de rendimiento** y calidad

### 🐛 Corregido

- Problemas con dependencias de OpenCV en contenedores
- Manejo de PDFs encriptados
- Timeout en procesamiento de archivos grandes
- Compatibilidad con diferentes formatos de imagen

### 💥 Cambios que Rompen Compatibilidad

- **Estructura de respuesta** actualizada con más campos
- **Endpoints renombrados** para mayor claridad
- **Nuevos campos requeridos** en algunas respuestas

## [1.2.0] - 2025-09-27

### ✨ Añadido

- **Dockerfile optimizado** para producción
- **Health checks** para contenedores
- **Variables de entorno** configurables
- **Soporte para EasyPanel** y otros PaaS

### 🔧 Mejorado

- **Instalación de dependencias** más eficiente
- **Tamaño de imagen Docker** optimizado
- **Configuración CORS** más flexible

### 🐛 Corregido

- Problemas con librerías gráficas en contenedores slim
- Dependencias faltantes para Tesseract

## [1.1.0] - 2025-09-26

### ✨ Añadido

- **Soporte para múltiples idiomas** (español, inglés, catalán)
- **Procesamiento de PDFs** con pdf2image
- **Extracción básica de datos** estructurados
- **Configuración de calidad** de procesamiento

### 🔧 Mejorado

- **Precisión del OCR** con preprocesamiento de imágenes
- **Manejo de archivos grandes** con límites configurables
- **Respuestas de API** más informativas

### 🐛 Corregido

- Errores con archivos corruptos
- Problemas de memoria con PDFs grandes
- Encoding de caracteres especiales

## [1.0.0] - 2025-09-25

### ✨ Añadido

- **API REST básica** con FastAPI
- **OCR básico** con Tesseract
- **Soporte para imágenes** (JPG, PNG, TIFF, BMP)
- **Endpoint de procesamiento** `/ocr/process`
- **Documentación inicial**

### 🔧 Características Iniciales

- Extracción de texto básica
- Respuestas JSON estructuradas
- Manejo básico de errores
- CORS habilitado

---

## 🔮 Próximas Versiones

### [2.2.0] - Planificado

- **Integración con AI/ML** para mejor comprensión de documentos
- **API de webhooks** para notificaciones asíncronas
- **Soporte para más idiomas** europeos
- **Métricas avanzadas** con Prometheus
- **Rate limiting** y autenticación

### [2.3.0] - En Consideración

- **Procesamiento de facturas internacionales**
- **Extracción de tablas** complejas
- **Soporte para documentos manuscritos**
- **Integración con cloud storage** (S3, GCS)
- **API GraphQL** como alternativa

---

## 📊 Estadísticas de Versiones

| Versión | Fecha      | Commits | Líneas Añadidas | Líneas Eliminadas |
| ------- | ---------- | ------- | --------------- | ----------------- |
| 2.1.0   | 2025-09-29 | 15      | 1,250           | 180               |
| 2.0.0   | 2025-09-28 | 32      | 2,100           | 450               |
| 1.2.0   | 2025-09-27 | 8       | 320             | 95                |
| 1.1.0   | 2025-09-26 | 12      | 680             | 120               |
| 1.0.0   | 2025-09-25 | 25      | 1,500           | 0                 |

---

## 🏷️ Convenciones de Etiquetado

Este proyecto usa las siguientes etiquetas para organizar los cambios:

- ✨ **Añadido** - Para nuevas funcionalidades
- 🔧 **Mejorado** - Para mejoras en funcionalidades existentes
- 🐛 **Corregido** - Para correcciones de bugs
- 💥 **Cambios que Rompen Compatibilidad** - Para cambios incompatibles
- 🔒 **Seguridad** - Para vulnerabilidades corregidas
- 📚 **Documentación** - Solo cambios en documentación
- 🧪 **Testing** - Añadir o corregir tests
- 🚀 **Performance** - Mejoras de rendimiento

---

## 🤝 Contribuyendo

Para contribuir al changelog:

1. Cada PR debe actualizar este archivo
2. Usar el formato de [Keep a Changelog](https://keepachangelog.com/)
3. Incluir enlaces a issues/PRs cuando sea relevante
4. Categorizar cambios apropiadamente
5. Mantener orden cronológico (más reciente arriba)

---

## 📋 Template para Nuevas Versiones

```markdown
## [X.Y.Z] - YYYY-MM-DD

### ✨ Añadido

- Nueva funcionalidad X
- Soporte para Y

### 🔧 Mejorado

- Mejora en Z
- Optimización de W

### 🐛 Corregido

- Bug fix para A
- Corrección en B

### 💥 Cambios que Rompen Compatibilidad

- Cambio incompatible en C
```
