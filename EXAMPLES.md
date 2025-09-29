# 📚 Ejemplos de Uso - Cakely OCR Service

Esta guía contiene ejemplos prácticos de cómo usar la API del servicio OCR.

## 🔥 Ejemplos Básicos

### 1. Procesamiento Simple de Imagen

```bash
# Procesar una imagen con calidad balanceada
curl -X POST "http://localhost:8001/ocr/process" \
  -F "file=@ticket.jpg"
```

```javascript
// JavaScript/Node.js
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:8001/ocr/process", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

```python
# Python
import requests

with open('factura.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8001/ocr/process', files=files)
    result = response.json()
    print(result['raw_text'])
```

### 2. Procesamiento con Alta Calidad

```bash
curl -X POST "http://localhost:8001/ocr/process?quality=high&extract_items=true" \
  -F "file=@factura_detallada.pdf"
```

## 🏭 Casos de Uso Específicos

### Factura Eléctrica de Naturgy

```bash
# Procesamiento especializado
curl -X POST "http://localhost:8001/ocr/process-utility-bill?quality=high" \
  -F "file=@factura_naturgy.pdf"
```

**Respuesta esperada:**

```json
{
  "success": true,
  "supplier": "naturgy",
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
    "vat": 8.95,
    "services": 8.95
  }
}
```

### Validación de Factura Empresarial

```bash
# Validar si cumple requisitos fiscales españoles
curl -X POST "http://localhost:8001/ocr/validate" \
  -F "file=@factura_empresa.pdf"
```

**Respuesta:**

```json
{
  "valid": true,
  "completeness_percentage": 92.5,
  "validation_results": {
    "has_tax_id": true,
    "has_invoice_number": true,
    "has_date": true,
    "has_total": true,
    "has_supplier": true,
    "has_vat_info": true
  },
  "extracted_summary": {
    "supplier": "EMPRESA EJEMPLO SL",
    "invoice_number": "2024-001234",
    "total": 234.56,
    "tax_id": "B12345678"
  }
}
```

## 🔧 Diagnóstico y Debug

### Imagen de Baja Calidad

```bash
# 1. Primero diagnosticar
curl -X POST "http://localhost:8001/ocr/debug" \
  -F "file=@imagen_borrosa.jpg"

# 2. Usar mejoramiento automático
curl -X POST "http://localhost:8001/ocr/enhance" \
  -F "file=@imagen_borrosa.jpg"
```

### PDF Problemático

```bash
# Debug específico para PDF
curl -X POST "http://localhost:8001/ocr/debug-pdf" \
  -F "file=@documento_escaneado.pdf"
```

**Respuesta:**

```json
{
  "pdf_info": {
    "has_native_text": false,
    "size_mb": 3.2
  },
  "conversion_results": {
    "dpi_200": {
      "success": true,
      "pages_converted": 3,
      "ocr_preview": {
        "confidence": 78.5,
        "text_length": 1250
      }
    }
  },
  "recommendations": [
    "⚠️ PDF es imagen escaneada - requiere OCR",
    "✅ Conversión exitosa con DPI 200"
  ]
}
```

## 🚀 Casos de Uso Avanzados

### Procesamiento por Lotes

```bash
# Procesar múltiples archivos
curl -X POST "http://localhost:8001/ocr/batch?quality=balanced" \
  -F "files=@factura1.pdf" \
  -F "files=@factura2.pdf" \
  -F "files=@ticket1.jpg"
```

```python
# Python - Batch processing
import requests
import os

files = []
for filename in ['doc1.pdf', 'doc2.jpg', 'doc3.png']:
    files.append(('files', open(filename, 'rb')))

response = requests.post(
    'http://localhost:8001/ocr/batch',
    files=files
)

# Cerrar archivos
for _, file_obj in files:
    file_obj.close()

result = response.json()
for doc in result['results']:
    print(f"{doc['filename']}: {doc['document_type']}")
```

### Integración con Base de Datos

```python
import requests
import sqlite3
from datetime import datetime

def process_invoice_to_db(file_path):
    # Procesar factura
    with open(file_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8001/ocr/process-utility-bill',
            files={'file': f},
            params={'quality': 'high'}
        )

    data = response.json()

    if data['success']:
        # Insertar en base de datos
        conn = sqlite3.connect('facturas.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO facturas (
                filename, supplier, invoice_number,
                total_amount, period_start, period_end,
                processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['filename'],
            data.get('supplier'),
            data['billing_information'].get('invoice_number'),
            data['amounts'].get('total'),
            data['billing_information'].get('period_start'),
            data['billing_information'].get('period_end'),
            datetime.now()
        ))

        conn.commit()
        conn.close()

        return True
    return False

# Uso
success = process_invoice_to_db('factura_endesa.pdf')
print(f"Procesado: {success}")
```

## 🌐 Integración Frontend

### React Component

```jsx
import React, { useState } from "react";
import axios from "axios";

const OCRUploader = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const processFile = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:8001/ocr/process?quality=high",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} accept=".pdf,.jpg,.png" />
      <button onClick={processFile} disabled={!file || loading}>
        {loading ? "Procesando..." : "Procesar OCR"}
      </button>

      {result && (
        <div>
          <h3>Resultado:</h3>
          <p>
            <strong>Tipo:</strong> {result.document_type}
          </p>
          <p>
            <strong>Confianza:</strong> {result.metrics.confidence}%
          </p>

          {result.extracted_data.amounts && (
            <div>
              <h4>Importes:</h4>
              <p>Total: €{result.extracted_data.amounts.total}</p>
            </div>
          )}

          <details>
            <summary>Texto completo</summary>
            <pre>{result.raw_text}</pre>
          </details>
        </div>
      )}
    </div>
  );
};

export default OCRUploader;
```

### Vue.js Component

```vue
<template>
  <div>
    <input
      type="file"
      @change="handleFileChange"
      accept=".pdf,.jpg,.png"
      ref="fileInput"
    />
    <button @click="processFile" :disabled="!file || loading">
      {{ loading ? "Procesando..." : "Procesar OCR" }}
    </button>

    <div v-if="result" class="result">
      <h3>Resultado:</h3>
      <div class="info-grid">
        <div><strong>Archivo:</strong> {{ result.filename }}</div>
        <div><strong>Tipo:</strong> {{ result.document_type }}</div>
        <div><strong>Confianza:</strong> {{ result.metrics.confidence }}%</div>
        <div>
          <strong>Tiempo:</strong> {{ result.metrics.processing_time }}s
        </div>
      </div>

      <div v-if="result.extracted_data.financial?.total" class="amounts">
        <h4>Información Financiera:</h4>
        <p>Total: €{{ result.extracted_data.financial.total }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      file: null,
      result: null,
      loading: false,
    };
  },
  methods: {
    handleFileChange(event) {
      this.file = event.target.files[0];
    },
    async processFile() {
      if (!this.file) return;

      this.loading = true;
      const formData = new FormData();
      formData.append("file", this.file);

      try {
        const response = await axios.post(
          "http://localhost:8001/ocr/process",
          formData
        );
        this.result = response.data;
      } catch (error) {
        console.error("Error:", error);
        alert("Error procesando archivo");
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>
```

## 📊 Monitoreo y Analytics

### Script de Monitoreo

```bash
#!/bin/bash
# monitor_ocr.sh

API_URL="http://localhost:8001"

# Health check
echo "=== Health Check ==="
curl -s "$API_URL/health" | jq '.status'

# Test Tesseract
echo "=== Tesseract Status ==="
curl -s "$API_URL/ocr/test-tesseract" | jq '.tesseract_working'

# Cache status
echo "=== Cache Info ==="
curl -s "$API_URL/health" | jq '.cache'

echo "Monitoreo completado: $(date)"
```

### Métricas con Python

```python
import requests
import time
import json
from datetime import datetime

class OCRMonitor:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        self.metrics = []

    def health_check(self):
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            return {
                'timestamp': datetime.now().isoformat(),
                'status': response.status_code,
                'healthy': response.status_code == 200,
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }

    def performance_test(self, test_file):
        """Test de rendimiento con archivo específico"""
        start_time = time.time()

        try:
            with open(test_file, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/ocr/process",
                    files={'file': f},
                    timeout=60
                )

            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'total_time': end_time - start_time,
                    'processing_time': data['metrics']['processing_time'],
                    'confidence': data['metrics']['confidence'],
                    'text_length': data['metrics']['text_length']
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

# Uso
monitor = OCRMonitor()
health = monitor.health_check()
print(f"Service healthy: {health['healthy']}")

if health['healthy']:
    perf = monitor.performance_test('test_document.pdf')
    print(f"Performance test: {perf}")
```

## 🔄 Automatización y CI/CD

### GitHub Actions Workflow

```yaml
# .github/workflows/ocr-service.yml
name: OCR Service CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -t ocr-service .

      - name: Start service
        run: |
          docker run -d -p 8001:8001 --name ocr-test ocr-service
          sleep 30

      - name: Health check
        run: |
          curl -f http://localhost:8001/health

      - name: Test Tesseract
        run: |
          curl -f http://localhost:8001/ocr/test-tesseract

      - name: Test OCR functionality
        run: |
          # Crear imagen de prueba simple
          echo "Running OCR tests..."
          # Aquí irían tests más específicos

      - name: Cleanup
        run: docker stop ocr-test && docker rm ocr-test
```

### Script de Despliegue

```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Desplegando OCR Service..."

# Variables
IMAGE_NAME="ocr-service"
CONTAINER_NAME="ocr-prod"
PORT=8001

# Construir nueva imagen
echo "📦 Construyendo imagen..."
docker build -t $IMAGE_NAME:latest .

# Detener contenedor existente
echo "🛑 Deteniendo contenedor anterior..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Ejecutar nuevo contenedor
echo "▶️ Iniciando nuevo contenedor..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  --restart unless-stopped \
  -e ENVIRONMENT=production \
  -e ALLOWED_ORIGINS="*" \
  -v $(pwd)/cache:/app/cache \
  $IMAGE_NAME:latest

# Verificar salud
echo "🔍 Verificando salud del servicio..."
sleep 10

for i in {1..12}; do
  if curl -f http://localhost:$PORT/health; then
    echo "✅ Servicio desplegado correctamente!"
    exit 0
  fi
  echo "⏳ Esperando... ($i/12)"
  sleep 5
done

echo "❌ Error: El servicio no responde después del despliegue"
docker logs $CONTAINER_NAME
exit 1
```

---

¿Te gustaría que añada más ejemplos específicos o casos de uso particulares?
