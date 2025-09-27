from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import io
import re
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Cakely OCR Service", version="1.0.0")

# Configurar CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar ruta de Tesseract (ajustar según tu instalación)
# En Ubuntu/Debian: apt-get install tesseract-ocr tesseract-ocr-spa
# En macOS: brew install tesseract tesseract-lang
tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Configuraciones desde variables de entorno
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB por defecto
MAX_PAGES_PDF = int(os.getenv("MAX_PAGES_PDF", "3"))
TESSERACT_LANGUAGES = os.getenv("TESSERACT_LANGUAGES", "spa+eng")

class OCRProcessor:
    def __init__(self):
        self.supported_languages = ['spa', 'eng']  # Español e Inglés
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesar imagen para mejorar OCR"""
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Reducir ruido
        denoised = cv2.medianBlur(gray, 3)
        
        # Mejorar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extraer texto de imagen usando Tesseract"""
        # Convertir PIL a OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocesar
        processed = self.preprocess_image(opencv_image)
        
        # OCR con configuración optimizada para facturas
        custom_config = f'--oem 3 --psm 6 -l {TESSERACT_LANGUAGES}'
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extraer texto de PDF"""
        try:
            # Convertir PDF a imágenes
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=MAX_PAGES_PDF)
            
            all_text = []
            for i, image in enumerate(images):
                text = self.extract_text_from_image(image)
                all_text.append(f"--- Página {i+1} ---\n{text}")
            
            return "\n\n".join(all_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error procesando PDF: {str(e)}")
    
    def extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extraer datos estructurados del texto OCR"""
        patterns = {
            # Montos en diferentes formatos
            'amount': r'(?:€\s*)?(\d+[.,]\d{2})(?:\s*€|\s*EUR)?',
            # CIF/NIF español
            'tax_id': r'(?:CIF|NIF|VAT)[:\s]*([A-Z]\d{7,8}[A-Z0-9])',
            # Fechas DD/MM/YYYY o DD-MM-YYYY
            'date': r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',
            # Número de factura/ticket
            'invoice_number': r'(?:factura|invoice|ticket|n[°º])[:\s]*([A-Z0-9\-\/]+)',
            # IVA
            'vat': r'IVA[:\s]*(\d+[.,]\d{2})',
            # Total
            'total': r'(?:total|importe)[:\s]*(\d+[.,]\d{2})',
            # Email
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            # Teléfono
            'phone': r'(\+?[0-9\s\-\(\)]{9,15})',
        }
        
        extracted = {}
        
        # Buscar cada patrón
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if key == 'amount' or key == 'total' or key == 'vat':
                # Para montos, tomar el más alto
                if matches:
                    amounts = [float(m.replace(',', '.')) for m in matches]
                    extracted[key] = max(amounts)
            elif key == 'date':
                # Para fechas, formatear a ISO
                if matches:
                    day, month, year = matches[0]
                    extracted[key] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                # Para otros, tomar el primero
                if matches:
                    extracted[key] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        # Intentar extraer nombre del proveedor (primera línea que parezca nombre)
        lines = text.split('\n')
        for line in lines[:5]:  # Solo primeras 5 líneas
            line = line.strip()
            if len(line) > 5 and len(line) < 50 and not any(char.isdigit() for char in line):
                if not any(keyword in line.lower() for keyword in ['factura', 'ticket', 'fecha', 'cif', 'nif']):
                    extracted['supplier_name'] = line
                    break
        
        return extracted

# Instancia del procesador
ocr_processor = OCRProcessor()

@app.get("/")
async def root():
    return {"message": "Cakely OCR Service", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ocr/process")
async def process_ocr(file: UploadFile = File(...)):
    """Procesar archivo y extraer texto + datos estructurados"""
    
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Tipo de archivo no especificado")
    
    # Leer archivo
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {str(e)}")
    
    # Procesar según tipo de archivo
    try:
        if file.content_type.startswith('image/'):
            # Procesar imagen
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            text = ocr_processor.extract_text_from_image(image)
            
        elif file.content_type == 'application/pdf':
            # Procesar PDF
            text = ocr_processor.extract_text_from_pdf(content)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo no soportado: {file.content_type}"
            )
        
        # Extraer datos estructurados
        extracted_data = ocr_processor.extract_data_from_text(text)
        
        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "raw_text": text,
            "extracted_data": extracted_data,
            "confidence": "high" if len(text) > 50 else "low"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando OCR: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)