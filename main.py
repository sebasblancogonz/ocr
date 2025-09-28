from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import io
import re
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import os
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
import tempfile
from pathlib import Path
# Importar el procesador especializado para facturas eléctricas
# from electricity_bill_processor import ElectricityBillProcessor, ElectricitySupplier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="Cakely OCR Service", 
    version="2.0.0",
    description="Servicio avanzado de OCR para procesamiento de facturas y tickets"
)

# Configurar CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuraciones
tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
MAX_PAGES_PDF = int(os.getenv("MAX_PAGES_PDF", "5"))
TESSERACT_LANGUAGES = os.getenv("TESSERACT_LANGUAGES", "spa+eng")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/ocr_cache"))

# Crear directorio de caché si no existe
if ENABLE_CACHE:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pool de threads para procesamiento paralelo
executor = ThreadPoolExecutor(max_workers=4)

class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    TICKET = "ticket"
    ELECTRICITY_BILL = "electricity_bill"
    GAS_BILL = "gas_bill"
    UNKNOWN = "unknown"

class ProcessingQuality(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"

class OCRProcessor:
    def __init__(self):
        self.supported_languages = ['spa', 'eng', 'fra', 'deu', 'ita', 'por']
        self.document_patterns = self._load_document_patterns()
        self.electricity_patterns = self._load_electricity_patterns()
    
    def _load_electricity_patterns(self) -> Dict:
        """Patrones específicos para facturas eléctricas españolas"""
        return {
            'general': {
                'invoice_number': [
                    r'(?:n[°ºo]\s*factura|factura|invoice)[\s:]*([A-Z0-9\-\/]+)',
                    r'(?:número|num\.|n[°ºo])[\s:]*([A-Z0-9\-\/]+)',
                ],
                'cups': r'(?:CUPS|cups)[\s:]*([A-Z]{2}\d{4}\s*\d{10}\s*[A-Z]{2}\d{1}[A-Z]{1})',
                'billing_period': [
                    r'(?:periodo|período)[\s:]*(?:del?\s*)?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s*(?:a|al|-)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                    r'desde\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s*hasta\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                ],
                'total_amount': [
                    r'total\s*a\s*pagar[\s:]*(?:€\s*)?(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})(?:\s*€)?',
                    r'(\d{1,3}[.,]\d{2})\s*€[\s]*total'
                ],
                'electricity_amount': r'(?:consumo\s*)?electricidad[\s:]*(?:€\s*)?(\d{1,3}[.,]\d{2})(?:\s*€)?',
                'gas_amount': r'(?:consumo\s*)?gas[\s:]*(?:€\s*)?(\d{1,3}[.,]\d{2})(?:\s*€)?',
                'vat_amount': r'IVA[\s:]*(?:€\s*)?(\d{1,3}[.,]\d{2})(?:\s*€)?',
                'services_amount': r'servicios[\s:]*(?:€\s*)?(\d{1,3}[.,]\d{2})(?:\s*€)?',
                'kwh_consumed': r'(\d{1,5})\s*kWh',
                'contracted_power': r'potencia\s*contratada[\s:]*(\d+[.,]\d+)\s*kW',
            },
            'naturgy': {
                'customer_greeting': r'Hola,?\s*([^\n]+)',
                'total_pattern': r'Total\s*a\s*pagar[\s:]*(\d{1,3}[.,]\d{2})\s*€',
                'gas_pattern': r'Gas[\s:]*(\d{1,3}[.,]\d{2})\s*€',
                'electricity_pattern': r'Electricidad[\s:]*(\d{1,3}[.,]\d{2})\s*€',
                'services_pattern': r'Servicios[\s:]*(\d{1,3}[.,]\d{2})\s*€',
                'iva_pattern': r'IVA[\s:]*(\d{1,3}[.,]\d{2})\s*€'
            }
        }
        
    def _load_document_patterns(self) -> Dict:
        """Cargar patrones de extracción específicos para España y Europa"""
        return {
            'spanish': {
                'amount': r'(?:€\s*)?(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})(?:\s*€|\s*EUR)?',
                'tax_id': r'(?:CIF|NIF|VAT|DNI)[:\s]*([A-Z]\d{7,8}[A-Z0-9]|\d{8}[A-Z])',
                'date': r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})',
                'invoice_number': r'(?:factura|n[°ºo]|número|num\.?)[:\s]*([A-Z0-9\-\/]+)',
                'vat_rate': r'IVA[:\s]*(\d{1,2})[%\s]',
                'vat_amount': r'IVA[:\s]*(?:€\s*)?(\d+[.,]\d{2})',
                'base_amount': r'(?:base\s*imponible|subtotal)[:\s]*(?:€\s*)?(\d+[.,]\d{2})',
                'total': r'(?:total|importe\s*total|total\s*factura)[:\s]*(?:€\s*)?(\d+[.,]\d{2})',
                'iban': r'(?:IBAN|iban)[:\s]*([A-Z]{2}\d{2}[\s]?(?:\d{4}[\s]?){5})',
                'payment_method': r'(?:forma\s*de\s*pago|método\s*pago)[:\s]*([A-Za-z\s]+)',
            },
            'european': {
                'amount': r'(?:€\s*)?(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})(?:\s*€|\s*EUR)?',
                'vat_number': r'(?:VAT|USt-?IdNr\.?|TVA|P\.?\s*IVA)[:\s]*([A-Z]{2}[A-Z0-9]+)',
                'date_eu': r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})',
                'date_iso': r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})',
            }
        }
    
    def detect_document_type(self, text: str) -> DocumentType:
        """Detectar tipo de documento basado en palabras clave"""
        text_lower = text.lower()
        
        # Detectar facturas de servicios públicos primero
        electricity_keywords = ['electricidad', 'electricity', 'kwh', 'potencia contratada', 
                               'cups', 'naturgy', 'endesa', 'iberdrola', 'edp', 
                               'consumo eléctrico', 'energía']
        gas_keywords = ['gas natural', 'gas', 'm3', 'metro cúbico', 'consumo gas']
        
        # Verificar si es factura eléctrica
        if any(keyword in text_lower for keyword in electricity_keywords):
            # Verificar si también tiene gas (factura combinada)
            if any(keyword in text_lower for keyword in gas_keywords):
                return DocumentType.ELECTRICITY_BILL  # Las combinadas las tratamos como eléctricas
            return DocumentType.ELECTRICITY_BILL
        
        # Verificar si es solo gas
        if any(keyword in text_lower for keyword in gas_keywords):
            return DocumentType.GAS_BILL
        
        # Otros tipos de documentos
        invoice_keywords = ['factura', 'invoice', 'proforma', 'albaran']
        receipt_keywords = ['recibo', 'receipt', 'justificante', 'comprobante']
        ticket_keywords = ['ticket', 'tique', 'nota', 'vale']
        
        if any(keyword in text_lower for keyword in invoice_keywords):
            return DocumentType.INVOICE
        elif any(keyword in text_lower for keyword in receipt_keywords):
            return DocumentType.RECEIPT
        elif any(keyword in text_lower for keyword in ticket_keywords):
            return DocumentType.TICKET
        else:
            return DocumentType.UNKNOWN
    
    def preprocess_image(self, image: np.ndarray, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> np.ndarray:
        """Preprocesar imagen según calidad solicitada con mejoras avanzadas"""
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Verificar y ajustar el tamaño de la imagen
        height, width = gray.shape
        if height < 300 or width < 300:
            # Redimensionar imagen pequeña para mejor OCR
            scale_factor = max(500/width, 500/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
        if quality == ProcessingQuality.FAST:
            # Procesamiento básico pero efectivo
            # Mejorar contraste simple
            enhanced = cv2.equalizeHist(gray)
            # Binarización básica
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        # Reducir ruido con parámetros ajustados
        denoised = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Detectar y corregir rotación automáticamente
        try:
            # Detectar líneas para determinar rotación
            edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 5:
                # Calcular ángulo promedio de las líneas
                angles = []
                for rho, theta in lines[:10]:  # Usar solo las primeras 10 líneas
                    angle = (theta * 180 / np.pi) - 90
                    if abs(angle) < 45:  # Solo ángulos razonables
                        angles.append(angle)
                
                if angles:
                    rotation_angle = np.median(angles)
                    if abs(rotation_angle) > 0.5:  # Solo rotar si hay desviación significativa
                        (h, w) = denoised.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except:
            pass  # Si falla la detección de rotación, continuar sin rotar
        
        if quality == ProcessingQuality.BALANCED:
            # Mejorar contraste adaptativo
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Binarización mejorada
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Limpieza morfológica
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
        
        elif quality == ProcessingQuality.HIGH:
            # Procesamiento completo para máxima precisión
            
            # Detectar y corregir inclinación
            coords = np.column_stack(np.where(denoised > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if angle != 0:
                (h, w) = denoised.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = denoised
            
            # Mejorar contraste adaptativo
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            enhanced = clahe.apply(rotated)
            
            # Morfología para mejorar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(
                morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
    
    def extract_text_from_image(self, image: Image.Image, quality: ProcessingQuality = ProcessingQuality.BALANCED, languages: str = None) -> Tuple[str, float]:
        """Extraer texto con medición de confianza mejorada"""
        
        # Convertir PIL a OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocesar con mejoras
        processed = self.preprocess_image(opencv_image, quality)
        
        # Configuración OCR mejorada según calidad
        psm_modes = {
            ProcessingQuality.FAST: 6,       # Uniform block of text
            ProcessingQuality.BALANCED: 3,   # Automatic page segmentation
            ProcessingQuality.HIGH: 1        # Automatic page segmentation with OSD
        }
        
        lang = languages or TESSERACT_LANGUAGES
        
        # Configuración más robusta
        base_config = f'--oem 3 --psm {psm_modes[quality]} -l {lang}'
        
        # Parámetros adicionales para mejorar la detección
        if quality == ProcessingQuality.HIGH:
            # Configuración más agresiva para alta calidad
            whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ€.,;:!?()[]{}'\\-+*/=@#$%&|<>^_`~ "
            custom_config = f'{base_config} -c tessedit_char_whitelist={whitelist}'
        else:
            custom_config = base_config
        
        try:
            # Intentar OCR múltiple con diferentes configuraciones si es necesario
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Si el texto es muy corto o contiene muchos caracteres raros, intentar con otra configuración
            if len(text.strip()) < 10 or self._has_too_many_invalid_chars(text):
                # Intentar con modo PSM diferente
                fallback_psm = 6 if psm_modes[quality] != 6 else 3
                fallback_config = f'--oem 3 --psm {fallback_psm} -l {lang}'
                fallback_text = pytesseract.image_to_string(processed, config=fallback_config)
                
                # Usar el texto más largo y con menos caracteres inválidos
                if len(fallback_text.strip()) > len(text.strip()) and not self._has_too_many_invalid_chars(fallback_text):
                    text = fallback_text
                    custom_config = fallback_config
            
            # OCR con datos de confianza usando la configuración final
            data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calcular confianza promedio (solo palabras con confianza > 30)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 30]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Limpiar texto de caracteres extraños
            cleaned_text = self._clean_extracted_text(text)
            
            return cleaned_text.strip(), avg_confidence
            
        except Exception as e:
            logger.error(f"Error en extracción OCR: {e}")
            return "", 0.0
    
    def _has_too_many_invalid_chars(self, text: str) -> bool:
        """Verificar si el texto tiene demasiados caracteres inválidos"""
        if not text:
            return True
            
        # Contar caracteres válidos vs inválidos
        valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,;:!?()[]{}"\'-+*/=@#$%&|\\<>^_`~€')
        total_chars = len(text)
        
        if total_chars == 0:
            return True
            
        valid_ratio = valid_chars / total_chars
        return valid_ratio < 0.7  # Si menos del 70% son caracteres válidos
    
    def _clean_extracted_text(self, text: str) -> str:
        """Limpiar texto extraído de caracteres problemáticos"""
        if not text:
            return text
            
        # Reemplazar secuencias de caracteres problemáticos comunes
        replacements = {
            r'[^\w\s.,;:!?()[\]{}"\'\-+*/=@#$%&|\\<>^_`~€À-ÿ]': ' ',  # Caracteres no válidos
            r'\s+': ' ',  # Múltiples espacios
            r'([A-Za-z])\s+([A-Za-z])(?=\s|$)': r'\1\2',  # Letras separadas por espacios
            r'(\d)\s+(\d)': r'\1\2',  # Números separados
        }
        
        cleaned = text
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned.strip()
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> Tuple[str, float]:
        """Extraer texto de PDF con procesamiento mejorado"""
        try:
            # Primero intentar extraer texto nativo del PDF
            native_text = self._extract_native_pdf_text(pdf_bytes)
            if native_text and len(native_text.strip()) > 50:
                logger.info("Usando texto nativo del PDF")
                return native_text, 95.0  # Alta confianza para texto nativo
            
            logger.info("PDF sin texto nativo, usando OCR")
            
            # Configurar DPI según calidad (aumentado para mejor OCR)
            dpi_settings = {
                ProcessingQuality.FAST: 200,
                ProcessingQuality.BALANCED: 250,
                ProcessingQuality.HIGH: 300
            }
            
            # Convertir PDF a imágenes con configuración mejorada
            images = convert_from_bytes(
                pdf_bytes, 
                first_page=1, 
                last_page=MAX_PAGES_PDF,
                dpi=dpi_settings[quality],
                fmt='PNG',  # PNG para mejor calidad
                thread_count=2,  # Usar múltiples threads
                poppler_path=None  # Usar poppler del sistema
            )
            
            if not images:
                raise Exception("No se pudieron convertir las páginas del PDF")
            
            logger.info(f"Convertidas {len(images)} páginas del PDF a imágenes")
            
            # Procesar páginas
            all_text = []
            total_confidence = 0
            successful_pages = 0
            
            for i, image in enumerate(images):
                try:
                    # Verificar que la imagen se convirtió correctamente
                    if image.size[0] < 100 or image.size[1] < 100:
                        logger.warning(f"Página {i+1}: imagen muy pequeña {image.size}")
                        continue
                    
                    # Procesar OCR
                    text, confidence = self.extract_text_from_image(image, quality)
                    
                    if text and len(text.strip()) > 5:  # Solo incluir si hay texto válido
                        all_text.append(f"--- Página {i+1} ---\n{text}")
                        total_confidence += confidence
                        successful_pages += 1
                        logger.info(f"Página {i+1}: {len(text)} caracteres, confianza {confidence:.1f}%")
                    else:
                        logger.warning(f"Página {i+1}: sin texto extraído")
                        all_text.append(f"--- Página {i+1} ---\n[Sin texto detectado]")
                        
                except Exception as page_error:
                    logger.error(f"Error procesando página {i+1}: {page_error}")
                    all_text.append(f"--- Página {i+1} ---\n[Error procesando página]")
            
            # Calcular confianza promedio
            avg_confidence = total_confidence / successful_pages if successful_pages > 0 else 0
            
            final_text = "\n\n".join(all_text)
            
            if not final_text.strip() or successful_pages == 0:
                logger.error("No se pudo extraer texto de ninguna página")
                return "[No se pudo extraer texto del PDF]", 0.0
            
            logger.info(f"Procesamiento completado: {successful_pages}/{len(images)} páginas exitosas")
            return final_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error procesando PDF: {str(e)}")
    
    def _extract_native_pdf_text(self, pdf_bytes: bytes) -> str:
        """Intentar extraer texto nativo del PDF"""
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            
            if pdf_reader.is_encrypted:
                return ""
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(f"--- Página {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extrayendo texto nativo de página {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_parts) if text_parts else ""
            
        except ImportError:
            logger.warning("PyPDF2 no disponible, usando solo OCR")
            return ""
        except Exception as e:
            logger.warning(f"Error extrayendo texto nativo: {e}")
            return ""
    
    def extract_structured_data(self, text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Extracción avanzada de datos estructurados"""
        
        # Si es factura eléctrica, usar extractor especializado
        if doc_type in [DocumentType.ELECTRICITY_BILL, DocumentType.GAS_BILL]:
            return self.extract_electricity_bill_data(text, doc_type)
        
        # Para otros tipos de documento, usar el extractor general
        patterns = self.document_patterns['spanish']
        extracted = {
            'document_type': doc_type.value,
            'extraction_timestamp': datetime.now().isoformat(),
            'financial': {},
            'identification': {},
            'dates': [],
            'contacts': {},
            'items': []
        }
        
        # Extraer datos financieros
        for key in ['amount', 'total', 'base_amount', 'vat_amount']:
            matches = re.findall(patterns.get(key, ''), text, re.IGNORECASE | re.MULTILINE)
            if matches:
                amounts = []
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    # Normalizar formato numérico
                    normalized = match.replace('.', '').replace(',', '.')
                    try:
                        amounts.append(float(normalized))
                    except ValueError:
                        continue
                
                if amounts:
                    if key == 'total':
                        extracted['financial'][key] = max(amounts)
                    else:
                        extracted['financial'][key] = amounts[0] if len(amounts) == 1 else amounts
        
        # Extraer IVA
        vat_rate_matches = re.findall(patterns['vat_rate'], text, re.IGNORECASE)
        if vat_rate_matches:
            extracted['financial']['vat_rate'] = int(vat_rate_matches[0])
        
        # Extraer identificación fiscal
        tax_matches = re.findall(patterns['tax_id'], text, re.IGNORECASE)
        if tax_matches:
            extracted['identification']['tax_id'] = tax_matches[0]
        
        # Extraer número de factura
        invoice_matches = re.findall(patterns['invoice_number'], text, re.IGNORECASE)
        if invoice_matches:
            extracted['identification']['invoice_number'] = invoice_matches[0]
        
        # Extraer IBAN
        iban_matches = re.findall(patterns['iban'], text, re.IGNORECASE)
        if iban_matches:
            extracted['financial']['iban'] = iban_matches[0].replace(' ', '')
        
        # Extraer fechas
        date_matches = re.findall(patterns['date'], text, re.IGNORECASE)
        for date_match in date_matches[:3]:  # Máximo 3 fechas
            day, month, year = date_match
            if len(year) == 2:
                year = '20' + year
            try:
                date_obj = datetime(int(year), int(month), int(day))
                extracted['dates'].append(date_obj.strftime('%Y-%m-%d'))
            except ValueError:
                continue
        
        # Extraer contactos
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        phone_pattern = r'(\+?34)?[\s.-]?[6789]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}'
        
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            extracted['contacts']['email'] = email_matches[0]
        
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            extracted['contacts']['phone'] = re.sub(r'[\s.-]', '', phone_matches[0])
        
        # Extraer nombre del proveedor
        extracted['identification']['supplier'] = self._extract_supplier_name(text)
        
        # Intentar extraer líneas de items (para facturas detalladas)
        if doc_type == DocumentType.INVOICE:
            extracted['items'] = self._extract_line_items(text)
        
        # Calcular score de calidad
        extracted['quality_score'] = self._calculate_quality_score(extracted)
        
        return extracted
    
    def _extract_supplier_name(self, text: str) -> Optional[str]:
        """Extraer nombre del proveedor usando heurísticas mejoradas"""
        lines = text.split('\n')
        
        # Patrones a excluir
        exclude_patterns = [
            r'^factura', r'^invoice', r'^ticket', r'^fecha', r'^date',
            r'^cif', r'^nif', r'^vat', r'^total', r'^iva', r'^\d+',
            r'^€', r'^eur', r'^[a-z]{1,3}$'
        ]
        
        for line in lines[:10]:  # Buscar en primeras 10 líneas
            line = line.strip()
            
            # Validaciones
            if len(line) < 3 or len(line) > 60:
                continue
            
            # Excluir líneas con patrones no deseados
            if any(re.match(pattern, line.lower()) for pattern in exclude_patterns):
                continue
            
            # Debe contener al menos una letra mayúscula
            if not any(c.isupper() for c in line):
                continue
            
            # No debe ser solo números
            if line.replace(' ', '').replace('-', '').replace('.', '').isdigit():
                continue
            
            # Posible nombre de empresa
            if re.match(r'^[A-Z][A-Za-z\s&,.\-]+', line):
                return line
        
        return None
    
    def _extract_line_items(self, text: str) -> List[Dict]:
        """Extraer items de línea de una factura"""
        items = []
        
        # Patrón para líneas de items (descripción + cantidad + precio)
        item_pattern = r'([A-Za-z\s]+)\s+(\d+)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})'
        
        matches = re.findall(item_pattern, text, re.MULTILINE)
        
        for match in matches:
            description, quantity, unit_price, total = match
            
            try:
                items.append({
                    'description': description.strip(),
                    'quantity': int(quantity),
                    'unit_price': float(unit_price.replace(',', '.')),
                    'total': float(total.replace(',', '.'))
                })
            except ValueError:
                continue
        
        return items
    
    def _calculate_quality_score(self, extracted_data: Dict) -> float:
        """Calcular puntuación de calidad de extracción"""
        score = 0
        max_score = 100
        
        # Puntuar según campos extraídos
        scoring_rules = {
            'financial': {
                'total': 20,
                'vat_amount': 10,
                'base_amount': 10,
                'vat_rate': 5
            },
            'identification': {
                'tax_id': 15,
                'invoice_number': 10,
                'supplier': 10
            },
            'dates': 10,  # Al menos una fecha
            'contacts': {
                'email': 5,
                'phone': 5
            },
            'items': 10  # Si tiene items detallados
        }
        
        # Evaluar financial
        for field, points in scoring_rules['financial'].items():
            if field in extracted_data.get('financial', {}):
                score += points
        
        # Evaluar identification
        for field, points in scoring_rules['identification'].items():
            if field in extracted_data.get('identification', {}):
                score += points
        
        # Evaluar dates
        if extracted_data.get('dates'):
            score += scoring_rules['dates']
        
        # Evaluar contacts
        for field, points in scoring_rules['contacts'].items():
            if field in extracted_data.get('contacts', {}):
                score += points
        
        # Evaluar items
        if len(extracted_data.get('items', [])) > 0:
            score += scoring_rules['items']
        
        return min(score / max_score * 100, 100)
    
    def extract_electricity_bill_data(self, text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Extracción especializada para facturas de electricidad y gas"""
        
        # Detectar proveedor
        text_lower = text.lower()
        supplier = 'unknown'
        
        suppliers = {
            'naturgy': ['naturgy', 'naturgy iberia', 'gas natural'],
            'endesa': ['endesa', 'endesa energía'],
            'iberdrola': ['iberdrola', 'iberdrola clientes'],
            'edp': ['edp', 'edp comercializadora'],
            'repsol': ['repsol', 'repsol luz'],
            'totalenergies': ['totalenergies', 'total energies']
        }
        
        for supplier_name, keywords in suppliers.items():
            if any(keyword in text_lower for keyword in keywords):
                supplier = supplier_name
                break
        
        # Estructura de datos para factura eléctrica
        extracted = {
            'document_type': doc_type.value,
            'supplier': supplier,
            'extraction_timestamp': datetime.now().isoformat(),
            'billing_info': {},
            'consumption': {},
            'amounts': {},
            'supply_info': {},
            'breakdown': {},
            'financial': {},
            'identification': {},
            'dates': []
        }
        
        # Usar patrones específicos del proveedor si está identificado
        patterns = self.electricity_patterns['general'].copy()
        if supplier in self.electricity_patterns:
            supplier_patterns = self.electricity_patterns[supplier]
            patterns.update(supplier_patterns)
        
        # Extraer CUPS (crítico en facturas eléctricas)
        cups_match = re.search(patterns.get('cups', ''), text, re.IGNORECASE)
        if cups_match:
            extracted['supply_info']['cups'] = cups_match.group(1).replace(' ', '')
            extracted['identification']['cups'] = cups_match.group(1).replace(' ', '')
        
        # Extraer número de factura
        for pattern in [patterns.get('invoice_number')] if isinstance(patterns.get('invoice_number'), str) else patterns.get('invoice_number', []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['billing_info']['invoice_number'] = match.group(1)
                extracted['identification']['invoice_number'] = match.group(1)
                break
        
        # Extraer periodo de facturación
        for pattern in patterns.get('billing_period', []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_date = self._parse_date_flexible(match.group(1))
                end_date = self._parse_date_flexible(match.group(2))
                extracted['billing_info']['period_start'] = start_date
                extracted['billing_info']['period_end'] = end_date
                extracted['dates'] = [start_date, end_date]
                break
        
        # Extraer importes según el proveedor
        if supplier == 'naturgy':
            # Patrones específicos de Naturgy
            total_match = re.search(patterns.get('total_pattern', ''), text, re.IGNORECASE)
            if total_match:
                total = self._parse_amount(total_match.group(1))
                extracted['amounts']['total'] = total
                extracted['financial']['total'] = total
            
            gas_match = re.search(patterns.get('gas_pattern', ''), text, re.IGNORECASE)
            if gas_match:
                gas = self._parse_amount(gas_match.group(1))
                extracted['amounts']['gas'] = gas
                extracted['breakdown']['gas'] = gas
            
            elec_match = re.search(patterns.get('electricity_pattern', ''), text, re.IGNORECASE)
            if elec_match:
                electricity = self._parse_amount(elec_match.group(1))
                extracted['amounts']['electricity'] = electricity
                extracted['breakdown']['electricity'] = electricity
            
            iva_match = re.search(patterns.get('iva_pattern', ''), text, re.IGNORECASE)
            if iva_match:
                vat = self._parse_amount(iva_match.group(1))
                extracted['amounts']['vat'] = vat
                extracted['financial']['vat_amount'] = vat
            
            services_match = re.search(patterns.get('services_pattern', ''), text, re.IGNORECASE)
            if services_match:
                services = self._parse_amount(services_match.group(1))
                extracted['amounts']['services'] = services
                extracted['breakdown']['services'] = services
        else:
            # Patrones generales
            for pattern in patterns.get('total_amount', []):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    total = self._parse_amount(match.group(1))
                    extracted['amounts']['total'] = total
                    extracted['financial']['total'] = total
                    break
            
            # Electricidad
            elec_match = re.search(patterns.get('electricity_amount', ''), text, re.IGNORECASE)
            if elec_match:
                electricity = self._parse_amount(elec_match.group(1))
                extracted['amounts']['electricity'] = electricity
                extracted['breakdown']['electricity'] = electricity
            
            # Gas
            gas_match = re.search(patterns.get('gas_amount', ''), text, re.IGNORECASE)
            if gas_match:
                gas = self._parse_amount(gas_match.group(1))
                extracted['amounts']['gas'] = gas
                extracted['breakdown']['gas'] = gas
            
            # IVA
            vat_match = re.search(patterns.get('vat_amount', ''), text, re.IGNORECASE)
            if vat_match:
                vat = self._parse_amount(vat_match.group(1))
                extracted['amounts']['vat'] = vat
                extracted['financial']['vat_amount'] = vat
        
        # Extraer consumo en kWh
        kwh_match = re.search(patterns.get('kwh_consumed', ''), text, re.IGNORECASE)
        if kwh_match:
            extracted['consumption']['kwh'] = int(kwh_match.group(1))
        
        # Extraer potencia contratada
        power_match = re.search(patterns.get('contracted_power', ''), text, re.IGNORECASE)
        if power_match:
            extracted['consumption']['contracted_power_kw'] = float(power_match.group(1).replace(',', '.'))
        
        # Calcular base imponible si tenemos total e IVA
        if extracted.get('financial', {}).get('total') and extracted.get('financial', {}).get('vat_amount'):
            base = extracted['financial']['total'] - extracted['financial']['vat_amount']
            extracted['financial']['base_amount'] = round(base, 2)
        
        # Buscar dirección del suministro
        address_pattern = r'(?:dirección|direccion)\s*(?:de\s*)?suministro[\s:]*([^\n]+(?:\n[^\n]+)?)'
        address_match = re.search(address_pattern, text, re.IGNORECASE)
        if address_match:
            extracted['supply_info']['address'] = address_match.group(1).strip()
        
        # Calcular score de calidad específico para facturas eléctricas
        extracted['quality_score'] = self._calculate_electricity_quality_score(extracted)
        
        return extracted
    
    def _parse_amount(self, amount_str: str) -> float:
        """Convertir string de importe a float"""
        amount_str = amount_str.strip().replace(' ', '')
        
        # Manejar formato europeo (1.234,56) o americano (1,234.56)
        if '.' in amount_str and ',' in amount_str:
            # Determinar cuál es el separador decimal
            if amount_str.rindex(',') > amount_str.rindex('.'):
                # Formato europeo: punto para miles, coma para decimales
                amount_str = amount_str.replace('.', '').replace(',', '.')
            else:
                # Formato americano: coma para miles, punto para decimales
                amount_str = amount_str.replace(',', '')
        elif ',' in amount_str:
            # Solo coma, asumir que es decimal
            amount_str = amount_str.replace(',', '.')
        
        try:
            return float(amount_str)
        except ValueError:
            return 0.0
    
    def _parse_date_flexible(self, date_str: str) -> str:
        """Parser de fecha más flexible"""
        date_str = date_str.strip()
        
        # Formatos comunes en facturas españolas
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%d %m %Y', '%d %b %Y', '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                if date_obj.year < 100:
                    date_obj = date_obj.replace(year=2000 + date_obj.year if date_obj.year < 50 else 1900 + date_obj.year)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str
    
    def _calculate_electricity_quality_score(self, data: Dict) -> float:
        """Score de calidad específico para facturas eléctricas"""
        score = 0
        
        # Criterios de puntuación
        scoring = {
            'invoice_number': 10,
            'total': 20,
            'cups': 15,
            'period_dates': 10,
            'supplier': 5,
            'electricity_amount': 10,
            'gas_amount': 5,
            'vat': 10,
            'consumption_kwh': 10,
            'contracted_power': 5
        }
        
        if data.get('billing_info', {}).get('invoice_number'):
            score += scoring['invoice_number']
        
        if data.get('amounts', {}).get('total') or data.get('financial', {}).get('total'):
            score += scoring['total']
        
        if data.get('supply_info', {}).get('cups'):
            score += scoring['cups']
        
        if data.get('billing_info', {}).get('period_start') and data.get('billing_info', {}).get('period_end'):
            score += scoring['period_dates']
        
        if data.get('supplier') and data.get('supplier') != 'unknown':
            score += scoring['supplier']
        
        if data.get('amounts', {}).get('electricity'):
            score += scoring['electricity_amount']
        
        if data.get('amounts', {}).get('gas'):
            score += scoring['gas_amount']
        
        if data.get('amounts', {}).get('vat'):
            score += scoring['vat']
        
        if data.get('consumption', {}).get('kwh'):
            score += scoring['consumption_kwh']
        
        if data.get('consumption', {}).get('contracted_power_kw'):
            score += scoring['contracted_power']
        
        return min(score, 100)

class SimpleCache:
    """Cache simple basado en archivos"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    
    def get_cache_key(self, content: bytes) -> str:
        """Generar clave de caché basada en hash del contenido"""
        return hashlib.sha256(content).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Obtener del caché"""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Verificar que no esté expirado (24 horas)
                    created = datetime.fromisoformat(data['created'])
                    if (datetime.now() - created).days < 1:
                        return data['result']
            except Exception:
                pass
        return None
    
    def set(self, key: str, value: Dict):
        """Guardar en caché"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'created': datetime.now().isoformat(),
                    'result': value
                }, f)
        except Exception as e:
            logger.error(f"Error guardando en caché: {e}")

# Instancias globales
ocr_processor = OCRProcessor()
cache = SimpleCache(CACHE_DIR) if ENABLE_CACHE else None

# Endpoints

@app.get("/")
async def root():
    return {
        "message": "Cakely OCR Service",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/ocr/process",
            "/ocr/batch",
            "/ocr/analyze",
            "/docs"
        ]
    }

@app.get("/health")
async def health():
    """Health check con información detallada del sistema"""
    try:
        # Verificar Tesseract
        result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, '--version'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        tesseract_ok = result.returncode == 0
        tesseract_version = result.stdout.split('\n')[0] if tesseract_ok else "N/A"
        
        # Verificar idiomas disponibles
        langs_result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, '--list-langs'],
            capture_output=True,
            text=True,
            timeout=5
        )
        available_languages = langs_result.stdout.strip().split('\n')[1:] if langs_result.returncode == 0 else []
        
        return {
            "status": "healthy",
            "service": "OCR Service",
            "version": "2.0.0",
            "tesseract": {
                "available": tesseract_ok,
                "version": tesseract_version,
                "languages": available_languages
            },
            "cache": {
                "enabled": ENABLE_CACHE,
                "directory": str(CACHE_DIR) if ENABLE_CACHE else None
            },
            "limits": {
                "max_file_size": MAX_FILE_SIZE,
                "max_pdf_pages": MAX_PAGES_PDF
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "OCR Service"
            }
        )

@app.post("/ocr/process")
async def process_ocr(
    file: UploadFile = File(...),
    quality: ProcessingQuality = Query(ProcessingQuality.BALANCED, description="Calidad de procesamiento"),
    extract_items: bool = Query(False, description="Extraer items de línea"),
    use_cache: bool = Query(True, description="Usar caché si está disponible")
):
    """
    Procesar archivo y extraer texto con datos estructurados.
    
    - **quality**: fast, balanced, or high (trade-off entre velocidad y precisión)
    - **extract_items**: intentar extraer items individuales de facturas
    - **use_cache**: usar resultados cacheados si están disponibles
    """
    
    # Validar tipo de archivo
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Tipo de archivo no especificado")
    
    allowed_types = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'application/pdf']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Tipos permitidos: {allowed_types}"
        )
    
    # Leer archivo
    try:
        content = await file.read()
        
        # Validar tamaño
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {str(e)}")
    
    # Verificar caché
    cache_key = None
    if cache and use_cache and ENABLE_CACHE:
        cache_key = cache.get_cache_key(content)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Resultado obtenido de caché para {file.filename}")
            cached_result['from_cache'] = True
            return cached_result
    
    # Procesar según tipo de archivo
    start_time = datetime.now()
    
    try:
        if file.content_type.startswith('image/'):
            # Procesar imagen
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            text, confidence = ocr_processor.extract_text_from_image(image, quality)
            
        elif file.content_type == 'application/pdf':
            # Procesar PDF
            text, confidence = ocr_processor.extract_text_from_pdf(content, quality)
        
        # Detectar tipo de documento
        doc_type = ocr_processor.detect_document_type(text)
        
        # Extraer datos estructurados
        extracted_data = ocr_processor.extract_structured_data(text, doc_type)
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "document_type": doc_type.value,
            "raw_text": text,
            "extracted_data": extracted_data,
            "metrics": {
                "confidence": round(confidence, 2),
                "quality_score": round(extracted_data['quality_score'], 2),
                "processing_time": round(processing_time, 3),
                "text_length": len(text),
                "quality_setting": quality.value
            },
            "from_cache": False
        }
        
        # Guardar en caché
        if cache and use_cache and ENABLE_CACHE and cache_key:
            cache.set(cache_key, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando OCR: {str(e)}")

@app.post("/ocr/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    quality: ProcessingQuality = Query(ProcessingQuality.BALANCED),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Procesar múltiples archivos en lote.
    Limitado a 10 archivos por petición.
    """
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Máximo 10 archivos por lote"
        )
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Procesar cada archivo
            content = await file.read()
            
            if file.content_type.startswith('image/'):
                image = Image.open(io.BytesIO(content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                text, confidence = ocr_processor.extract_text_from_image(image, quality)
            elif file.content_type == 'application/pdf':
                text, confidence = ocr_processor.extract_text_from_pdf(content, quality)
            else:
                errors.append({
                    "filename": file.filename,
                    "error": f"Tipo no soportado: {file.content_type}"
                })
                continue
            
            doc_type = ocr_processor.detect_document_type(text)
            extracted_data = ocr_processor.extract_structured_data(text, doc_type)
            
            results.append({
                "filename": file.filename,
                "document_type": doc_type.value,
                "extracted_data": extracted_data,
                "confidence": round(confidence, 2)
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "success": len(errors) == 0,
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@app.post("/ocr/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analizar documento sin extraer texto completo.
    Útil para obtener metadatos y vista previa rápida.
    """
    
    try:
        content = await file.read()
        
        analysis = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "size_mb": round(len(content) / 1024 / 1024, 2)
        }
        
        if file.content_type.startswith('image/'):
            # Analizar imagen
            image = Image.open(io.BytesIO(content))
            
            analysis.update({
                "type": "image",
                "dimensions": {
                    "width": image.width,
                    "height": image.height
                },
                "format": image.format,
                "mode": image.mode,
                "dpi": image.info.get('dpi', (72, 72)),
                "has_transparency": image.mode in ('RGBA', 'LA', 'P'),
                "estimated_quality": _estimate_image_quality(image)
            })
            
            # Vista previa rápida del texto (solo primera parte)
            text, confidence = ocr_processor.extract_text_from_image(
                image.crop((0, 0, image.width, min(500, image.height))),
                ProcessingQuality.FAST
            )
            
            analysis["preview"] = {
                "text": text[:500] + "..." if len(text) > 500 else text,
                "confidence": round(confidence, 2)
            }
            
        elif file.content_type == 'application/pdf':
            # Analizar PDF
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            
            analysis.update({
                "type": "pdf",
                "num_pages": len(pdf_reader.pages),
                "is_encrypted": pdf_reader.is_encrypted,
                "metadata": {
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "subject": pdf_reader.metadata.get('/Subject', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                } if pdf_reader.metadata else {}
            })
            
            # Intentar extraer texto nativo del PDF
            first_page_text = ""
            try:
                first_page_text = pdf_reader.pages[0].extract_text()[:500]
            except:
                pass
            
            analysis["has_text_layer"] = bool(first_page_text.strip())
            
            if first_page_text:
                analysis["preview"] = {
                    "text": first_page_text,
                    "source": "native_pdf"
                }
        
        # Detectar tipo de documento probable
        if "preview" in analysis and analysis["preview"]["text"]:
            doc_type = ocr_processor.detect_document_type(analysis["preview"]["text"])
            analysis["probable_document_type"] = doc_type.value
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando documento: {str(e)}")

def _estimate_image_quality(image: Image.Image) -> str:
    """Estimar calidad de imagen para OCR"""
    
    # Factores de calidad
    width, height = image.size
    resolution = width * height
    
    if resolution < 500000:  # < 0.5 MP
        return "low"
    elif resolution < 2000000:  # < 2 MP
        return "medium"
    else:
        return "high"

@app.get("/ocr/supported-languages")
async def get_supported_languages():
    """Obtener idiomas soportados para OCR"""
    
    try:
        result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, '--list-langs'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            languages = result.stdout.strip().split('\n')[1:]  # Skip header
            
            # Mapear códigos a nombres completos
            language_names = {
                'spa': 'Español',
                'eng': 'English',
                'fra': 'Français',
                'deu': 'Deutsch',
                'ita': 'Italiano',
                'por': 'Português',
                'cat': 'Català',
                'eus': 'Euskera',
                'glg': 'Galego'
            }
            
            supported = []
            for lang in languages:
                supported.append({
                    'code': lang,
                    'name': language_names.get(lang, lang),
                    'available': True
                })
            
            return {
                "default": TESSERACT_LANGUAGES,
                "supported": supported,
                "total": len(supported)
            }
        else:
            raise Exception("No se pudo obtener la lista de idiomas")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/ocr/cache")
async def clear_cache():
    """Limpiar caché de OCR"""
    
    if not ENABLE_CACHE:
        return {"message": "Caché deshabilitado"}
    
    try:
        import shutil
        
        # Contar archivos antes de limpiar
        files_count = len(list(CACHE_DIR.glob("*.json")))
        
        # Limpiar directorio
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        return {
            "success": True,
            "message": "Caché limpiado",
            "files_removed": files_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando caché: {str(e)}")

@app.post("/ocr/validate")
async def validate_document(file: UploadFile = File(...)):
    """
    Validar si un documento cumple con requisitos mínimos para facturación.
    """
    
    try:
        # Procesar documento
        content = await file.read()
        
        if file.content_type.startswith('image/'):
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text, confidence = ocr_processor.extract_text_from_image(image, ProcessingQuality.HIGH)
        elif file.content_type == 'application/pdf':
            text, confidence = ocr_processor.extract_text_from_pdf(content, ProcessingQuality.HIGH)
        else:
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado")
        
        # Detectar tipo y extraer datos
        doc_type = ocr_processor.detect_document_type(text)
        extracted = ocr_processor.extract_structured_data(text, doc_type)
        
        # Validaciones requeridas para factura española
        validations = {
            "has_tax_id": bool(extracted.get('identification', {}).get('tax_id')),
            "has_invoice_number": bool(extracted.get('identification', {}).get('invoice_number')),
            "has_date": bool(extracted.get('dates')),
            "has_total": bool(extracted.get('financial', {}).get('total')),
            "has_supplier": bool(extracted.get('identification', {}).get('supplier')),
            "has_vat_info": bool(extracted.get('financial', {}).get('vat_rate') or 
                                 extracted.get('financial', {}).get('vat_amount')),
        }
        
        # Campos recomendados
        recommendations = {
            "has_base_amount": bool(extracted.get('financial', {}).get('base_amount')),
            "has_payment_method": bool(extracted.get('financial', {}).get('payment_method')),
            "has_iban": bool(extracted.get('financial', {}).get('iban')),
            "has_email": bool(extracted.get('contacts', {}).get('email')),
            "has_items": bool(extracted.get('items')),
        }
        
        # Calcular validez
        required_fields = sum(validations.values())
        total_required = len(validations)
        is_valid = required_fields >= (total_required - 1)  # Permitir 1 campo faltante
        
        # Calcular completitud
        optional_fields = sum(recommendations.values())
        completeness = (required_fields + optional_fields) / (len(validations) + len(recommendations)) * 100
        
        return {
            "valid": is_valid,
            "document_type": doc_type.value,
            "completeness_percentage": round(completeness, 1),
            "validation_results": validations,
            "recommendations": recommendations,
            "missing_required": [k for k, v in validations.items() if not v],
            "missing_optional": [k for k, v in recommendations.items() if not v],
            "quality_metrics": {
                "ocr_confidence": round(confidence, 2),
                "extraction_quality": round(extracted['quality_score'], 2)
            },
            "extracted_summary": {
                "supplier": extracted.get('identification', {}).get('supplier'),
                "invoice_number": extracted.get('identification', {}).get('invoice_number'),
                "date": extracted.get('dates', [None])[0] if extracted.get('dates') else None,
                "total": extracted.get('financial', {}).get('total'),
                "tax_id": extracted.get('identification', {}).get('tax_id')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validando documento: {str(e)}")

@app.post("/ocr/debug-pdf")
async def debug_pdf(file: UploadFile = File(...)):
    """
    Debug específico para PDFs - analiza el contenido y la conversión a imágenes
    """
    
    if not file.content_type or file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Solo se admiten archivos PDF")
    
    try:
        content = await file.read()
        
        # Información básica del PDF
        pdf_info = {
            "size_bytes": len(content),
            "size_mb": round(len(content) / 1024 / 1024, 2)
        }
        
        # Intentar extraer texto nativo
        native_text = ocr_processor._extract_native_pdf_text(content)
        has_native_text = bool(native_text and len(native_text.strip()) > 10)
        
        pdf_info["has_native_text"] = has_native_text
        pdf_info["native_text_length"] = len(native_text) if native_text else 0
        pdf_info["native_text_preview"] = native_text[:200] + "..." if native_text and len(native_text) > 200 else native_text
        
        # Intentar conversión a imágenes
        conversion_results = {}
        
        for dpi in [150, 200, 300]:
            try:
                images = convert_from_bytes(
                    content,
                    first_page=1,
                    last_page=2,  # Solo primeras 2 páginas para debug
                    dpi=dpi,
                    fmt='PNG'
                )
                
                conversion_results[f"dpi_{dpi}"] = {
                    "success": True,
                    "pages_converted": len(images),
                    "image_sizes": [img.size for img in images[:2]] if images else []
                }
                
                # OCR rápido en primera página si existe
                if images:
                    first_page_text, confidence = ocr_processor.extract_text_from_image(
                        images[0], 
                        ProcessingQuality.FAST
                    )
                    conversion_results[f"dpi_{dpi}"]["ocr_preview"] = {
                        "text": first_page_text[:200] + "..." if len(first_page_text) > 200 else first_page_text,
                        "confidence": confidence,
                        "text_length": len(first_page_text)
                    }
                
            except Exception as e:
                conversion_results[f"dpi_{dpi}"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Recomendaciones
        recommendations = []
        
        if has_native_text:
            recommendations.append("✅ PDF tiene texto nativo - OCR no necesario")
        else:
            recommendations.append("⚠️ PDF es imagen escaneada - requiere OCR")
        
        if not any(result.get("success", False) for result in conversion_results.values()):
            recommendations.append("❌ Error convirtiendo PDF a imágenes")
        elif all(result.get("ocr_preview", {}).get("confidence", 0) < 50 
                for result in conversion_results.values() if result.get("success")):
            recommendations.append("⚠️ Baja confianza OCR - verificar calidad del escaneo")
        
        return {
            "filename": file.filename,
            "pdf_info": pdf_info,
            "conversion_results": conversion_results,
            "recommendations": recommendations,
            "best_approach": "native_text" if has_native_text else "ocr_high_dpi"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en debug PDF: {str(e)}")

@app.post("/ocr/debug")
async def debug_ocr(
    file: UploadFile = File(...),
    save_processed_image: bool = Query(False, description="Guardar imagen procesada para debug")
):
    """
    Endpoint de debug para diagnosticar problemas de OCR.
    Devuelve información detallada del procesamiento.
    """
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Solo se admiten imágenes para debug")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Información original de la imagen
        original_info = {
            "size": image.size,
            "mode": image.mode,
            "format": getattr(image, 'format', 'Unknown'),
            "has_transparency": image.mode in ('RGBA', 'LA', 'P')
        }
        
        # Convertir para procesamiento
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Probar diferentes calidades y configuraciones
        results = {}
        
        for quality in [ProcessingQuality.FAST, ProcessingQuality.BALANCED, ProcessingQuality.HIGH]:
            processed = ocr_processor.preprocess_image(opencv_image, quality)
            
            # Diferentes modos PSM
            for psm in [3, 6, 8, 11]:
                config_name = f"{quality.value}_psm{psm}"
                config = f'--oem 3 --psm {psm} -l {TESSERACT_LANGUAGES}'
                
                try:
                    text = pytesseract.image_to_string(processed, config=config)
                    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calcular estadísticas
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    results[config_name] = {
                        "text": text[:500] + "..." if len(text) > 500 else text,
                        "text_length": len(text),
                        "confidence": round(avg_confidence, 2),
                        "word_count": len([w for w in data['text'] if w.strip()]),
                        "config": config
                    }
                except Exception as e:
                    results[config_name] = {"error": str(e)}
        
        # Encontrar mejor resultado
        best_result = None
        best_score = 0
        
        for name, result in results.items():
            if 'error' not in result:
                # Score = confianza * longitud de texto
                score = result['confidence'] * result['text_length']
                if score > best_score:
                    best_score = score
                    best_result = name
        
        return {
            "filename": file.filename,
            "original_image": original_info,
            "processing_results": results,
            "best_configuration": best_result,
            "recommendations": {
                "image_too_small": image.size[0] < 500 or image.size[1] < 500,
                "low_confidence_detected": best_score < 1000,
                "suggested_improvements": [
                    "Aumentar resolución de la imagen" if image.size[0] < 500 else None,
                    "Mejorar iluminación y contraste" if best_score < 500 else None,
                    "Verificar que el texto esté derecho" if best_score < 800 else None
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en debug: {str(e)}")

@app.post("/ocr/process-utility-bill")
async def process_utility_bill(
    file: UploadFile = File(...),
    quality: ProcessingQuality = Query(ProcessingQuality.HIGH, description="Calidad de procesamiento (HIGH recomendado para facturas)"),
    extract_consumption_data: bool = Query(True, description="Extraer datos de consumo histórico")
):
    """
    Endpoint especializado para procesar facturas de servicios públicos (electricidad, gas).
    Optimizado para Naturgy, Endesa, Iberdrola, EDP, etc.
    """
    
    # Validar tipo de archivo
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Tipo de archivo no especificado")
    
    allowed_types = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'application/pdf']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}"
        )
    
    try:
        content = await file.read()
        
        # Validar tamaño
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        start_time = datetime.now()
        
        # Procesar imagen o PDF con alta calidad para mejor extracción
        if file.content_type.startswith('image/'):
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text, confidence = ocr_processor.extract_text_from_image(image, quality)
        elif file.content_type == 'application/pdf':
            text, confidence = ocr_processor.extract_text_from_pdf(content, quality)
        
        # Detectar tipo de documento
        doc_type = ocr_processor.detect_document_type(text)
        
        # Validar que es una factura de servicios
        if doc_type not in [DocumentType.ELECTRICITY_BILL, DocumentType.GAS_BILL]:
            # Intentar detectar si es factura de servicios aunque no se detectó inicialmente
            if any(word in text.lower() for word in ['cups', 'kwh', 'potencia', 'naturgy', 'endesa', 'iberdrola']):
                doc_type = DocumentType.ELECTRICITY_BILL
            else:
                logger.warning(f"Documento no es factura de servicios: {doc_type}")
        
        # Extraer datos especializados
        extracted_data = ocr_processor.extract_electricity_bill_data(text, doc_type)
        
        # Validación específica para facturas de servicios
        validation = {
            'has_cups': bool(extracted_data.get('supply_info', {}).get('cups')),
            'has_invoice_number': bool(extracted_data.get('billing_info', {}).get('invoice_number')),
            'has_period': bool(
                extracted_data.get('billing_info', {}).get('period_start') and 
                extracted_data.get('billing_info', {}).get('period_end')
            ),
            'has_total': bool(extracted_data.get('amounts', {}).get('total')),
            'has_breakdown': bool(
                extracted_data.get('amounts', {}).get('electricity') or 
                extracted_data.get('amounts', {}).get('gas')
            ),
            'has_vat': bool(extracted_data.get('amounts', {}).get('vat')),
            'supplier_detected': extracted_data.get('supplier') != 'unknown'
        }
        
        # Calcular validez
        is_valid = sum([
            validation['has_invoice_number'],
            validation['has_total'],
            validation['has_period'] or validation['has_cups']  # Al menos uno de estos
        ]) >= 2
        
        # Tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Preparar respuesta con formato específico para facturas de servicios
        result = {
            "success": True,
            "filename": file.filename,
            "document_type": doc_type.value,
            "supplier": extracted_data.get('supplier', 'unknown'),
            "validation": {
                "is_valid": is_valid,
                "details": validation,
                "quality_score": extracted_data.get('quality_score', 0),
                "missing_fields": [k for k, v in validation.items() if not v]
            },
            "billing_information": {
                "invoice_number": extracted_data.get('billing_info', {}).get('invoice_number'),
                "period_start": extracted_data.get('billing_info', {}).get('period_start'),
                "period_end": extracted_data.get('billing_info', {}).get('period_end'),
                "cups": extracted_data.get('supply_info', {}).get('cups')
            },
            "amounts": {
                "total": extracted_data.get('amounts', {}).get('total'),
                "electricity": extracted_data.get('amounts', {}).get('electricity'),
                "gas": extracted_data.get('amounts', {}).get('gas'),
                "vat": extracted_data.get('amounts', {}).get('vat'),
                "services": extracted_data.get('amounts', {}).get('services'),
                "base_amount": extracted_data.get('financial', {}).get('base_amount')
            },
            "consumption": extracted_data.get('consumption', {}),
            "metrics": {
                "ocr_confidence": round(confidence, 2),
                "extraction_quality": round(extracted_data.get('quality_score', 0), 2),
                "processing_time": round(processing_time, 3),
                "quality_setting": quality.value
            },
            "raw_text": text if len(text) < 5000 else text[:5000] + "...",
            "full_extraction": extracted_data
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando factura de servicios: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando factura: {str(e)}")

# Middleware para logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    
    # Procesar request
    response = await call_next(request)
    
    # Log de la petición
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Manejadores de errores
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "type": "validation_error"
        }
    )

# Cleanup al cerrar
@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar la aplicación"""
    executor.shutdown(wait=True)
    logger.info("OCR Service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    # Configuración de desarrollo
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
    