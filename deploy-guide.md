# 🚀 Guía de Despliegue - Servicio OCR

> **Nota**: Esta guía está actualizada para la versión 2.1.0. Para la documentación completa, ver [README.md](README.md).

## 📋 Requisitos del VPS

- **OS**: Ubuntu 20.04+ o Debian 11+
- **RAM**: Mínimo 2GB (recomendado 4GB para OCR intensivo)
- **CPU**: 1-2 cores
- **Storage**: 10GB para aplicación + cache
- **Docker**: Versión 20.10+
- **Puertos**: 8001 disponible

## 🐳 Despliegue con Docker (Recomendado)

### 1. Preparar el VPS

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Instalar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reiniciar sesión para aplicar cambios de grupo
```

### 2. Subir archivos al VPS

```bash
# Desde tu máquina local
scp -r . usuario@tu-vps-ip:/home/usuario/ocr-service
```

### 3. Desplegar en el VPS

```bash
# En el VPS
cd /home/usuario/ocr-service

# Construir y levantar el servicio
docker-compose up -d --build

# Verificar que funciona
curl http://localhost:8001/health
```

## Opción 2: Con Nginx y SSL (Producción)

### 1. Configurar Nginx como proxy reverso

Instalar Nginx:

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
```

### 2. Configurar dominio

Crear archivo de configuración:

```bash
sudo nano /etc/nginx/sites-available/ocr-service
```

### 3. Obtener certificado SSL

```bash
sudo certbot --nginx -d tu-dominio.com
```

## Opción 3: Con systemd (Sin Docker)

### 1. Instalar dependencias del sistema

```bash
sudo apt install python3-pip python3-venv tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng poppler-utils -y
```

### 2. Configurar entorno virtual

```bash
cd /opt
sudo mkdir ocr-service
sudo chown $USER:$USER ocr-service
cd ocr-service

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Monitoreo y Logs

### Ver logs del contenedor

```bash
docker-compose logs -f cakely-ocr
```

### Monitorear recursos

```bash
docker stats cakely-ocr-service
```

## Variables de Entorno de Producción

Crear archivo `.env`:

```env
ENVIRONMENT=production
LOG_LEVEL=info
MAX_FILE_SIZE=10MB
ALLOWED_ORIGINS=https://tu-dominio.com
```

## Backup y Mantenimiento

### Backup automático

```bash
# Crear script de backup
#!/bin/bash
docker save cakely-ocr-service > /backup/ocr-service-$(date +%Y%m%d).tar
```

### Actualización

```bash
cd /home/usuario/ocr-service
git pull origin main  # Si usas git
docker-compose down
docker-compose up -d --build
```

## Seguridad

1. **Firewall**: Configurar UFW
2. **Límites de recursos**: Añadir límites al docker-compose
3. **Rate limiting**: Implementar en Nginx
4. **Validación de archivos**: Mejorar validaciones

## Testing en Producción

```bash
# Test básico
curl -X POST "http://tu-vps-ip:8001/ocr/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-image.jpg"
```
