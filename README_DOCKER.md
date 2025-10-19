# 🐳 RAG Hukuki Chatbot - Docker Deployment

## 🚀 Hızlı Başlangıç

### 1. **Docker ile Çalıştırma**

```bash
# Basit çalıştırma
./docker-run.sh

# Veya manuel olarak
docker-compose up -d
```

### 2. **Erişim**

- **Web Arayüzü:** http://localhost:5001
- **API Health:** http://localhost:5001/health
- **API Models:** http://localhost:5001/api/models

## 🔧 Docker Yapılandırması

### **Dockerfile Özellikleri:**
- ✅ Python 3.11-slim base image
- ✅ Tüm bağımlılıklar otomatik yüklenir
- ✅ M4 Mac OpenMP fix dahil
- ✅ Cache dizini otomatik oluşturulur
- ✅ Port 5001 expose edilir

### **docker-compose.yml Özellikleri:**
- ✅ Volume mounting (cache, data)
- ✅ Environment variables
- ✅ Health check
- ✅ Auto-restart
- ✅ Port mapping

## 📁 Dosya Yapısı

```
.
├── Dockerfile              # Docker image tanımı
├── docker-compose.yml      # Docker Compose yapılandırması
├── .dockerignore           # Docker ignore dosyası
├── docker-run.sh          # Otomatik çalıştırma scripti
├── requirements_rag.txt    # Python bağımlılıkları
├── config.yaml            # Uygulama yapılandırması
└── app_rag.py             # Ana uygulama
```

## 🛠️ Gelişmiş Kullanım

### **Environment Variables:**

```bash
# .env dosyası oluştur
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

### **Volume Mounting:**

```yaml
volumes:
  - ./cache:/app/cache      # Cache dosyaları
  - ./data:/app/data        # PDF verileri
```

### **Health Check:**

```bash
# Container durumunu kontrol et
docker-compose ps

# Logları görüntüle
docker-compose logs -f rag-chatbot

# Health check
curl http://localhost:5001/health
```

## 🔍 Troubleshooting

### **Container Başlamıyor:**
```bash
# Logları kontrol et
docker-compose logs rag-chatbot

# Container'ı yeniden başlat
docker-compose restart rag-chatbot
```

### **Model Yüklenmiyor:**
```bash
# Cache dizinini kontrol et
ls -la cache/

# Container'ı temizle ve yeniden build et
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### **Port Çakışması:**
```bash
# Farklı port kullan
docker-compose up -d --scale rag-chatbot=0
docker-compose up -d -p 5002:5001
```

## 📊 Performans

### **RAM Kullanımı:**
- **Llama-3.1-8B:** ~9GB
- **E5-Large:** ~2GB
- **Toplam:** ~12GB

### **Disk Kullanımı:**
- **Base Image:** ~1GB
- **Models:** ~12GB
- **Cache:** Değişken

## 🌐 Production Deployment

### **Docker Swarm:**
```bash
# Swarm mode'da çalıştır
docker stack deploy -c docker-compose.yml rag-stack
```

### **Kubernetes:**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-chatbot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rag-chatbot
  template:
    metadata:
      labels:
        app: rag-chatbot
    spec:
      containers:
      - name: rag-chatbot
        image: law-rag-chatbot:latest
        ports:
        - containerPort: 5001
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
```

## 🎯 Avantajlar

- ✅ **Taşınabilir:** Herhangi bir sistemde çalışır
- ✅ **İzole:** Sistem bağımlılıkları yok
- ✅ **Ölçeklenebilir:** Docker Swarm/K8s ile
- ✅ **Güvenli:** Container izolasyonu
- ✅ **Kolay:** Tek komutla çalıştırma
