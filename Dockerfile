# RAG Hukuki Chatbot Docker Image
FROM python:3.11-slim

# Sistem paketlerini güncelle ve gerekli paketleri yükle
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    zlib1g-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Python bağımlılıklarını kopyala ve yükle
COPY requirements_rag.txt .
RUN pip install --no-cache-dir -r requirements_rag.txt

# Uygulama dosyalarını kopyala
COPY . .

# Cache dizinini oluştur
RUN mkdir -p cache

# Port'u expose et
EXPOSE 5001

# M4 Mac OpenMP fix
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV MKL_SERVICE_FORCE_INTEL=1

# Uygulamayı başlat
CMD ["python", "app_rag.py"]
