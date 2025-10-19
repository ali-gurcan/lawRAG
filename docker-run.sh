#!/bin/bash

# RAG Hukuki Chatbot Docker Runner
echo "🚀 RAG Hukuki Chatbot Docker Başlatılıyor..."

# .env dosyasını kontrol et
if [ ! -f .env ]; then
    echo "⚠️  .env dosyası bulunamadı!"
    echo "📝 .env dosyası oluşturuluyor..."
    echo "HF_TOKEN=your_huggingface_token_here" > .env
    echo "✅ .env dosyası oluşturuldu. Lütfen HF_TOKEN'ı güncelleyin."
    exit 1
fi

# Cache dizinini oluştur
mkdir -p cache
mkdir -p data

# Docker image'ı build et
echo "🔨 Docker image build ediliyor..."
docker build -t law-rag-chatbot .

# Container'ı çalıştır
echo "🐳 Container başlatılıyor..."
docker-compose up -d

# Container durumunu kontrol et
echo "📊 Container durumu:"
docker-compose ps

# Logları göster
echo "📋 Loglar:"
docker-compose logs -f rag-chatbot
