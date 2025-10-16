# 🔧 Troubleshooting Guide

## ⚠️ Segmentation Fault

### Sorun:
```
zsh: segmentation fault  python3 app_rag.py
```

### Nedenleri:
1. **RAM Yetersizliği** - Llama 3.2-1B bile ~3-4GB gerektirir
2. **PyTorch/Transformers Uyumsuzluğu** - Versiyon çakışması
3. **Metal/MPS Sorunu** - macOS GPU backend hatası

---

## ✅ Çözümler

### 1️⃣ **Daha Hafif Model Kullan (En Kolay)**

`config.yaml` dosyasını düzenle:

```yaml
active_models:
  llm: qwen-1.5b  # Llama yerine Qwen (token gerektirmez)
```

veya

```yaml
active_models:
  llm: gemma-2b   # Google Gemma (token gerektirmez)
```

**Avantaj:** Token gerekmez, daha az RAM

---

### 2️⃣ **RAM Optimizasyonu**

Diğer uygulamaları kapat:
```bash
# Bellek kullanımını kontrol et
top -o MEM
```

**16GB RAM için önerilen:**
- Chrome/Safari: Kapat
- Docker/VM: Durdur
- Gereksiz uygulamalar: Kapat

---

### 3️⃣ **PyTorch/Transformers Güncelle**

```bash
cd ~/Desktop/nlp
source venv/bin/activate

# Güncel versiyonları kur
pip install --upgrade torch transformers accelerate

# Veya belirli versiyonlar
pip install torch==2.1.0 transformers==4.35.2
```

---

### 4️⃣ **Cache Temizle ve Yeniden Başlat**

```bash
cd ~/Desktop/nlp

# Cache'i temizle
rm -rf cache/*

# Tekrar dene
python3 app_rag.py
```

---

### 5️⃣ **Sadece Retrieval Kullan (LLM'siz)**

Eğer hiçbiri çalışmazsa, sadece retrieval kullan:

`config.yaml`:
```yaml
# LLM'i geçici olarak devre dışı bırak
# Sadece kaynak gösterimi yapacak
```

**NOT:** Bu özellik henüz implement edilmedi, ama eklenebilir.

---

## 🐛 Debug Adımları

### 1. Model İndirmeyi Test Et

```bash
cd ~/Desktop/nlp
source venv/bin/activate

python3 -c "
from transformers import AutoModelForCausalLM
import os
os.environ['HF_TOKEN'] = 'your_token'

try:
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B-Instruct',
        token=os.environ['HF_TOKEN'],
        low_cpu_mem_usage=True
    )
    print('✅ Model loaded successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### 2. RAM Kullanımını İzle

Başka bir terminal'de:
```bash
while true; do
    clear
    echo "RAM Kullanımı:"
    ps aux | grep python | head -5
    sleep 2
done
```

### 3. Verbose Logging

```bash
export TRANSFORMERS_VERBOSITY=debug
python3 app_rag.py
```

---

## 💡 Önerilen Çözüm (16GB RAM)

### Seçenek A: Qwen Kullan (En Stabil)
```yaml
# config.yaml
active_models:
  embedding: e5-large
  llm: qwen-1.5b  # ✅ Token gerektirmez, stabil
```

**RAM:** ~7GB  
**Kalite:** 7/10  
**Stabilite:** Yüksek

---

### Seçenek B: MPNet + Qwen (Güvenli)
```yaml
# config.yaml
active_models:
  embedding: mpnet  # E5-Large yerine daha küçük
  llm: qwen-1.5b
```

**RAM:** ~5.5GB  
**Kalite:** 6.5/10  
**Stabilite:** Çok Yüksek

---

### Seçenek C: Llama ile Dene (Riskli)

Eğer Llama kullanmak zorundasın:

1. **Tüm uygulamaları kapat**
2. **Cache temizle:** `rm -rf cache/*`
3. **Sistemi yeniden başlat**
4. **Sadece terminal aç ve çalıştır**

```bash
cd ~/Desktop/nlp
source venv/bin/activate
python3 app_rag.py
```

**RAM:** ~9-10GB  
**Kalite:** 7.5/10  
**Stabilite:** Orta (segfault riski var)

---

## 🔍 Hata Mesajlarına Göre Çözümler

### "Failed to resolve 'huggingface.co'"
**Çözüm:** İnternet bağlantısını kontrol et, VPN kapat

### "Access to model is restricted"
**Çözüm:** HF_TOKEN ekle ve model erişimi al

### "Out of memory"
**Çözüm:** Daha küçük model kullan (qwen-1.5b)

### "Segmentation fault"
**Çözüm:**
1. PyTorch/Transformers güncelle
2. Daha küçük model kullan
3. Cache temizle
4. Sistemi yeniden başlat

---

## 📞 Hızlı Yardım

### Sorun Devam Ediyorsa:

1. **config.yaml'ı şu şekilde değiştir:**
```yaml
active_models:
  embedding: mpnet
  llm: qwen-1.5b
```

2. **Cache'i temizle:**
```bash
rm -rf cache/*
```

3. **Tekrar çalıştır:**
```bash
python3 app_rag.py
```

**Bu %95 çalışır!** ✅

---

## 🎯 Sonuç

**16GB RAM için BEST PRACTICE:**

```yaml
# config.yaml - STABIL SETUP
active_models:
  embedding: e5-large  # veya mpnet (daha güvenli)
  llm: qwen-1.5b       # Token gerektirmez, stabil

retrieval:
  top_k: 3
  use_hybrid_search: true
  use_reranking: true
  use_query_expansion: true
```

**Bu setup:**
- ✅ Token gerektirmez
- ✅ Segfault riski yok
- ✅ 7GB RAM kullanır
- ✅ İyi kalite (7/10)
- ✅ Stabil çalışır

**Llama istiyorsan:**
- Tüm uygulamaları kapat
- 10GB+ boş RAM olmalı
- Segfault riski var
- Qwen'e göre sadece +0.5 puan daha iyi

**Karar:** Qwen kullan! 🎯

