# 🏗️ Project Structure - OOP Design

## 📁 Folder Organization

```
nlp/
├── 📂 core/                     # Core components & configuration
│   ├── __init__.py             # Core module exports
│   ├── config.py               # Configuration management (Singleton)
│   │
│   ├── 📂 models/              # Model factories
│   │   ├── __init__.py         # Model exports
│   │   └── factory.py          # Factory pattern for models
│   │
│   └── 📂 strategies/          # Retrieval strategies
│       ├── __init__.py         # Strategy exports
│       └── retrieval.py        # Strategy pattern implementations
│
├── 📂 services/                 # Business logic layer
│   ├── __init__.py             # Service exports
│   ├── rag_engine.py           # Main RAG orchestrator
│   └── pdf_processor.py        # PDF processing service
│
├── 📂 static/                   # Frontend assets
│   ├── css/                    # Stylesheets
│   └── js/                     # JavaScript
│
├── 📂 templates/                # HTML templates
│   └── index_chatgpt.html      # Main UI
│
├── 📂 docs/                     # PDF documents (user data)
│   └── *.pdf                   # Your PDF files here
│
├── 📂 cache/                    # Model & data cache
│   ├── chunks.pkl              # Processed chunks
│   ├── faiss.index             # Vector index
│   ├── bm25.pkl                # BM25 index
│   └── */                      # Model caches
│
├── 📄 config.yaml              # User configuration
├── 📄 app_rag.py               # Flask application (entry point)
├── 📄 requirements_rag.txt     # Python dependencies
├── 📄 README.md                # Main documentation
└── 📄 .env                     # Environment variables (HF_TOKEN)
```

---

## 🎯 Design Principles

### 1. **Separation of Concerns**
Each module has a specific responsibility:
- `core/` → Configuration & base components
- `services/` → Business logic
- `static/` & `templates/` → Presentation
- `docs/` → User data

### 2. **Dependency Flow**
```
app_rag.py
    ↓
core/ (config)
    ↓
services/ (business logic)
    ↓
core/models/ & core/strategies/ (implementations)
```

### 3. **No Circular Dependencies**
- Core doesn't import from services
- Services can import from core
- App imports from both

---

## 📦 Module Descriptions

### `core/` - Configuration & Base Components
**Responsibility:** Configuration management, base classes, design patterns

**Files:**
- `config.py` - Singleton configuration, dataclasses, model catalog
- `models/factory.py` - Factory pattern for creating models
- `strategies/retrieval.py` - Strategy pattern for retrieval algorithms

**Key Classes:**
- `RAGConfig` (Singleton)
- `ModelManager` (Singleton)
- `EmbeddingModelFactory`, `LLMModelFactory`, `RerankerModelFactory`
- `RetrievalStrategy` and implementations

---

### `services/` - Business Logic
**Responsibility:** Core business logic, orchestration

**Files:**
- `rag_engine.py` - Main RAG orchestrator, answer generation
- `pdf_processor.py` - PDF parsing, chunking

**Key Classes:**
- `RAGEngine` - Orchestrates retrieval and generation
- `PDFProcessor` - Processes PDF documents

---

### `static/` & `templates/` - Presentation Layer
**Responsibility:** User interface

**Files:**
- `templates/index_chatgpt.html` - Main UI
- `static/css/` - Styles
- `static/js/` - Frontend logic (streaming, chat, etc.)

---

## 🔄 Data Flow

```
1. User Query (Frontend)
       ↓
2. Flask App (app_rag.py)
       ↓
3. RAG Engine (services/rag_engine.py)
       ↓
4. Retrieval Strategy (core/strategies/retrieval.py)
       ↓
5. Models (core/models/factory.py)
       ↓
6. Response Generation
       ↓
7. Streaming to Frontend
```

---

## ⚙️ Configuration Flow

```
1. config.yaml (User edits)
       ↓
2. RAGConfig loads (core/config.py)
       ↓
3. Validates settings
       ↓
4. Provides to RAG Engine
       ↓
5. Used throughout application
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
# Test configuration
test_core_config.py

# Test factories
test_models_factory.py

# Test strategies
test_retrieval_strategy.py

# Test services
test_rag_engine.py
test_pdf_processor.py
```

### Integration Tests
```python
# Test full pipeline
test_integration.py
```

---

## 🚀 Development Workflow

### 1. **Add New Model**
Edit: `core/config.py` → Add to catalog
Use: Edit `config.yaml`

### 2. **Add New Strategy**
Create: `core/strategies/retrieval.py` → New class
Register: `RetrievalStrategyFactory`

### 3. **Modify Business Logic**
Edit: `services/rag_engine.py`
Keep: Dependency injection pattern

### 4. **Change UI**
Edit: `templates/`, `static/`
Keep: API contract stable

---

## 📊 Code Statistics

| Directory | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| `core/` | 6 | ~600 | Config & patterns |
| `services/` | 3 | ~1200 | Business logic |
| `static/` | 3+ | ~800 | Frontend |
| `templates/` | 1 | ~200 | UI |
| **Total** | **13+** | **~2800** | **Clean code!** |

---

## 🎨 Design Patterns Map

| Pattern | Location | Usage |
|---------|----------|-------|
| **Singleton** | `core/config.py` | RAGConfig, ModelManager |
| **Factory** | `core/models/factory.py` | Model creation |
| **Strategy** | `core/strategies/retrieval.py` | Retrieval algorithms |
| **Decorator** | `core/strategies/retrieval.py` | Feature composition |
| **Dependency Injection** | Everywhere | Loose coupling |

---

## 🔧 Import Conventions

### ✅ Good Imports
```python
# From core
from core import get_config
from core.models import ModelManager
from core.strategies import RetrievalStrategyFactory

# From services
from services import RAGEngine, PDFProcessor
```

### ❌ Avoid
```python
# Don't import internals directly
from core.config import RAGConfig  # Use get_config() instead
```

---

## 📝 Adding New Features

### Example: Add New Embedding Model

**Step 1:** Edit `core/config.py`
```python
embedding_models = {
    'new-model': EmbeddingModelConfig(
        name='org/new-model',
        display_name='New Model',
        size_gb=1.5,
        quality_score=9.0,
        dimension=768
    )
}
```

**Step 2:** Edit `config.yaml`
```yaml
active_models:
  embedding: new-model
```

**Step 3:** Run
```bash
python3 app_rag.py
```

**No code changes needed!** ✅

---

## 🎯 Benefits of This Structure

1. **Modularity** - Each module is independent
2. **Testability** - Easy to test in isolation
3. **Maintainability** - Clear responsibility
4. **Scalability** - Easy to add features
5. **Readability** - Logical organization
6. **Reusability** - Components are reusable

---

## 🔒 File Permissions

```bash
# Configuration (user-editable)
config.yaml          # Read/Write by user
.env                 # Read/Write by user (contains secrets)

# Code (version controlled)
*.py                 # Read by user, edited by devs
*.md                 # Read by user

# Data (generated)
cache/*              # Read/Write by app (can delete)
docs/*               # Read/Write by user (PDF files)
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation, quick start |
| `STRUCTURE.md` | This file - project structure |

**All other MD files removed!** Clean and focused. ✅

---

**This structure follows industry best practices and SOLID principles!**

