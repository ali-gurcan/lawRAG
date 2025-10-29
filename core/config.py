"""
Configuration management for RAG system
Implements proper OOP design with singleton pattern
"""
import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    display_name: str
    size_gb: float
    quality_score: float
    requires_token: bool = False
    device: str = 'cpu'
    dtype: str = 'float32'
    max_length: int = 512


@dataclass
class EmbeddingModelConfig(ModelConfig):
    """Embedding model specific configuration"""
    dimension: int = 384
    batch_size: int = 64
    
    
@dataclass
class LLMModelConfig(ModelConfig):
    """LLM specific configuration"""
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 3
    use_hybrid_search: bool = True
    use_reranking: bool = True
    use_query_expansion: bool = True
    hybrid_alpha: float = 0.7  # Weight for dense retrieval (0-1)
    reranking_beta: float = 0.6  # Weight for reranker (0-1)
    retrieval_k_multiplier: int = 3  # Get k*multiplier candidates for reranking


@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    use_article_chunking: bool = True  # For legal documents
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


@dataclass
class CacheConfig:
    """Cache configuration"""
    cache_dir: str = 'cache'
    chunks_file: str = 'chunks.pkl'
    index_file: str = 'faiss.index'
    bm25_file: str = 'bm25.pkl'
    embeddings_file: str = 'embeddings.pkl'
    enable_cache: bool = True
    force_refresh: bool = False


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = '0.0.0.0'
    port: int = 5001
    debug: bool = False
    threaded: bool = True


@dataclass
class ConfidenceConfig:
    """Confidence scoring configuration"""
    low_threshold: int = 60  # Below this is low confidence
    medium_threshold: int = 80  # Above this is high confidence
    enable_warnings: bool = True
    calculation_method: str = 'weighted_top'  # average, weighted_top, top_score_only
    top_score_weight: float = 0.6  # Weight for top score in weighted_top method
    boost_high_scores: bool = True  # Boost if top score > 0.85


@dataclass
class StreamingConfig:
    """Streaming configuration"""
    enable_streaming: bool = True
    chunk_size: int = 1  # Tokens per chunk
    buffer_size: int = 5


@dataclass
class LLMGenerationConfig:
    """LLM generation configuration"""
    max_tokens: int = 50
    context_limit: int = 500  # Max characters sent to LLM
    length_budget: int = 2000  # Max characters for combined context
    max_answer_words: int = 50  # Max words in answer (prompt guidance)


class RAGConfig:
    """
    Main RAG configuration with singleton pattern
    Loads from config.yaml or uses defaults
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
        
        self.config_path = config_path or 'config.yaml'
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_path):
            print(f"📄 Loading config from {self.config_path}")
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            self._apply_config(config_data)
        else:
            print(f"⚙️  Using default configuration")
            self._use_defaults()
    
    def _use_defaults(self):
        """Use default configuration"""
        # Embedding models catalog (only BGE-M3 kept)
        self.embedding_models = {
            'bge-m3': EmbeddingModelConfig(
                name='BAAI/bge-m3',
                display_name='BGE-M3',
                size_gb=2.0,
                quality_score=9.6,
                dimension=1024,
                batch_size=16,
                max_length=8192
            )
        }
        
        # LLM models catalog (only Llama-3.1-8B kept)
        self.llm_models = {
            'llama-8b': LLMModelConfig(
                name='meta-llama/Llama-3.1-8B-Instruct',
                display_name='Llama-3.1-8B',
                size_gb=9.0,
                quality_score=9.2,
                requires_token=True,
                device='cpu',
                dtype='float32',
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            )
        }
        
        # Reranker models
        self.reranker_models = {
            'ms-marco': ModelConfig(
                name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                display_name='MS-MARCO MiniLM',
                size_gb=0.5,
                quality_score=8.5
            )
        }
        
        # Active models (selected from catalog)
        self.active_embedding_model = 'bge-m3'
        self.active_llm_model = 'llama-8b'
        self.active_reranker_model = 'ms-marco'
        
        # Other configs
        self.retrieval = RetrievalConfig()
        self.chunking = ChunkingConfig()
        self.cache = CacheConfig()
        self.server = ServerConfig()
        self.confidence = ConfidenceConfig()
        self.streaming = StreamingConfig()
        self.llm_generation = LLMGenerationConfig()
        
        # Environment
        self.hf_token = os.getenv('HF_TOKEN')
        self.docs_dir = 'docs'
        self.templates_dir = 'templates'
        self.static_dir = 'static'
    
    def _apply_config(self, config_data: Dict[str, Any]):
        """Apply configuration from loaded data"""
        self._use_defaults()  # Start with defaults
        
        # Override with loaded config
        if 'active_models' in config_data:
            self.active_embedding_model = config_data['active_models'].get(
                'embedding', self.active_embedding_model
            )
            self.active_llm_model = config_data['active_models'].get(
                'llm', self.active_llm_model
            )
            self.active_reranker_model = config_data['active_models'].get(
                'reranker', self.active_reranker_model
            )
        
        # Override retrieval config
        if 'retrieval' in config_data:
            for key, value in config_data['retrieval'].items():
                if hasattr(self.retrieval, key):
                    setattr(self.retrieval, key, value)
        
        # Override other configs similarly
        for config_name in ['chunking', 'cache', 'server', 'confidence', 'streaming', 'llm_generation']:
            if config_name in config_data:
                config_obj = getattr(self, config_name)
                for key, value in config_data[config_name].items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def get_embedding_model(self) -> EmbeddingModelConfig:
        """Get active embedding model config"""
        return self.embedding_models[self.active_embedding_model]
    
    def get_llm_model(self) -> LLMModelConfig:
        """Get active LLM model config"""
        return self.llm_models[self.active_llm_model]
    
    def get_reranker_model(self) -> ModelConfig:
        """Get active reranker model config"""
        return self.reranker_models[self.active_reranker_model]
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check if active models exist
        if self.active_embedding_model not in self.embedding_models:
            errors.append(f"Invalid embedding model: {self.active_embedding_model}")
        
        if self.active_llm_model not in self.llm_models:
            errors.append(f"Invalid LLM model: {self.active_llm_model}")
        
        # Check if required token exists for gated models
        llm_config = self.get_llm_model()
        if llm_config.requires_token and not self.hf_token:
            errors.append(f"Model {llm_config.name} requires HF_TOKEN but not found")
        
        # Check directories
        if not os.path.exists(self.docs_dir):
            print(f"⚠️  Warning: {self.docs_dir}/ not found, creating...")
            os.makedirs(self.docs_dir, exist_ok=True)
        
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   • {error}")
            return False
        
        return True
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file"""
        path = path or self.config_path
        
        config_data = {
            'active_models': {
                'embedding': self.active_embedding_model,
                'llm': self.active_llm_model,
                'reranker': self.active_reranker_model
            },
            'retrieval': asdict(self.retrieval),
            'chunking': asdict(self.chunking),
            'cache': asdict(self.cache),
            'server': asdict(self.server),
            'confidence': asdict(self.confidence),
            'streaming': asdict(self.streaming)
        }
        
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False)
            else:
                json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved to {path}")
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("⚙️  RAG SYSTEM CONFIGURATION")
        print("="*70)
        
        emb = self.get_embedding_model()
        llm = self.get_llm_model()
        
        print(f"\n🔍 EMBEDDING MODEL:")
        print(f"   Name: {emb.display_name}")
        print(f"   Size: {emb.size_gb}GB")
        print(f"   Quality: {emb.quality_score}/10")
        print(f"   Dimension: {emb.dimension}")
        
        print(f"\n🦙 LLM MODEL:")
        print(f"   Name: {llm.display_name}")
        print(f"   Size: {llm.size_gb}GB")
        print(f"   Quality: {llm.quality_score}/10")
        print(f"   Token Required: {'Yes' if llm.requires_token else 'No'}")
        
        print(f"\n🎯 RETRIEVAL:")
        print(f"   Top-K: {self.retrieval.top_k}")
        print(f"   Hybrid Search: {'ON' if self.retrieval.use_hybrid_search else 'OFF'}")
        print(f"   Reranking: {'ON' if self.retrieval.use_reranking else 'OFF'}")
        print(f"   Query Expansion: {'ON' if self.retrieval.use_query_expansion else 'OFF'}")
        
        print(f"\n📊 CHUNKING:")
        print(f"   Chunk Size: {self.chunking.chunk_size}")
        print(f"   Overlap: {self.chunking.chunk_overlap}")
        print(f"   Article-based: {'ON' if self.chunking.use_article_chunking else 'OFF'}")
        
        print(f"\n💾 CACHE:")
        print(f"   Directory: {self.cache.cache_dir}")
        print(f"   Enabled: {'Yes' if self.cache.enable_cache else 'No'}")
        
        print(f"\n🌐 SERVER:")
        print(f"   Host: {self.server.host}")
        print(f"   Port: {self.server.port}")
        
        total_ram = emb.size_gb + llm.size_gb + 0.5  # + overhead
        print(f"\n💻 ESTIMATED RAM: ~{total_ram:.1f}GB")
        
        print("="*70 + "\n")


# Global config instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> RAGConfig:
    """Get configuration singleton"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig(config_path)
    return _config_instance

