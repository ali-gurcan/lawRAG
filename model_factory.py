"""
Model Factory - Creates models based on configuration
Implements Factory Pattern for clean OOP design
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Any
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import ModelConfig, EmbeddingModelConfig, LLMModelConfig


class ModelFactory(ABC):
    """Abstract factory for creating models"""
    
    @abstractmethod
    def create_model(self, config: ModelConfig, **kwargs) -> Any:
        """Create model based on configuration"""
        pass


class EmbeddingModelFactory(ModelFactory):
    """Factory for creating embedding models"""
    
    def create_model(self, config: EmbeddingModelConfig, cache_dir: str = 'cache', **kwargs) -> Any:
        """
        Create embedding model
        
        Args:
            config: Embedding model configuration
            cache_dir: Cache directory for models
            
        Returns:
            Loaded embedding model
        """
        print(f"      📥 Loading embedding model: {config.display_name}")
        print(f"         Model: {config.name}")
        print(f"         Size: {config.size_gb}GB")
        print(f"         Quality: {config.quality_score}/10")
        
        try:
            # Create cache path
            model_cache = os.path.join(cache_dir, 'sentence_transformers')
            os.makedirs(model_cache, exist_ok=True)
            
            # Check for special models (BGE-M3)
            if 'bge-m3' in config.name.lower():
                print(f"      🚀 Using FlagEmbedding for BGE-M3...")
                model = BGEM3FlagModel(
                    config.name,
                    use_fp16=True
                )
                model.is_bge_m3 = True
            else:
                # Standard SentenceTransformer
                model = SentenceTransformer(
                    config.name,
                    cache_folder=model_cache
                )
                model.is_bge_m3 = False
            
            # Set batch size
            if hasattr(model, 'encode'):
                model.batch_size = config.batch_size
            
            print(f"      ✅ Embedding model loaded! (Dim: {config.dimension})")
            return model
            
        except Exception as e:
            print(f"      ❌ Failed to load embedding model: {str(e)}")
            raise


class LLMModelFactory(ModelFactory):
    """Factory for creating LLM models"""
    
    def create_model(
        self, 
        config: LLMModelConfig, 
        cache_dir: str = 'cache',
        hf_token: Optional[str] = None,
        **kwargs
    ) -> tuple:
        """
        Create LLM model and tokenizer
        
        Args:
            config: LLM model configuration
            cache_dir: Cache directory
            hf_token: Hugging Face token for gated models
            
        Returns:
            (tokenizer, model) tuple
        """
        print(f"\n      🤖 Loading LLM: {config.display_name}")
        print(f"         Model: {config.name}")
        print(f"         Size: {config.size_gb}GB")
        print(f"         Quality: {config.quality_score}/10")
        
        # Check token requirement
        if config.requires_token and not hf_token:
            raise ValueError(
                f"Model {config.name} requires HF_TOKEN but not provided.\n"
                f"Please set HF_TOKEN environment variable."
            )
        
        try:
            llm_cache = os.path.join(cache_dir, 'llm_model')
            os.makedirs(llm_cache, exist_ok=True)
            
            token_kwargs = {'token': hf_token} if hf_token else {}
            
            # Load tokenizer
            print(f"      📚 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                config.name,
                cache_dir=llm_cache,
                **token_kwargs
            )
            
            # Load model
            print(f"      🔧 Loading model (this may take a few minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                config.name,
                cache_dir=llm_cache,
                torch_dtype=getattr(torch, config.dtype),
                low_cpu_mem_usage=True,
                device_map=config.device,
                **token_kwargs
            )
            model.eval()
            
            print(f"      ✅ LLM loaded! (Device: {config.device})")
            return tokenizer, model
            
        except Exception as e:
            print(f"      ❌ Failed to load LLM: {str(e)}")
            if 'gated' in str(e).lower() or '401' in str(e):
                print(f"      🔐 Access denied! Model requires authentication.")
                print(f"      💡 Steps:")
                print(f"         1. Visit: https://huggingface.co/{config.name}")
                print(f"         2. Request access")
                print(f"         3. Create token: https://huggingface.co/settings/tokens")
                print(f"         4. Set HF_TOKEN in .env file")
            raise


class RerankerModelFactory(ModelFactory):
    """Factory for creating reranker models"""
    
    def create_model(self, config: ModelConfig, cache_dir: str = 'cache', **kwargs) -> Any:
        """
        Create reranker (cross-encoder) model
        
        Args:
            config: Model configuration
            cache_dir: Cache directory
            
        Returns:
            Loaded cross-encoder model
        """
        print(f"      📥 Loading reranker: {config.display_name}")
        
        try:
            model = CrossEncoder(config.name)
            print(f"      ✅ Reranker loaded!")
            return model
        except Exception as e:
            print(f"      ⚠️  Failed to load reranker: {str(e)}")
            print(f"      💡 Reranking will be disabled")
            return None


class ModelManager:
    """
    Manages all models using factory pattern
    Singleton pattern for resource efficiency
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.embedding_factory = EmbeddingModelFactory()
        self.llm_factory = LLMModelFactory()
        self.reranker_factory = RerankerModelFactory()
        
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.reranker_model = None
        
        self._initialized = True
    
    def load_embedding_model(self, config: EmbeddingModelConfig, cache_dir: str = 'cache'):
        """Load embedding model"""
        if self.embedding_model is None:
            self.embedding_model = self.embedding_factory.create_model(config, cache_dir=cache_dir)
        return self.embedding_model
    
    def load_llm(self, config: LLMModelConfig, cache_dir: str = 'cache', hf_token: Optional[str] = None):
        """Load LLM model and tokenizer"""
        if self.llm_model is None:
            self.llm_tokenizer, self.llm_model = self.llm_factory.create_model(
                config, cache_dir=cache_dir, hf_token=hf_token
            )
        return self.llm_tokenizer, self.llm_model
    
    def load_reranker(self, config: ModelConfig, cache_dir: str = 'cache'):
        """Load reranker model"""
        if self.reranker_model is None:
            self.reranker_model = self.reranker_factory.create_model(config, cache_dir=cache_dir)
        return self.reranker_model
    
    def clear(self):
        """Clear all loaded models (free memory)"""
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.reranker_model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ All models cleared from memory")

