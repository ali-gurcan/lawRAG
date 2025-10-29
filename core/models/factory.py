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

from core.config import ModelConfig, EmbeddingModelConfig, LLMModelConfig


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
            # Optional alternate backend: Ollama HTTP API
            backend = os.getenv('LLM_BACKEND', 'transformers').lower().strip()
            if backend in ('ollama',):
                print("      🦙 Backend: Ollama (HTTP API)")
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
                model_name = os.getenv('OLLAMA_MODEL') or 'llama3.1:8b-instruct'
                try:
                    import requests  # lightweight HTTP client
                except Exception as ie:
                    raise ImportError("requests paketi gerekli. Kur: pip install requests") from ie

                class OllamaClient:
                    def __init__(self, base_url: str, model: str):
                        self.base_url = base_url.rstrip('/')
                        self.model = model
                        self.is_ollama = True
                    def chat(self, messages, stream=False, options=None):
                        payload = {
                            'model': self.model,
                            'messages': messages,
                            'stream': bool(stream),
                        }
                        if options:
                            payload['options'] = options
                        url = f"{self.base_url}/api/chat"
                        resp = requests.post(url, json=payload, timeout=120, stream=stream)
                        resp.raise_for_status()
                        return resp

                client = OllamaClient(base_url, model_name)
                print(f"      🔗 Ollama URL: {base_url}")
                print(f"      📦 Model: {model_name}")
                return None, client

            # Optional alternate backend: llama.cpp (GGUF, true Q4_K_M)
            if backend in ('llama_cpp', 'llamacpp', 'llama-cpp'):
                print("      🦙 Backend: llama.cpp (GGUF)")
                gguf_path = os.getenv('LLM_GGUF_PATH')
                if not gguf_path or not os.path.exists(gguf_path):
                    raise ValueError("LLM_BACKEND=llama_cpp seçildi ancak LLM_GGUF_PATH bulunamadı veya dosya yok.")
                try:
                    from llama_cpp import Llama
                except Exception as ie:
                    raise ImportError("llama-cpp-python yüklü değil. Kur: pip install llama-cpp-python==0.2.*") from ie

                # Build llama.cpp model (CPU by default). Q4_K_M dosyasını kullanın.
                n_threads = max(1, os.cpu_count() or 1)
                ctx = int(os.getenv('LLM_CTX', '4096'))
                print(f"      📦 GGUF: {gguf_path}")
                print(f"      ⚙️  ctx={ctx}, threads={n_threads}")
                llm = Llama(
                    model_path=gguf_path,
                    n_ctx=ctx,
                    n_threads=n_threads,
                    logits_all=False,
                    verbose=False,
                )
                # Mark for engine logic
                setattr(llm, 'is_llama_cpp', True)
                print("      ✅ Llama.cpp GGUF model yüklendi")
                # Tokenizer yok; üst katman özel yol kullanacak
                return None, llm

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
            
            # Check if this is a Llama model (requires special handling)
            is_llama = 'llama' in config.name.lower()

            # Quantization preferences (transformers/bitsandbytes)
            # Note: Q4_K_M is a llama.cpp (GGUF) scheme; in transformers we approximate with 4-bit NF4
            quant_env = os.getenv('LLM_QUANT', '').lower().strip()
            use_4bit = quant_env in ('q4_k_m', '4bit', 'nf4')
            if use_4bit:
                print("      🧮 Quantization: 4-bit (NF4, transformers path)")
            
            if config.device == 'cpu':
                print(f"      💻 CPU mode - using ultra-safe loading...")
                
                if is_llama:
                    print(f"      🦙 Llama - using EXACT old working code...")
                    # TAM ESKİ ÇALIŞAN KOD (model_factory.py'den):
                    model_kwargs = dict(
                        cache_dir=llm_cache,
                        low_cpu_mem_usage=True,
                        **token_kwargs
                    )
                    if use_4bit:
                        # bitsandbytes 4-bit NF4
                        model_kwargs.update({
                            'load_in_4bit': True,
                            'bnb_4bit_quant_type': 'nf4',
                            'bnb_4bit_compute_dtype': torch.bfloat16,
                            'bnb_4bit_use_double_quant': True,
                            'device_map': 'auto',
                        })
                    else:
                        model_kwargs.update({
                            'torch_dtype': getattr(torch, config.dtype),
                            'device_map': config.device,
                        })
                    model = AutoModelForCausalLM.from_pretrained(
                        config.name,
                        **model_kwargs
                    )
                    print(f"      🎯 Using EXACT old working parameters")
                else:
                    # Standard loading for other models
                    model_kwargs = dict(
                        cache_dir=llm_cache,
                        low_cpu_mem_usage=True,
                        **token_kwargs
                    )
                    if use_4bit:
                        model_kwargs.update({
                            'load_in_4bit': True,
                            'bnb_4bit_quant_type': 'nf4',
                            'bnb_4bit_compute_dtype': torch.bfloat16,
                            'bnb_4bit_use_double_quant': True,
                            'device_map': 'auto',
                        })
                    else:
                        model_kwargs.update({
                            'torch_dtype': torch.float32,
                        })
                    model = AutoModelForCausalLM.from_pretrained(
                        config.name,
                        **model_kwargs
                    )
                    # Move to CPU explicitly
                    if not use_4bit:
                        model = model.to('cpu')
            else:
                # GPU mode
                dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32
                model_kwargs = dict(
                    cache_dir=llm_cache,
                    **token_kwargs
                )
                if use_4bit:
                    model_kwargs.update({
                        'load_in_4bit': True,
                        'bnb_4bit_quant_type': 'nf4',
                        'bnb_4bit_compute_dtype': torch.bfloat16,
                        'bnb_4bit_use_double_quant': True,
                        'device_map': 'auto',
                    })
                else:
                    model_kwargs.update({
                        'torch_dtype': dtype,
                        'device_map': config.device,
                    })
                model = AutoModelForCausalLM.from_pretrained(
                    config.name,
                    **model_kwargs
                )
            
            # ESKİ KODDA .eval() HER ZAMAN VARDI!
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

