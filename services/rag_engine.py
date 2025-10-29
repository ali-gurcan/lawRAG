"""
RAG (Retrieval Augmented Generation) Engine - OOP Refactored
Uses PDF documents + E5-Large + Llama 3.2 for Q&A
Enhanced with: Query Expansion, Hybrid Search (BM25+Dense), Reranking
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from services.pdf_processor import PDFProcessor
from tqdm import tqdm
import torch
import json
from rank_bm25 import BM25Okapi
import re
import time

# Import OOP components
from core.config import RAGConfig
from core.models import ModelManager
from core.strategies import RetrievalStrategyFactory

class RAGEngine:
    def __init__(self, config: RAGConfig):
        """
        Initialize Enhanced RAG Engine with configuration
        
        Args:
            config: RAGConfig instance with all settings
        """
        self.config = config
        
        # Get model configurations
        self.emb_config = config.get_embedding_model()
        self.llm_config = config.get_llm_model()
        self.reranker_config = config.get_reranker_model()
        
        # Settings from config
        self.cache_dir = config.cache.cache_dir
        self.top_k = config.retrieval.top_k
        self.use_llm_generation = True  # Always use LLM
        self.hf_token = config.hf_token
        self.use_hybrid_search = config.retrieval.use_hybrid_search
        self.use_reranking = config.retrieval.use_reranking
        self.use_query_expansion = config.retrieval.use_query_expansion
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Paths
        self.chunks_cache = os.path.join(self.cache_dir, config.cache.chunks_file)
        self.index_cache = os.path.join(self.cache_dir, config.cache.index_file)
        self.embeddings_cache = os.path.join(self.cache_dir, config.cache.embeddings_file)
        self.bm25_cache = os.path.join(self.cache_dir, config.cache.bm25_file)
        self.metadata_cache = os.path.join(self.cache_dir, 'cache_metadata.pkl')
        
        # Initialize
        self.model = None
        self.is_bge_m3 = False
        self.llm_tokenizer = None
        self.llm_model = None
        self.chunks = []
        self.index = None
        self.dimension = None
        self.bm25 = None
        self.reranker = None
        self.retrieval_strategy = None
        
        # Model manager (singleton)
        self.model_manager = ModelManager()
        
        print("🚀 Initializing Enhanced RAG Engine...")
        print(f"   ✅ Hybrid Search: {'ON' if self.use_hybrid_search else 'OFF'}")
        print(f"   ✅ Reranking: {'ON' if self.use_reranking else 'OFF'}")
        print(f"   ✅ Query Expansion: {'ON' if self.use_query_expansion else 'OFF'}")
        
        # Load models using factory pattern
        self._load_embedding_model()
        
        if self.use_reranking:
            self._load_reranker()
        
        if self.use_llm_generation:
            self._load_llm_model()
    
    def _load_embedding_model(self):
        """Load embedding model using factory pattern"""
        try:
            # Use ModelManager to load embedding model
            self.model = self.model_manager.load_embedding_model(
                self.emb_config,
                cache_dir=self.cache_dir
            )
            
            # Set dimension and check if BGE-M3
            self.is_bge_m3 = hasattr(self.model, 'is_bge_m3') and self.model.is_bge_m3
            self.dimension = self.emb_config.dimension
            
        except Exception as e:
            print(f"      ❌ Embedding model yüklenemedi: {str(e)}")
            raise
    
    def _load_reranker(self):
        """Load reranker model using factory pattern"""
        try:
            self.reranker = self.model_manager.load_reranker(
                self.reranker_config,
                cache_dir=self.cache_dir
            )
            if self.reranker is None:
                self.use_reranking = False
        except Exception as e:
            print(f"      ⚠️  Reranker yüklenemedi: {str(e)}")
            self.use_reranking = False
    
    def _load_llm_model(self):
        """Load LLM using factory pattern"""
        try:
            self.llm_tokenizer, self.llm_model = self.model_manager.load_llm(
                self.llm_config,
                cache_dir=self.cache_dir,
                hf_token=self.hf_token
            )
        except Exception as e:
            print(f"      ❌ LLM yüklenemedi: {str(e)}")
            raise

    def _get_pdf_signature(self, pdf_paths: List[str]) -> Dict:
        """
        Generate signature for PDF files (hash + mtime) for cache invalidation
        """
        signature = {}
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                # Get modification time
                mtime = os.path.getmtime(pdf_path)
                # Get file size as hash component
                size = os.path.getsize(pdf_path)
                # Simple signature: filename + mtime + size
                sig_key = os.path.basename(pdf_path)
                signature[sig_key] = {
                    'mtime': mtime,
                    'size': size,
                    'path': pdf_path
                }
        return signature
    
    def _is_cache_valid(self, pdf_paths: List[str]) -> bool:
        """
        Check if cache is valid by comparing PDF signatures
        """
        if not os.path.exists(self.metadata_cache):
            return False
        
        try:
            with open(self.metadata_cache, 'rb') as f:
                cached_metadata = pickle.load(f)
            
            current_signature = self._get_pdf_signature(pdf_paths)
            
            # Check if same files with same signatures
            cached_files = set(cached_metadata.get('pdf_signature', {}).keys())
            current_files = set(current_signature.keys())
            
            if cached_files != current_files:
                print("   ⚠️  PDF list changed, cache invalidated")
                return False
            
            # Check if any file was modified
            for filename, sig in current_signature.items():
                cached_sig = cached_metadata.get('pdf_signature', {}).get(filename)
                if not cached_sig:
                    return False
                if sig['mtime'] != cached_sig.get('mtime') or sig['size'] != cached_sig.get('size'):
                    print(f"   ⚠️  {filename} was modified, cache invalidated")
                    return False
            
            return True
        except Exception as e:
            print(f"   ⚠️  Cache metadata check failed: {str(e)}")
            return False
    
    def load_cache(self, pdf_paths: Optional[List[str]] = None) -> bool:
        """
        Load cached data (chunks, embeddings, index, BM25)
        Returns True if cache loaded successfully, False otherwise
        """
        try:
            # Validate cache against PDF files if provided
            if pdf_paths and not self._is_cache_valid(pdf_paths):
                return False
            
            # Check if all required cache files exist
            cache_files = [self.chunks_cache, self.index_cache]
            if self.use_hybrid_search:
                cache_files.append(self.bm25_cache)
            
            if not all(os.path.exists(f) for f in cache_files):
                print("   💡 Cache files not found, will create new ones")
                return False
            
            print("   📂 Loading cached data...")
            
            # Load chunks
            with open(self.chunks_cache, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"      ✅ Chunks loaded: {len(self.chunks)} chunks")
            
            # Load index (embeddings are stored in FAISS index)
            self.index = faiss.read_index(self.index_cache)
            self.dimension = self.index.d
            print(f"      ✅ FAISS index loaded: {self.index.ntotal} vectors")
            
            # Load BM25 if hybrid search is enabled
            if self.use_hybrid_search and os.path.exists(self.bm25_cache):
                with open(self.bm25_cache, 'rb') as f:
                    self.bm25 = pickle.load(f)
                print(f"      ✅ BM25 index loaded")
            
            print("   🚀 Cache loaded successfully!")
            return True
            
        except Exception as e:
            print(f"   ⚠️  Cache loading failed: {str(e)}")
            return False
    
    def process_pdfs(self, pdf_paths: List[str], force_refresh=False):
        """
        Process PDFs and create vector index
        
        Args:
            pdf_paths: List of PDF file paths or directories
            force_refresh: Force reprocessing even if cache exists
        """
        # Try to load from cache
        if not force_refresh and self.load_cache(pdf_paths):
            print("      ✅ Cache kullanılıyor (hızlı başlatma)")
            return
        
        print()
        print("      " + "─"*60)
        print("      🔄 PDF İşleme Başlıyor (İlk seferlik işlem)")
        print("      " + "─"*60)
        
        # Initialize PDF processor (larger chunks = faster processing)
        processor = PDFProcessor(chunk_size=1000, chunk_overlap=100)
        
        # Process all PDFs
        all_chunks = []
        total_pdfs = len(pdf_paths)
        
        for idx, path in enumerate(pdf_paths, 1):
            print(f"\n      📄 [{idx}/{total_pdfs}] İşleniyor: {os.path.basename(path)}")
            print(f"      " + "─"*60)
            
            import time
            pdf_start_time = time.time()
            
            if os.path.isdir(path):
                chunks = processor.process_directory(path)
            elif os.path.isfile(path):
                chunks = processor.process_pdf(path)
            else:
                print(f"      ⚠️  Dosya bulunamadı: {path}")
                continue
            
            pdf_elapsed = time.time() - pdf_start_time
            
            all_chunks.extend(chunks)
            print(f"      ✅ {len(chunks)} chunk oluşturuldu ({pdf_elapsed:.1f} saniye)")
            print(f"      📊 Toplam: {len(all_chunks)} chunk")
        
        self.chunks = all_chunks
        
        if not self.chunks:
            print("      ❌ Hiç chunk oluşturulamadı!")
            return
        
        # Create embeddings
        print(f"\n      🔮 Embedding oluşturuluyor ({len(self.chunks)} chunk)...")
        total_chars = sum(len(chunk['text']) for chunk in self.chunks)
        avg_chunk_size = total_chars // len(self.chunks)
        print(f"      📊 Ortalama chunk boyutu: {avg_chunk_size:,} karakter")
        print(f"      ⏳ Tahmini süre: {len(self.chunks) // 50}-{len(self.chunks) // 30} dakika...")
        
        # Filter out empty/whitespace-only texts to avoid tokenizer empty batch errors
        texts = [chunk['text'] for chunk in self.chunks if chunk.get('text') and chunk['text'].strip()]
        if not texts:
            raise ValueError("No valid chunk texts to encode for embeddings.")
        
        # Increase batch size for faster encoding
        import time
        start_time = time.time()
        
        # Use different encoding based on model type
        if hasattr(self, 'is_bge_m3') and self.is_bge_m3:
            # BGE-M3 encoding (returns dense embeddings by default)
            print(f"      💫 BGE-M3 dense encoding (batch işleniyor)...")
            
            # Ultra-small batches to prevent 32GB RAM spikes
            batch_size = 4  # Very small batch for extreme memory efficiency
            import gc
            
            # Incremental stacking to avoid accumulating large lists
            embeddings_list = []
            total_processed = 0
            
            num_batches = (len(texts) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Ultra-strict filtering - no empty strings, no None, must be valid text
                batch_texts = [
                    str(t).strip() 
                    for t in batch_texts 
                    if t is not None 
                    and isinstance(t, (str, bytes))
                    and str(t).strip()
                    and len(str(t).strip()) > 10
                    and len(str(t).strip()) < 50000  # Max length protection
                ]
                
                # Final validation - must have at least 1 valid text
                if not batch_texts or len(batch_texts) == 0:
                    print(f"         ⚠️  Batch {batch_idx + 1} atlandı: boş batch")
                    continue
                
                # Ensure all texts are truly strings
                batch_texts = [str(t) for t in batch_texts if t]
                if not batch_texts:
                    continue
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    print(f"      🔄 Batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} texts)")
                
                try:
                    # Validate before encoding
                    if not isinstance(batch_texts, list) or len(batch_texts) == 0:
                        raise ValueError(f"Invalid batch_texts: {type(batch_texts)}, length: {len(batch_texts) if hasattr(batch_texts, '__len__') else 'N/A'}")
                    
                    # Encode with minimum memory footprint
                    # Use smaller internal batch_size to avoid tokenizer issues
                    outputs = self.model.encode(
                        batch_texts,
                        batch_size=min(len(batch_texts), 2),  # Max 2 at a time internally
                        max_length=4096
                        # Note: normalize_embeddings not supported by BGE-M3 tokenizer
                    )
                    
                    # Validate output structure
                    if not outputs or 'dense_vecs' not in outputs:
                        raise ValueError(f"Invalid output structure: {type(outputs)}, keys: {outputs.keys() if isinstance(outputs, dict) else 'N/A'}")
                    dense_vecs = outputs['dense_vecs']
                    batch_count = len(batch_texts)  # Save before deletion
                    if isinstance(dense_vecs, np.ndarray):
                        embeddings_list.append(dense_vecs)
                    else:
                        embeddings_list.append(np.array(dense_vecs))
                    
                    # Aggressive cleanup after each batch
                    del outputs
                    del dense_vecs
                    del batch_texts
                    gc.collect()
                    
                    # Clear PyTorch cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    
                    total_processed += batch_count
                except Exception as e:
                    print(f"         ⚠️  Batch {batch_idx + 1} atlandı: {e}")
                    continue
            
            # Stack incrementally in chunks to avoid 32GB spike
            if not embeddings_list:
                raise ValueError("No embeddings generated!")
            
            print(f"      📊 {total_processed} text encoded, stacking embeddings...")
            # Stack in smaller chunks
            chunk_size = 20  # Stack 20 batches at a time
            stacked_chunks = []
            for i in range(0, len(embeddings_list), chunk_size):
                chunk = embeddings_list[i:i + chunk_size]
                stacked = np.vstack(chunk)
                stacked_chunks.append(stacked)
                del chunk
                gc.collect()
            
            # Final stack
            embeddings = np.vstack(stacked_chunks) if stacked_chunks else embeddings_list[0]
            del embeddings_list
            del stacked_chunks
            gc.collect()
        else:
            # Standard SentenceTransformer encoding
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=64,
                convert_to_numpy=True
            )
        
        elapsed_time = time.time() - start_time
        print(f"      ✅ Embedding tamamlandı ({elapsed_time:.1f} saniye)")
        
        # Create FAISS index
        print(f"\n      📚 FAISS vektör veritabanı oluşturuluyor...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        # Free large temporary arrays ASAP
        try:
            del embeddings
            del texts
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        
        print(f"      ✅ {self.index.ntotal} vektör eklendi")
        
        # Build BM25 index for hybrid search
        if self.use_hybrid_search:
            print(f"\n      📊 BM25 index oluşturuluyor (hybrid search için)...")
            tokenized_chunks = [self._tokenize(chunk['text']) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            # Free tokenized view
            try:
                del tokenized_chunks
            except Exception:
                pass
            _gc.collect()
            print(f"      ✅ BM25 index hazır!")
        
        # Save to cache
        print(f"\n      💾 Cache kaydediliyor (sonraki başlatmalar hızlı olacak)...")
        with open(self.chunks_cache, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        faiss.write_index(self.index, self.index_cache)
        
        if self.use_hybrid_search and self.bm25:
            with open(self.bm25_cache, 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # Save cache metadata (PDF signatures for invalidation)
        pdf_signature = self._get_pdf_signature(pdf_paths)
        cache_metadata = {
            'pdf_signature': pdf_signature,
            'created_at': time.time(),
            'num_chunks': len(self.chunks),
            'index_size': self.index.ntotal
        }
        with open(self.metadata_cache, 'wb') as f:
            pickle.dump(cache_metadata, f)
        
        cache_size = (os.path.getsize(self.chunks_cache) + os.path.getsize(self.index_cache)) / (1024*1024)
        print(f"      ✅ Cache kaydedildi (~{cache_size:.1f} MB)")
        print(f"      💡 Sonraki başlatmalarda bu işlem atlanacak!")
        print()
    
    def _tokenize(self, text):
        """Simple tokenization for BM25"""
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms for better retrieval"""
        if not self.use_query_expansion:
            return query
        
        # Extract key terms
        expanded = query
        
        # Legal terms expansion for Turkish
        legal_expansions = {
            r'\begemenlik\b': 'egemenlik millet yetki',
            r'\byasama\b': 'yasama meclis kanun',
            r'\byürütme\b': 'yürütme cumhurbaşkanı icra',
            r'\byargı\b': 'yargı mahkeme adalet',
            r'\bmadde\s+(\d+)': r'madde \1',
            r'\banayasa\b': 'anayasa temel kanun',
        }
        
        for pattern, expansion in legal_expansions.items():
            if re.search(pattern, query.lower()):
                expanded += ' ' + expansion
        
        return expanded.strip()
    
    def retrieve(self, query: str, k: int = None) -> List[Tuple[Dict, float]]:
        """
        Enhanced retrieval with query expansion, hybrid search, and reranking
        
        Args:
            query: User query
            k: Number of chunks to retrieve (default: self.top_k)
        
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None or not self.chunks:
            print("❌ No index available. Please process PDFs first.")
            return []
        
        k = k or self.top_k
        
        # 1. Query Expansion
        expanded_query = self._expand_query(query)
        
        # 2. Hybrid Search (BM25 + Dense)
        if self.use_hybrid_search and self.bm25:
            # Get more candidates for reranking
            retrieval_k = min(k * 3, 10)  # Get 3x more for reranking
            
            # Dense retrieval (E5-Large)
            if hasattr(self, 'is_bge_m3') and self.is_bge_m3:
                output = self.model.encode([expanded_query], max_length=8192)
                query_embedding = output['dense_vecs']
            else:
                query_embedding = self.model.encode([expanded_query], convert_to_numpy=True)
            
            distances, indices = self.index.search(query_embedding.astype('float32'), retrieval_k)
            
            # BM25 retrieval
            query_tokens = self._tokenize(expanded_query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Combine scores (weighted average)
            combined_results = {}
            alpha = 0.7  # Weight for dense retrieval (0.7 = 70% dense, 30% BM25)
            
            # Add dense results
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    dense_score = 1 / (1 + distance)
                    combined_results[idx] = alpha * dense_score
            
            # Add BM25 results
            bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            for idx, score in enumerate(bm25_scores):
                normalized_score = score / bm25_max
                if idx in combined_results:
                    combined_results[idx] += (1 - alpha) * normalized_score
                else:
                    combined_results[idx] = (1 - alpha) * normalized_score
            
            # Sort by combined score
            sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:retrieval_k]
            results = [(self.chunks[idx], score) for idx, score in sorted_results if idx < len(self.chunks)]
        else:
            # Standard dense retrieval only
            if hasattr(self, 'is_bge_m3') and self.is_bge_m3:
                output = self.model.encode([expanded_query], max_length=8192)
                query_embedding = output['dense_vecs']
            else:
                query_embedding = self.model.encode([expanded_query], convert_to_numpy=True)
            
            distances, indices = self.index.search(query_embedding.astype('float32'), k * 2)
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    similarity = 1 / (1 + distance)
                    results.append((chunk, similarity))
        
        # 3. Reranking with Cross-Encoder
        if self.use_reranking and self.reranker and len(results) > k:
            # Prepare pairs for reranking
            pairs = [[query, chunk['text']] for chunk, _ in results]
            
            # Get reranker scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine with retrieval scores (weighted)
            beta = 0.6  # Weight for reranker (60% reranker, 40% retrieval)
            final_scores = []
            for i, (chunk, retrieval_score) in enumerate(results):
                # Normalize reranker score to 0-1
                rerank_score = (rerank_scores[i] + 10) / 20  # MS-MARCO scores range ~[-10, 10]
                rerank_score = max(0, min(1, rerank_score))
                
                combined_score = beta * rerank_score + (1 - beta) * retrieval_score
                final_scores.append((chunk, combined_score))
            
            # Sort by combined score and take top-k
            final_scores.sort(key=lambda x: x[1], reverse=True)
            results = final_scores[:k]
        else:
            # Just take top-k
            results = results[:k]
        
        return results
    
    def calculate_confidence(self, scores):
        """
        Calculate overall confidence from retrieval scores
        Uses configurable method from confidence config
        """
        if not scores:
            return 0.0
        
        conf_config = self.config.confidence
        method = getattr(conf_config, 'calculation_method', 'weighted_top')
        
        if method == 'top_score_only':
            # Use only the top score
            confidence = max(scores) if scores else 0.0
        elif method == 'weighted_top':
            # Weighted combination: top score gets more weight
            sorted_scores = sorted(scores, reverse=True)
            top_weight = getattr(conf_config, 'top_score_weight', 0.6)
            if len(sorted_scores) >= 2:
                # Weighted average: top score * weight + others * (1-weight) / (n-1)
                remaining_weight = (1.0 - top_weight) / (len(sorted_scores) - 1)
                confidence = (sorted_scores[0] * top_weight + 
                            sum(sorted_scores[1:]) * remaining_weight)
            else:
                confidence = sorted_scores[0] if sorted_scores else 0.0
        else:  # 'average' (default fallback)
            # Simple average
            confidence = sum(scores) / len(scores)
        
        # Boost if top score is very high and boost is enabled
        if getattr(conf_config, 'boost_high_scores', True) and scores:
            top_score = max(scores)
            if top_score > 0.85:
                # Boost by up to 10% for very high scores
                boost = (top_score - 0.85) * 0.4  # Scale from 0-0.15 to 0-0.06
                confidence = min(1.0, confidence + boost)
        
        # Normalize to 0-100%
        confidence_pct = min(100, int(confidence * 100))
        return confidence_pct / 100.0  # Return as 0-1 for consistency
    
    def _format_answer(self, answer_text: str, sources: List[Dict], confidence: float, is_context_only: bool = False) -> str:
        """
        Format answer with beautiful template for Turkish legal documents
        
        Args:
            answer_text: Generated or retrieved answer text
            sources: List of source metadata
            confidence: Average confidence score (0-1)
            is_context_only: Whether this is direct context (not LLM generated)
        
        Returns:
            Formatted answer string
        """
        confidence_pct = int(confidence * 100)

        def _short_quote(text: str, max_len: int = 140) -> str:
            if not text:
                return ""
            t = text.strip().replace("\n", " ")
            # Prefer the first sentence
            end_idx = -1
            for sep in [".", "!", "?"]:
                idx = t.find(sep)
                if idx != -1 and (end_idx == -1 or idx < end_idx):
                    end_idx = idx
            if 0 < end_idx <= max_len:
                t = t[: end_idx + 1]
            if len(t) > max_len:
                t = t[: max_len - 1].rstrip() + "…"
            return f"“{t}”"

        def _loc_label(src: Dict) -> str:
            article = src.get('article')
            if article:
                # Normalize formats like "MADDE 3" -> "Anayasa m.3" when possible
                m = re.search(r"MADDE\s*(\d+)", str(article))
                if m:
                    return f"m.{m.group(1)}"
                return str(article)
            chunk_id = src.get('chunk_id', 0)
            return f"Bölüm {chunk_id}" if chunk_id else "Konum"

        # Build formatted answer with clear markdown structure
        parts: List[str] = []
        
        # Clean answer text (remove any trailing punctuation we'll add)
        answer_clean = answer_text.strip().rstrip(".!?")
        if not answer_clean:
            answer_clean = "Cevap bulunamadı."

        # CEVAP section (bold header)
        parts.append("**CEVAP**")
        parts.append("")
        parts.append(answer_clean + ".")
        parts.append("")

        # DAYANAK section (up to top-3)
        if sources:
            parts.append("**DAYANAK**")
            parts.append("")
            for src in sources[:3]:
                label = _loc_label(src)
                quote = _short_quote(src.get('preview') or src.get('text') or '')
                if quote:
                    parts.append(f"- Anayasa {label} — {quote}")
                else:
                    parts.append(f"- Anayasa {label}")
            parts.append("")

        # GÜVEN section
        if confidence_pct >= 80:
            conf_text = "Yüksek"
            conf_emoji = "✓"
        elif confidence_pct >= 50:
            conf_text = "Orta"
            conf_emoji = "⚠"
        else:
            conf_text = "Düşük"
            conf_emoji = "✗"
        
        parts.append("**GÜVEN SEVİYESİ**")
        parts.append("")
        parts.append(f"{conf_emoji} {conf_text} ({confidence_pct}%)")

        return "\n\n".join(parts)
    
    def generate_answer(self, query: str, use_llm=None) -> Dict:
        """
        Generate answer using retrieved context
        
        Args:
            query: User query
            use_llm: Whether to use LLM for generation (not implemented yet)
        
        Returns:
            Answer dictionary with context and metadata
        """
        # Retrieve relevant chunks
        results = self.retrieve(query)
        
        if not results:
            return {
                'answer': 'Üzgünüm, bu konuda ilgili bilgi bulamadım. Lütfen sorunuzu farklı şekilde ifade edebilir misiniz?',
                'confidence': 0.0,
                'sources': [],
                'has_sources': False
            }
        
        # Extract context candidates and sources
        contexts = []
        sources = []
        scores_only = []
        for chunk, score in results:
            contexts.append(chunk['text'])
            scores_only.append(float(score))
            # Kaynak bilgisi zorunlu
            source_name = chunk['metadata'].get('source', 'Bilinmeyen Kaynak')
            chunk_id = chunk.get('id', 0)
            sources.append({
                'source': source_name,
                'score': float(score),
                'preview': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                'chunk_id': chunk_id,
                'start_char': chunk.get('start_char', 0),
                'end_char': chunk.get('end_char', 0),
                'article': chunk['metadata'].get('article', None)
            })
        
        # Calculate confidence using configurable method (not simple average)
        avg_confidence = self.calculate_confidence(scores_only)
        
        # Adaptive selection of top-n contexts (n in [1,3]) with length budget
        n_candidates = min(3, len(contexts))
        s1 = scores_only[0] if n_candidates >= 1 else 0.0
        s2 = scores_only[1] if n_candidates >= 2 else 0.0
        s3 = scores_only[2] if n_candidates >= 3 else 0.0
        selected_n = n_candidates
        if n_candidates >= 2 and (s1 - s2 >= 0.25) and (s1 >= 0.75):
            selected_n = 1
        elif n_candidates >= 3 and (s2 >= 0.55) and ((s2 - s3 >= 0.15) or (s1 - s3 >= 0.35)):
            selected_n = 2
        else:
            selected_n = n_candidates
        
        # Length budget from config (not hard-coded)
        length_budget = self.config.llm_generation.length_budget
        combined_context = ""
        for i in range(selected_n):
            piece = contexts[i]
            tentative = (combined_context + ("\n\n" if combined_context else "") + piece)
            if len(tentative) <= length_budget or i == 0:
                combined_context = tentative
            else:
                selected_n = i  # stop adding more
                break
        
        # Determine if we should use LLM
        if use_llm is None:
            use_llm = self.use_llm_generation
        
        # Generate answer with LLM if available
        if use_llm and self.llm_model:
            try:
                generated_answer = self._generate_with_llm(query, combined_context)
                
                # Format answer with beautiful template
                formatted_answer = self._format_answer(
                    generated_answer, 
                    sources, 
                    avg_confidence
                )
                
                # Check confidence and add warning if low
                confidence_pct = float(avg_confidence * 100)
                low_confidence = confidence_pct < 50
                
                result = {
                    'answer': formatted_answer,
                    'confidence': confidence_pct,
                    'sources': sources,
                    'num_sources': len(sources),
                    'has_sources': True,
                    'primary_source': sources[0]['source'],
                    'generated': True,
                    'low_confidence': low_confidence
                }
                
                # Add warning for low confidence
                if low_confidence:
                    result['warning'] = "⚠️ Bu cevaba düşük güvenle ulaştım. Daha spesifik bir soru sormayı deneyebilirsiniz."
                
                return result
            except Exception as e:
                print(f"      ⚠️  LLM generation hatası: {str(e)}")
                print(f"      💡 Retrieval-only cevap döndürülüyor...")
        
        # Fallback: Return retrieved context
        formatted_answer = self._format_answer(
            combined_context,
            sources,
            avg_confidence,
            is_context_only=True
        )
        
        # Check confidence
        confidence_pct = float(avg_confidence * 100)
        low_confidence = confidence_pct < 50
        
        result = {
            'answer': formatted_answer,
            'confidence': confidence_pct,
            'sources': sources,
            'num_sources': len(sources),
            'has_sources': True,
            'primary_source': sources[0]['source'],
            'generated': False,
            'low_confidence': low_confidence
        }
        
        # Add warning for low confidence
        if low_confidence:
            result['warning'] = "⚠️ Bu cevaba düşük güvenle ulaştım. Daha spesifik bir soru sormayı deneyebilirsiniz."
        
        return result
    
    def generate_answer_stream(self, query: str):
        """
        Generator function for streaming LLM responses
        Yields chunks of the answer as they're generated
        """
        # Retrieve relevant chunks
        results = self.retrieve(query)
        
        if not results:
            yield "data: " + json.dumps({'type': 'error', 'content': 'İlgili bilgi bulunamadı'}) + "\n\n"
            return
        
        # Extract context and calculate confidence (with adaptive selection)
        contexts_all = []
        sources_all = []
        scores_only = []
        for chunk, score in results:
            contexts_all.append(chunk['text'])
            scores_only.append(float(score))
            sources_all.append({
                'source': chunk['metadata'].get('source', 'Bilinmeyen Kaynak'),
                'score': float(score),
                'article': chunk['metadata'].get('article', None)
            })
        # Calculate confidence using configurable method
        avg_confidence = self.calculate_confidence(scores_only)
        confidence_pct = float(avg_confidence * 100)
        
        # Send metadata first
        yield "data: " + json.dumps({
            'type': 'metadata',
            'confidence': confidence_pct,
            'sources': sources_all[:3],
            'low_confidence': confidence_pct < 50
        }) + "\n\n"
        
        # Adaptive selection of context for streaming
        n_candidates = min(3, len(contexts_all))
        s1 = scores_only[0] if n_candidates >= 1 else 0.0
        s2 = scores_only[1] if n_candidates >= 2 else 0.0
        s3 = scores_only[2] if n_candidates >= 3 else 0.0
        selected_n = n_candidates
        if n_candidates >= 2 and (s1 - s2 >= 0.25) and (s1 >= 0.75):
            selected_n = 1
        elif n_candidates >= 3 and (s2 >= 0.55) and ((s2 - s3 >= 0.15) or (s1 - s3 >= 0.35)):
            selected_n = 2
        else:
            selected_n = n_candidates
        length_budget = self.config.llm_generation.length_budget
        combined_context = ""
        for i in range(selected_n):
            piece = contexts_all[i]
            tentative = (combined_context + ("\n\n" if combined_context else "") + piece)
            if len(tentative) <= length_budget or i == 0:
                combined_context = tentative
            else:
                break
        
        full_answer = ""
        if self.llm_model and self.llm_tokenizer:
            # Stream LLM response
            for token in self._generate_with_llm_stream(query, combined_context):
                full_answer += token
                yield "data: " + json.dumps({'type': 'token', 'content': token}) + "\n\n"
        else:
            # Fallback: stream retrieved context
            full_answer = combined_context
            for word in combined_context.split():
                yield "data: " + json.dumps({'type': 'token', 'content': word + ' '}) + "\n\n"
        
        # Format the complete answer with template
        formatted_answer = self._format_answer(
            full_answer if full_answer.strip() else combined_context,
            sources_all,
            avg_confidence
        )
        
        # Send formatted result
        yield "data: " + json.dumps({
            'type': 'formatted',
            'content': formatted_answer
        }) + "\n\n"
        
        # Send completion
        yield "data: " + json.dumps({'type': 'done'}) + "\n\n"
    
    def _generate_with_llm_stream(self, query: str, context: str):
        """Stream tokens from LLM (transformers or llama.cpp)."""
        # Ollama streaming path
        if hasattr(self.llm_model, 'is_ollama') and self.llm_model.is_ollama:
            context_limit = self.config.llm_generation.context_limit
            max_tokens = self.config.llm_generation.max_tokens
            max_words = self.config.llm_generation.max_answer_words
            context_snippet = context[:context_limit] if len(context) > context_limit else context
            sys = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Soru: {query}\n\nBağlam: {context_snippet}\n\nCevap (sadece sorunun cevabı, madde metnini tekrarlama):"}
            ]
            try:
                # Stream JSON lines
                resp = self.llm_model.chat(messages, stream=True, options={
                    'temperature': 0.1,
                    'top_p': 0.95,
                    'num_ctx': 4096,
                    'num_predict': max_tokens,
                })
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'message' in data and data['message'].get('content'):
                            yield data['message']['content']
                        elif 'done' in data and data['done']:
                            break
                    except Exception:
                        continue
                return
            except Exception as e:
                print(f"      ⚠️  Ollama stream hatası: {e}")
                return

        # llama.cpp streaming path
        if hasattr(self.llm_model, 'is_llama_cpp') and self.llm_model.is_llama_cpp:
            context_limit = self.config.llm_generation.context_limit
            max_tokens = self.config.llm_generation.max_tokens
            max_words = self.config.llm_generation.max_answer_words
            context_snippet = context[:context_limit] if len(context) > context_limit else context
            sys = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Soru: {query}\n\nBağlam: {context_snippet}\n\nCevap (sadece sorunun cevabı, madde metnini tekrarlama):"}
            ]
            try:
                # Use chat-completion style if available
                for chunk in self.llm_model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=0.95,
                ):
                    delta = chunk["choices"][0]["delta"].get("content", "") if "delta" in chunk["choices"][0] else chunk["choices"][0]["text"]
                    if delta:
                        yield delta
                return
            except Exception as e:
                print(f"      ⚠️  llama.cpp stream hatası: {e}")
                return
        
        # transformers streaming path (default)
        from transformers import TextIteratorStreamer
        import threading
        context_limit = self.config.llm_generation.context_limit
        max_tokens = self.config.llm_generation.max_tokens
        max_words = self.config.llm_generation.max_answer_words
        context_snippet = context[:context_limit] if len(context) > context_limit else context
        prompt = f"Soru: {query}\n\nBağlam: {context_snippet}\n\nCevap (sadece sorunun cevabı, madde metnini tekrarlama):"
        system_msg = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.05,
            no_repeat_ngram_size=2,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            if new_text:
                yield new_text
        thread.join()
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using LLM (llama.cpp or transformers)."""
        # Ollama synchronous path
        if hasattr(self.llm_model, 'is_ollama') and self.llm_model.is_ollama:
            # Dynamic limits from config
            context_limit = self.config.llm_generation.context_limit
            max_tokens = self.config.llm_generation.max_tokens
            max_words = self.config.llm_generation.max_answer_words
            context_snippet = context[:context_limit] if len(context) > context_limit else context
            
            sys = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Soru: {query}\n\nBağlam: {context_snippet}\n\nCevap (sadece sorunun cevabı, madde metnini tekrarlama):"}
            ]
            try:
                resp = self.llm_model.chat(messages, stream=False, options={
                    'temperature': 0.1,
                    'top_p': 0.95,
                    'num_ctx': 4096,
                    'num_predict': max_tokens,
                })
                data = resp.json()
                # Newer Ollama returns {'message': {'content': ...}}
                if 'message' in data and data['message'].get('content'):
                    return data['message']['content'].strip()
                # Older stream-like fallbacks
                if 'response' in data:
                    return str(data['response']).strip()
                return context[:300]
            except Exception as e:
                print(f"      ⚠️  Ollama generate hatası: {e}")
                return context[:300]

        # llama.cpp synchronous path
        if hasattr(self.llm_model, 'is_llama_cpp') and self.llm_model.is_llama_cpp:
            context_limit = self.config.llm_generation.context_limit
            max_tokens = self.config.llm_generation.max_tokens
            max_words = self.config.llm_generation.max_answer_words
            context_snippet = context[:context_limit] if len(context) > context_limit else context
            sys = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Soru: {query}\n\nBağlam: {context_snippet}\n\nCevap (sadece sorunun cevabı, madde metnini tekrarlama):"}
            ]
            try:
                out = self.llm_model.create_chat_completion(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=0.95,
                )
                return out["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"      ⚠️  llama.cpp generate hatası: {e}")
                return context[:300]

        # transformers path
        context_limit = self.config.llm_generation.context_limit
        max_tokens = self.config.llm_generation.max_tokens
        max_words = self.config.llm_generation.max_answer_words
        context_snippet = context[:context_limit] if len(context) > context_limit else context
        prompt = f"""Soru: {query}

Bağlam: {context_snippet}

Cevap (sadece sorunun cevabı, madde metnini tekrarlama):"""
        system_msg = f"Sen bir hukuki asistanısın. Soruya sadece kısa, özlü cevap ver. Madde metnini veya uzun açıklamaları tekrarlama. Sadece sorunun cevabını ver (1-2 cümle, maksimum {max_words} kelime)."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.05,
                no_repeat_ngram_size=2,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        generated_text = self.llm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return generated_text.strip()
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system"""
        return {
            'total_chunks': len(self.chunks),
            'embedding_model': self.embedding_model_name,
            'dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0,
            'cache_exists': all(os.path.exists(f) for f in [self.chunks_cache, self.index_cache])
        }
    
    def is_ready(self) -> bool:
        """Check if RAG engine is ready"""
        return self.model is not None and self.index is not None and len(self.chunks) > 0


if __name__ == "__main__":
    # Test RAG engine
    print("\n🧪 Testing RAG Engine\n")
    
    rag = RAGEngine()
    
    # Process Anayasa PDF
    pdf_path = "docs/anayasa.pdf"
    if os.path.exists(pdf_path):
        rag.process_pdfs([pdf_path])
        
        # Test query
        test_queries = [
            "Cumhuriyetin nitelikleri nelerdir?",
            "İnsan hakları nedir?",
            "Türkiye Cumhuriyeti nasıl tanımlanır?"
        ]
        
        print("\n" + "="*70)
        print("🔍 Testing Queries")
        print("="*70 + "\n")
        
        for query in test_queries:
            print(f"\n❓ Soru: {query}")
            print("-" * 70)
            
            result = rag.generate_answer(query)
            
            print(f"💡 Cevap: {result['answer'][:300]}...")
            print(f"📊 Güven: {result['confidence']:.2%}")
            print(f"📚 Kaynak sayısı: {result['num_sources']}")
            print()
    else:
        print(f"❌ PDF not found: {pdf_path}")

