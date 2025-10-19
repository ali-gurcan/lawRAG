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

    def load_cache(self) -> bool:
        """
        Load cached data (chunks, embeddings, index, BM25)
        Returns True if cache loaded successfully, False otherwise
        """
        try:
            # Check if all cache files exist
            cache_files = [self.chunks_cache, self.embeddings_cache, self.index_cache]
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
            
            # Load embeddings and index
            with open(self.embeddings_cache, 'rb') as f:
                embeddings = pickle.load(f)
            
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
        if not force_refresh and self.load_cache():
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
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Increase batch size for faster encoding
        import time
        start_time = time.time()
        
        # Use different encoding based on model type
        if hasattr(self, 'is_bge_m3') and self.is_bge_m3:
            # BGE-M3 encoding (returns dense embeddings by default)
            print(f"      💫 BGE-M3 dense encoding (batch işleniyor)...")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 16  # Smaller batch for memory efficiency
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                if (i // batch_size) % 5 == 0:  # Progress every 5 batches
                    print(f"      🔄 Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
                
                outputs = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    max_length=4096  # Reduced from 8192 for memory
                )
                all_embeddings.append(outputs['dense_vecs'])
            
            embeddings = np.vstack(all_embeddings)  # Combine all batches
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
        
        print(f"      ✅ {self.index.ntotal} vektör eklendi")
        
        # Build BM25 index for hybrid search
        if self.use_hybrid_search:
            print(f"\n      📊 BM25 index oluşturuluyor (hybrid search için)...")
            tokenized_chunks = [self._tokenize(chunk['text']) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)
            print(f"      ✅ BM25 index hazır!")
        
        # Save to cache
        print(f"\n      💾 Cache kaydediliyor (sonraki başlatmalar hızlı olacak)...")
        with open(self.chunks_cache, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        faiss.write_index(self.index, self.index_cache)
        
        if self.use_hybrid_search and self.bm25:
            with open(self.bm25_cache, 'wb') as f:
                pickle.dump(self.bm25, f)
        
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
        """Calculate overall confidence from retrieval scores"""
        if not scores:
            return 0.0
        
        # Average of top scores
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-100%
        confidence = min(100, int(avg_score * 100))
        return confidence
    
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
        
        # Extract context and sources
        contexts = []
        sources = []
        total_confidence = 0
        
        for chunk, score in results:
            contexts.append(chunk['text'])
            
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
            total_confidence += score
        
        avg_confidence = total_confidence / len(results)
        
        # Use only top 2 chunks for more focused, higher quality answers
        combined_context = "\n\n".join(contexts[:2])  # Top 2 chunks for better quality
        
        # Determine if we should use LLM
        if use_llm is None:
            use_llm = self.use_llm_generation
        
        # Generate answer with LLM if available
        if use_llm and self.llm_model and self.llm_tokenizer:
            try:
                generated_answer = self._generate_with_llm(query, combined_context)
                
                # Format answer with source citation
                source_info = f"\n\n📚 Kaynak: {sources[0]['source']}"
                if sources[0].get('article'):
                    source_info += f"\n📜 {sources[0]['article']}"
                
                answer_with_source = f"{generated_answer}{source_info}"
                
                # Check confidence and add warning if low
                confidence_pct = float(avg_confidence * 100)
                low_confidence = confidence_pct < 50
                
                result = {
                    'answer': answer_with_source,
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
        source_info = f"📚 Kaynak: {sources[0]['source']}"
        if sources[0].get('article'):
            source_info += f"\n📜 {sources[0]['article']}"
        
        answer_with_source = f"{combined_context}\n\n{source_info}"
        
        # Check confidence
        confidence_pct = float(avg_confidence * 100)
        low_confidence = confidence_pct < 50
        
        result = {
            'answer': answer_with_source,
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
        
        # Extract context and calculate confidence
        contexts = []
        sources = []
        total_confidence = 0
        
        for chunk, score in results:
            contexts.append(chunk['text'])
            sources.append({
                'source': chunk['metadata'].get('source', 'Bilinmeyen Kaynak'),
                'score': float(score),
                'article': chunk['metadata'].get('article', None)
            })
            total_confidence += score
        
        avg_confidence = total_confidence / len(results)
        confidence_pct = float(avg_confidence * 100)
        
        # Send metadata first
        yield "data: " + json.dumps({
            'type': 'metadata',
            'confidence': confidence_pct,
            'sources': sources[:3],
            'low_confidence': confidence_pct < 50
        }) + "\n\n"
        
        # Generate answer with LLM
        combined_context = "\n\n".join(contexts[:2])
        
        if self.llm_model and self.llm_tokenizer:
            # Stream LLM response
            for token in self._generate_with_llm_stream(query, combined_context):
                yield "data: " + json.dumps({'type': 'token', 'content': token}) + "\n\n"
        else:
            # Fallback: stream retrieved context
            for word in combined_context.split():
                yield "data: " + json.dumps({'type': 'token', 'content': word + ' '}) + "\n\n"
        
        # Send completion
        yield "data: " + json.dumps({'type': 'done'}) + "\n\n"
    
    def _generate_with_llm_stream(self, query: str, context: str):
        """Stream tokens from LLM using TextIteratorStreamer for non-blocking SSE."""
        from transformers import TextIteratorStreamer
        import threading
        
        # Build ultra-fast prompt for streaming
        prompt = f"S: {query}\nB: {context}\nC:"
        system_msg = "Kısa cevap ver."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize and move to correct device
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Launch generation in background thread
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=80,   # Çok kısa, ultra hızlı cevaplar
            do_sample=True,      # Hızlı sampling
            temperature=0.1,     # Çok düşük, hızlı
            top_p=0.95,          # Hızlı seçim
            top_k=20,            # Daha az seçenek (hızlı)
            repetition_penalty=1.05,  # Minimal tekrar kontrolü
            no_repeat_ngram_size=2,   # Minimal n-gram kontrolü
            pad_token_id=self.llm_tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield from streamer as tokens arrive
        for new_text in streamer:
            if new_text:
                yield new_text
        
        # Ensure thread finished
        thread.join()
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using LLM (non-streaming)"""
        # Create prompt for Turkish legal Q&A
        prompt = f"""S: {query}
B: {context}
C:"""

        system_msg = "Kısa cevap ver."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize and generate
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        
        # Generate with speed-optimized parameters
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=80,   # Çok kısa, ultra hızlı cevaplar
                do_sample=True,      # Hızlı sampling
                temperature=0.1,     # Çok düşük, hızlı
                top_p=0.95,          # Hızlı seçim
                top_k=20,            # Daha az seçenek (hızlı)
                repetition_penalty=1.05,  # Minimal tekrar kontrolü
                no_repeat_ngram_size=2,   # Minimal n-gram kontrolü
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode only the generated part
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

