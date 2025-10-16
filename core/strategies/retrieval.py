"""
Retrieval Strategies - Strategy Pattern for different retrieval methods
Implements Strategy Pattern for clean OOP design
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
from rank_bm25 import BM25Okapi
import re


class RetrievalStrategy(ABC):
    """Abstract strategy for document retrieval"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        pass


class DenseRetrievalStrategy(RetrievalStrategy):
    """Dense retrieval using embedding similarity (FAISS)"""
    
    def __init__(self, model, index, chunks, is_bge_m3=False):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.is_bge_m3 = is_bge_m3
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Dense retrieval with embeddings"""
        if self.index is None or not self.chunks:
            return []
        
        # Create query embedding
        if self.is_bge_m3:
            output = self.model.encode([query], max_length=8192)
            query_embedding = output['dense_vecs']
        else:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                results.append((chunk, similarity))
        
        return results


class SparseRetrievalStrategy(RetrievalStrategy):
    """Sparse retrieval using BM25"""
    
    def __init__(self, bm25: BM25Okapi, chunks: List[Dict]):
        self.bm25 = bm25
        self.chunks = chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """BM25 retrieval"""
        if self.bm25 is None or not self.chunks:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Normalize scores to 0-1
        max_score = max(scores) if max(scores) > 0 else 1.0
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                normalized_score = scores[idx] / max_score
                results.append((chunk, normalized_score))
        
        return results


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(
        self, 
        dense_strategy: DenseRetrievalStrategy,
        sparse_strategy: SparseRetrievalStrategy,
        alpha: float = 0.7
    ):
        """
        Args:
            dense_strategy: Dense retrieval strategy
            sparse_strategy: Sparse retrieval strategy
            alpha: Weight for dense (1-alpha for sparse), 0-1
        """
        self.dense_strategy = dense_strategy
        self.sparse_strategy = sparse_strategy
        self.alpha = alpha
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval combining dense and sparse"""
        # Get more candidates for better fusion
        retrieval_k = min(k * 3, 10)
        
        # Dense retrieval
        dense_results = self.dense_strategy.retrieve(query, retrieval_k)
        
        # Sparse retrieval
        sparse_results = self.sparse_strategy.retrieve(query, retrieval_k)
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for chunk, score in dense_results:
            chunk_id = id(chunk)  # Use object id as key
            combined_scores[chunk_id] = {
                'chunk': chunk,
                'score': self.alpha * score
            }
        
        # Add sparse scores
        for chunk, score in sparse_results:
            chunk_id = id(chunk)
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['score'] += (1 - self.alpha) * score
            else:
                combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'score': (1 - self.alpha) * score
                }
        
        # Sort by combined score and take top-k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        return [(item['chunk'], item['score']) for item in sorted_results]


class QueryExpansionMixin:
    """Mixin for query expansion"""
    
    @staticmethod
    def expand_query(query: str) -> str:
        """Expand query with related terms"""
        expanded = query
        
        # Turkish legal terms expansion
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


class RetrievalWithExpansion(RetrievalStrategy, QueryExpansionMixin):
    """Wrapper strategy that adds query expansion"""
    
    def __init__(self, base_strategy: RetrievalStrategy):
        self.base_strategy = base_strategy
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve with query expansion"""
        expanded_query = self.expand_query(query)
        return self.base_strategy.retrieve(expanded_query, k)


class RetrievalWithReranking(RetrievalStrategy):
    """Wrapper strategy that adds reranking"""
    
    def __init__(
        self, 
        base_strategy: RetrievalStrategy,
        reranker,
        beta: float = 0.6
    ):
        """
        Args:
            base_strategy: Base retrieval strategy
            reranker: Cross-encoder model for reranking
            beta: Weight for reranker score (0-1)
        """
        self.base_strategy = base_strategy
        self.reranker = reranker
        self.beta = beta
    
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve with reranking"""
        if self.reranker is None:
            return self.base_strategy.retrieve(query, k)
        
        # Get more candidates for reranking
        retrieval_k = k * 2
        initial_results = self.base_strategy.retrieve(query, retrieval_k)
        
        if len(initial_results) <= k:
            return initial_results
        
        # Prepare pairs for reranking
        pairs = [[query, chunk['text']] for chunk, _ in initial_results]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine scores
        final_scores = []
        for i, (chunk, retrieval_score) in enumerate(initial_results):
            # Normalize reranker score (MS-MARCO range: ~[-10, 10])
            rerank_score = (rerank_scores[i] + 10) / 20
            rerank_score = max(0, min(1, rerank_score))
            
            # Weighted combination
            combined_score = self.beta * rerank_score + (1 - self.beta) * retrieval_score
            final_scores.append((chunk, combined_score))
        
        # Sort and return top-k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:k]


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategies"""
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        model=None,
        index=None,
        chunks=None,
        bm25=None,
        reranker=None,
        is_bge_m3=False,
        alpha=0.7,
        beta=0.6,
        use_expansion=True,
        use_reranking=True
    ) -> RetrievalStrategy:
        """
        Create retrieval strategy based on type
        
        Args:
            strategy_type: 'dense', 'sparse', 'hybrid'
            model: Embedding model
            index: FAISS index
            chunks: Document chunks
            bm25: BM25 index
            reranker: Reranker model
            is_bge_m3: Whether using BGE-M3
            alpha: Hybrid weight
            beta: Reranking weight
            use_expansion: Enable query expansion
            use_reranking: Enable reranking
            
        Returns:
            Configured retrieval strategy
        """
        # Create base strategy
        if strategy_type == 'dense':
            base_strategy = DenseRetrievalStrategy(model, index, chunks, is_bge_m3)
        elif strategy_type == 'sparse':
            base_strategy = SparseRetrievalStrategy(bm25, chunks)
        elif strategy_type == 'hybrid':
            dense = DenseRetrievalStrategy(model, index, chunks, is_bge_m3)
            sparse = SparseRetrievalStrategy(bm25, chunks)
            base_strategy = HybridRetrievalStrategy(dense, sparse, alpha)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Wrap with query expansion if enabled
        if use_expansion:
            base_strategy = RetrievalWithExpansion(base_strategy)
        
        # Wrap with reranking if enabled
        if use_reranking and reranker is not None:
            base_strategy = RetrievalWithReranking(base_strategy, reranker, beta)
        
        return base_strategy

