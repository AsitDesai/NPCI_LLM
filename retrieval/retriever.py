"""
Semantic retriever for the RAG System.

This module provides semantic search and retrieval functionality
using LlamaIndex embeddings and Qdrant Cloud vector store.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from embeddings.embedder import LlamaIndexEmbedder
from embeddings.vector_store import QdrantVectorStore
from embeddings.models import get_default_embedding_config
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from semantic retrieval."""
    text: str
    score: float
    metadata: Dict[str, Any]
    source_document: str
    chunk_index: int


class SemanticRetriever:
    """
    Semantic retriever for finding relevant documents.
    
    This class handles query embedding and vector similarity search
    using the Qdrant Cloud vector store.
    """
    
    def __init__(self, 
                 embedder: Optional[LlamaIndexEmbedder] = None,
                 vector_store: Optional[QdrantVectorStore] = None):
        """
        Initialize the semantic retriever.
        
        Args:
            embedder: LlamaIndex embedder for query encoding
            vector_store: Qdrant vector store for similarity search
        """
        # Initialize embedder
        self.embedder = embedder or LlamaIndexEmbedder(get_default_embedding_config())
        
        # Initialize vector store
        self.vector_store = vector_store or QdrantVectorStore()
        
        # Retrieval settings
        self.top_k = getattr(settings, 'retrieval_top_k', 5)
        self.score_threshold = getattr(settings, 'retrieval_score_threshold', 0.3)
        
        logger.info(f"Semantic retriever initialized with top_k={self.top_k}, threshold={self.score_threshold}")
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                score_threshold: Optional[float] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieval results with text, scores, and metadata
        """
        try:
            start_time = time.time()
            
            # Use provided parameters or defaults
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.score_threshold
            
            logger.debug(f"Retrieving documents for query: '{query[:100]}...'")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Step 2: Search for similar embeddings in Qdrant Cloud
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # Step 3: Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    text=result['text'],
                    score=result['score'],
                    metadata=result['metadata'],
                    source_document=result['metadata'].get('doc_id', ''),
                    chunk_index=result['metadata'].get('chunk_index', 0)
                )
                retrieval_results.append(retrieval_result)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(retrieval_results)} documents in {processing_time:.3f}s")
            logger.debug(f"Top result score: {retrieval_results[0].score if retrieval_results else 'N/A'}")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_batch(self, 
                      queries: List[str], 
                      top_k: Optional[int] = None) -> List[List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            
        Returns:
            List of retrieval results for each query
        """
        try:
            start_time = time.time()
            
            # Generate embeddings for all queries
            query_embeddings = self.embedder.embed_texts(queries)
            
            # Search for each query
            all_results = []
            for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
                search_results = self.vector_store.search_similar(
                    query_embedding=embedding,
                    top_k=top_k or self.top_k,
                    score_threshold=self.score_threshold
                )
                
                # Convert to RetrievalResult objects
                query_results = []
                for result in search_results:
                    retrieval_result = RetrievalResult(
                        text=result['text'],
                        score=result['score'],
                        metadata=result['metadata'],
                        source_document=result['metadata'].get('doc_id', ''),
                        chunk_index=result['metadata'].get('chunk_index', 0)
                    )
                    query_results.append(retrieval_result)
                
                all_results.append(query_results)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Retrieved documents for {len(queries)} queries in {processing_time:.3f}s")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error during batch retrieval: {e}")
            return [[] for _ in queries]
    
    def get_retrieval_stats(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Get statistics about the retrieval results.
        
        Args:
            query: Original query
            results: Retrieval results
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                "query_length": len(query),
                "num_results": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "unique_sources": 0
            }
        
        scores = [result.score for result in results]
        sources = set(result.source_document for result in results)
        
        return {
            "query_length": len(query),
            "num_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "unique_sources": len(sources),
            "sources": list(sources)
        }
    
    def filter_results(self, 
                      results: List[RetrievalResult], 
                      min_score: float = 0.0,
                      max_results: Optional[int] = None) -> List[RetrievalResult]:
        """
        Filter retrieval results based on criteria.
        
        Args:
            results: List of retrieval results
            min_score: Minimum similarity score
            max_results: Maximum number of results to return
            
        Returns:
            Filtered list of retrieval results
        """
        # Filter by score
        filtered_results = [r for r in results if r.score >= min_score]
        
        # Sort by score (descending)
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit number of results
        if max_results:
            filtered_results = filtered_results[:max_results]
        
        logger.debug(f"Filtered {len(results)} results to {len(filtered_results)} results")
        
        return filtered_results
    
    def validate_retrieval(self, results: List[RetrievalResult]) -> bool:
        """
        Validate retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            True if results are valid, False otherwise
        """
        if not results:
            logger.warning("No retrieval results to validate")
            return False
        
        # Check for required fields
        for i, result in enumerate(results):
            if not result.text or not result.text.strip():
                logger.warning(f"Result {i} has empty text")
                return False
            
            if result.score < 0 or result.score > 1:
                logger.warning(f"Result {i} has invalid score: {result.score}")
                return False
        
        logger.debug(f"Validated {len(results)} retrieval results")
        return True 