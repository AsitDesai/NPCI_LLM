#!/usr/bin/env python3
"""
Enhanced Retriever

This module implements enhanced retrieval with reranking, exact keyword filtering,
and hybrid search combining semantic similarity with keyword matching.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RerankingConfig:
    """Configuration for reranking and retrieval."""
    rerank_top_k: int = 50
    keyword_boost: float = 2.0
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    min_score_threshold: float = 0.1
    exact_match_boost: float = 1.5


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    text: str
    score: float
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    document_type: str
    structural_tags: List[str]


class EnhancedRetriever:
    """
    Enhanced retriever with reranking and hybrid search.
    
    Implements the architecture requirements:
    - Multi-stage retrieval with reranking
    - Exact keyword filtering
    - Hybrid search (semantic + keyword)
    - Metadata-based filtering
    - Fallback to exact text matching
    """
    
    def __init__(self, embedder, vector_store, config: Optional[RerankingConfig] = None):
        """Initialize the enhanced retriever."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or RerankingConfig()
        
        # Setup stop words and important terms
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Important terms that should not be filtered out
        self.important_terms = {
            'browser', 'download', 'data', 'support', 'cancel', 'subscription',
            'payment', 'billing', 'account', 'password', 'refund', 'credit',
            'card', 'paypal', 'apple', 'chrome', 'firefox', 'safari', 'edge',
            'api', 'integration', 'security', 'encryption', 'pci', 'dss'
        }
        
        logger.info("Enhanced retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Enhanced retrieval with multi-stage approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        try:
            # Step 1: Initial semantic retrieval
            initial_results = self._semantic_retrieval(query, self.config.rerank_top_k)
            
            if not initial_results:
                # Fallback to exact text search
                logger.warning("No semantic results found, trying exact text search")
                return self._exact_text_search(query, top_k)
            
            # Step 2: Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Step 3: Rerank results with hybrid scoring
            reranked_results = self._rerank_results(initial_results, query, keywords)
            
            # Step 4: Filter by score threshold
            filtered_results = [
                result for result in reranked_results 
                if result.score >= self.config.min_score_threshold
            ]
            
            # Step 5: Return top_k results
            final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} results for query: {query}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return []
    
    def _semantic_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform initial semantic retrieval."""
        try:
            # Generate query embedding
            query_embedding, _ = self.embedder.generate_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding.cpu().numpy(),
                top_k=top_k
            )
            
            # Convert results to expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'text': result.get('text', ''),
                    'score': result.get('score', 0.0),
                    'metadata': result.get('metadata', {})
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: Search query
            
        Returns:
            List of extracted keywords
        """
        # Tokenize query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words but keep important terms
        keywords = []
        for word in words:
            if word not in self.stop_words or word in self.important_terms:
                keywords.append(word)
        
        # Also extract multi-word phrases
        phrases = self._extract_phrases(query)
        keywords.extend(phrases)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query."""
        phrases = []
        
        # Common phrases in our domain
        domain_phrases = [
            'payment status', 'billing information', 'account settings',
            'data export', 'browser support', 'customer support',
            'credit card', 'debit card', 'digital wallet', 'apple pay',
            'money back', 'free trial', 'subscription plan'
        ]
        
        query_lower = query.lower()
        for phrase in domain_phrases:
            if phrase in query_lower:
                phrases.append(phrase)
        
        return phrases
    
    def _rerank_results(self, initial_results: List[Dict[str, Any]], 
                       query: str, keywords: List[str]) -> List[RetrievalResult]:
        """
        Rerank results using hybrid scoring.
        
        Args:
            initial_results: Initial retrieval results
            query: Original query
            keywords: Extracted keywords
            
        Returns:
            Reranked results
        """
        reranked_results = []
        
        for result in initial_results:
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            semantic_score = result.get('score', 0.0)
            
            # Calculate keyword score
            keyword_score = self._calculate_keyword_score(text, keywords)
            
            # Calculate hybrid score
            hybrid_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score
            )
            
            # Apply exact match boost
            if self._has_exact_match(text, query, keywords):
                hybrid_score *= self.config.exact_match_boost
            
            # Create retrieval result
            retrieval_result = RetrievalResult(
                text=text,
                score=hybrid_score,
                metadata=metadata,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                document_type=metadata.get('document_type', 'unknown'),
                structural_tags=metadata.get('structural_tags', [])
            )
            
            reranked_results.append(retrieval_result)
        
        # Sort by hybrid score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_results
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword match score.
        
        Args:
            text: Document text
            keywords: Query keywords
            
        Returns:
            Keyword score (0.0 to 1.0)
        """
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
        
        return matches / len(keywords)
    
    def _has_exact_match(self, text: str, query: str, keywords: List[str]) -> bool:
        """
        Check for exact matches between query and text.
        
        Args:
            text: Document text
            query: Search query
            keywords: Query keywords
            
        Returns:
            True if exact match found
        """
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Check for exact phrase matches
        for keyword in keywords:
            if len(keyword.split()) > 1:  # Multi-word keyword
                if keyword.lower() in text_lower:
                    return True
        
        # Check for query terms
        query_terms = query_lower.split()
        for term in query_terms:
            if term in text_lower and term not in self.stop_words:
                return True
        
        return False
    
    def _exact_text_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Fallback exact text search when semantic retrieval fails.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of retrieval results
        """
        try:
            # Get all documents from vector store
            all_points = self.vector_store.client.scroll(
                collection_name="rag_embeddings",
                limit=100
            )
            
            results = []
            query_lower = query.lower()
            
            for point in all_points[0]:  # scroll returns (points, next_page_offset)
                text = point.payload.get('text', '')
                if text and query_lower in text.lower():
                    # Calculate simple relevance score
                    relevance = len(set(query_lower.split()) & set(text.lower().split())) / len(query_lower.split())
                    
                    result = RetrievalResult(
                        text=text,
                        score=relevance,
                        metadata=point.payload.get('metadata', {}),
                        semantic_score=0.0,
                        keyword_score=relevance,
                        document_type=point.payload.get('metadata', {}).get('document_type', 'unknown'),
                        structural_tags=point.payload.get('metadata', {}).get('structural_tags', [])
                    )
                    results.append(result)
            
            # Sort by relevance and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in exact text search: {e}")
            return []
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.
        
        Args:
            results: Retrieval results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {
                'total_results': 0,
                'avg_semantic_score': 0.0,
                'avg_keyword_score': 0.0,
                'avg_hybrid_score': 0.0,
                'document_types': {},
                'keyword_matches': 0
            }
        
        # Calculate averages
        avg_semantic = sum(r.semantic_score for r in results) / len(results)
        avg_keyword = sum(r.keyword_score for r in results) / len(results)
        avg_hybrid = sum(r.score for r in results) / len(results)
        
        # Document type distribution
        doc_types = Counter(r.document_type for r in results)
        
        # Keyword match count
        keyword_matches = sum(1 for r in results if r.keyword_score > 0)
        
        return {
            'total_results': len(results),
            'avg_semantic_score': avg_semantic,
            'avg_keyword_score': avg_keyword,
            'avg_hybrid_score': avg_hybrid,
            'document_types': dict(doc_types),
            'keyword_matches': keyword_matches
        }
