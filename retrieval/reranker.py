"""
Reranker for the RAG System.

This module provides optional reranking functionality for
retrieval results to improve relevance and quality.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .retriever import RetrievalResult
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking."""
    original_result: RetrievalResult
    rerank_score: float
    rerank_rank: int
    metadata: Dict[str, Any]


class Reranker:
    """
    Reranker for improving retrieval result quality.
    
    This class provides optional reranking functionality to
    improve the relevance of retrieved results.
    """
    
    def __init__(self, 
                 use_reranking: bool = False,
                 rerank_top_k: int = 10):
        """
        Initialize the reranker.
        
        Args:
            use_reranking: Whether to enable reranking
            rerank_top_k: Number of top results to rerank
        """
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        
        logger.info(f"Reranker initialized: use_reranking={use_reranking}, top_k={rerank_top_k}")
    
    def rerank(self, 
              results: List[RetrievalResult],
              query: str = "") -> List[RetrievalResult]:
        """
        Rerank retrieval results.
        
        Args:
            results: List of retrieval results
            query: Original query
            
        Returns:
            Reranked list of retrieval results
        """
        if not self.use_reranking:
            logger.debug("Reranking disabled, returning original results")
            return results
        
        if not results:
            logger.warning("No results to rerank")
            return results
        
        try:
            logger.debug(f"Reranking {len(results)} results")
            
            # For now, implement a simple score-based reranking
            # In the future, this could use cross-encoders or other methods
            
            # Take top k results for reranking
            top_results = results[:self.rerank_top_k]
            
            # Apply simple reranking (score adjustment based on content quality)
            reranked_results = self._simple_rerank(top_results, query)
            
            # Add remaining results without reranking
            if len(results) > self.rerank_top_k:
                reranked_results.extend(results[self.rerank_top_k:])
            
            logger.info(f"Reranked {len(top_results)} results")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results
    
    def _simple_rerank(self, 
                      results: List[RetrievalResult], 
                      query: str) -> List[RetrievalResult]:
        """
        Simple reranking based on content quality heuristics.
        
        Args:
            results: List of retrieval results
            query: Original query
            
        Returns:
            Reranked list of results
        """
        reranked = []
        
        for result in results:
            # Calculate additional quality score
            quality_score = self._calculate_quality_score(result, query)
            
            # Combine original score with quality score
            combined_score = (result.score + quality_score) / 2
            
            # Create new result with adjusted score
            reranked_result = RetrievalResult(
                text=result.text,
                score=combined_score,
                metadata=result.metadata,
                source_document=result.source_document,
                chunk_index=result.chunk_index
            )
            
            reranked.append(reranked_result)
        
        # Sort by new combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked
    
    def _calculate_quality_score(self, result: RetrievalResult, query: str) -> float:
        """
        Calculate quality score for a result.
        
        Args:
            result: Retrieval result
            query: Original query
            
        Returns:
            Quality score between 0 and 1
        """
        text = result.text.lower()
        query_terms = query.lower().split()
        
        # Simple quality heuristics
        quality_factors = []
        
        # 1. Query term coverage
        query_coverage = sum(1 for term in query_terms if term in text) / len(query_terms) if query_terms else 0
        quality_factors.append(query_coverage)
        
        # 2. Text length (prefer medium length)
        text_length = len(text)
        if text_length < 50:
            length_score = 0.3  # Too short
        elif text_length < 500:
            length_score = 1.0  # Good length
        elif text_length < 1000:
            length_score = 0.8  # Long but acceptable
        else:
            length_score = 0.5  # Too long
        quality_factors.append(length_score)
        
        # 3. Content structure (prefer structured content)
        has_structure = any(marker in text for marker in ['#', '##', '1.', '2.', '-', '*'])
        structure_score = 1.0 if has_structure else 0.7
        quality_factors.append(structure_score)
        
        # 4. Source quality (could be enhanced with source metadata)
        source_score = 1.0  # Default score, could be based on source reputation
        quality_factors.append(source_score)
        
        # Calculate average quality score
        avg_quality = sum(quality_factors) / len(quality_factors)
        
        return avg_quality
    
    def filter_duplicates(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Filter out duplicate or very similar results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Filtered list without duplicates
        """
        if not results:
            return results
        
        filtered = []
        seen_texts = set()
        
        for result in results:
            # Create a normalized version of the text for comparison
            normalized_text = self._normalize_text(result.text)
            
            if normalized_text not in seen_texts:
                filtered.append(result)
                seen_texts.add(normalized_text)
            else:
                logger.debug(f"Filtered duplicate result: {result.text[:100]}...")
        
        logger.info(f"Filtered {len(results) - len(filtered)} duplicate results")
        
        return filtered
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for duplicate detection.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation
        normalized = normalized.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        
        return normalized
    
    def get_rerank_stats(self, 
                        original_results: List[RetrievalResult],
                        reranked_results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Get statistics about the reranking process.
        
        Args:
            original_results: Original retrieval results
            reranked_results: Reranked results
            
        Returns:
            Dictionary with reranking statistics
        """
        if not original_results or not reranked_results:
            return {
                "original_count": len(original_results),
                "reranked_count": len(reranked_results),
                "avg_score_change": 0.0,
                "max_score_change": 0.0,
                "rank_changes": []
            }
        
        # Calculate score changes
        score_changes = []
        rank_changes = []
        
        for i, (orig, rerank) in enumerate(zip(original_results, reranked_results)):
            score_change = rerank.score - orig.score
            score_changes.append(score_change)
            
            # Find rank change
            orig_rank = i
            rerank_rank = next((j for j, r in enumerate(reranked_results) if r.text == orig.text), i)
            rank_change = orig_rank - rerank_rank
            rank_changes.append(rank_change)
        
        return {
            "original_count": len(original_results),
            "reranked_count": len(reranked_results),
            "avg_score_change": sum(score_changes) / len(score_changes),
            "max_score_change": max(score_changes),
            "min_score_change": min(score_changes),
            "avg_rank_change": sum(rank_changes) / len(rank_changes),
            "rank_changes": rank_changes,
            "score_changes": score_changes
        } 