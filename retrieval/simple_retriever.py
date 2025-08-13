#!/usr/bin/env python3
"""
Simple Semantic Retriever

This script provides a simple semantic retrieval system that:
- Works with simple text chunks
- Uses sentence-transformers for semantic search
- No complex category/scenario segregation
- Simple and straightforward retrieval
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleRetrievalResult:
    """Result from simple semantic retrieval."""
    chunk_id: str
    text: str
    score: float
    source_file: str


class SimpleSemanticRetriever:
    """
    Simple semantic retriever for finding relevant documents.
    
    This class handles query embedding and vector similarity search
    using sentence-transformers/all-MiniLM-L6-v2 and Qdrant.
    No complex filtering or categorization.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "simple_embeddings"):
        """
        Initialize the simple semantic retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            collection_name: Name of the Qdrant collection
        """
        self.model_name = model_name
        self.collection_name = collection_name
        
        # Initialize sentence transformer model with smart device selection
        try:
            import torch
            
            # Check GPU memory availability
            gpu_available = False
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory
                free_gb = free_memory / 1024**3
                
                # Consider GPU available if we have at least 2GB free
                gpu_available = free_gb > 2.0
                logger.info(f"GPU memory: {free_gb:.1f}GB free, {allocated_memory/1024**3:.1f}GB allocated")
            
            if gpu_available:
                try:
                    torch.cuda.empty_cache()
                    self.embedding_model = SentenceTransformer(model_name, device='cuda')
                    logger.info(f"Embedding model initialized on GPU: {model_name}")
                except Exception as e:
                    logger.warning(f"GPU loading failed: {e}, falling back to CPU")
                    gpu_available = False
            
            if not gpu_available:
                self.embedding_model = SentenceTransformer(model_name, device='cpu')
                logger.info(f"Embedding model initialized on CPU: {model_name}")
                
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
        
        # Initialize Qdrant client
        self._init_qdrant_client()
        
        # Retrieval settings
        self.top_k = getattr(settings, 'retrieval_top_k', 5)
        self.score_threshold = getattr(settings, 'retrieval_score_threshold', 0.3)
        
        logger.info(f"Simple semantic retriever initialized with model: {model_name}")
        logger.info(f"Collection: {self.collection_name}, top_k={self.top_k}, threshold={self.score_threshold}")
    
    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        try:
            # Use local Qdrant server
            self.qdrant_client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                https=False
            )
            
            logger.info(f"Qdrant client initialized: {settings.qdrant_host}:{settings.qdrant_port}")
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    def _ensure_float_embedding(self, embedding):
        """Ensure embedding is proper float list for Qdrant."""
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Convert all values to float for Qdrant compatibility
        if isinstance(embedding, list):
            float_embedding = [float(val) for val in embedding]
            return float_embedding
        
        return embedding
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                score_threshold: Optional[float] = None) -> List[SimpleRetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of SimpleRetrievalResult objects
        """
        try:
            start_time = time.time()
            
            # Use provided parameters or defaults
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.score_threshold
            
            logger.debug(f"Retrieving documents for query: '{query[:100]}...'")
            
            # Step 1: Generate query embedding with smart error handling
            try:
                query_embedding = self.embedding_model.encode(query, convert_to_list=True, batch_size=1)
                query_embedding = self._ensure_float_embedding(query_embedding)
            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cublas" in error_msg:
                    logger.warning(f"GPU query encoding failed ({e}), switching to CPU")
                    import torch
                    torch.cuda.empty_cache()
                    # Reinitialize model on CPU
                    self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
                    query_embedding = self.embedding_model.encode(query, convert_to_list=True, batch_size=1)
                    query_embedding = self._ensure_float_embedding(query_embedding)
                else:
                    raise e
            
            # Step 2: Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Step 3: Convert to SimpleRetrievalResult objects
            results = []
            for hit in search_results:
                payload = hit.payload
                
                result = SimpleRetrievalResult(
                    chunk_id=payload.get('chunk_id', ''),
                    text=payload.get('text', ''),
                    score=hit.score,
                    source_file=payload.get('source_file', '')
                )
                results.append(result)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.get_collection_info()
            
            return {
                "collection_info": collection_info,
                "retrieval_settings": {
                    "top_k": self.top_k,
                    "score_threshold": self.score_threshold,
                    "model_name": self.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}


def main():
    """Test the simple semantic retriever."""
    print("🧪 SIMPLE SEMANTIC RETRIEVER TEST")
    print("="*50)
    
    try:
        # Initialize retriever
        retriever = SimpleSemanticRetriever()
        
        # Get collection info
        print("\n📊 COLLECTION INFORMATION:")
        collection_info = retriever.get_collection_info()
        
        if "error" not in collection_info:
            print(f"   Collection: {collection_info['collection_name']}")
            print(f"   Vector size: {collection_info['vector_size']}")
            print(f"   Points count: {collection_info['points_count']}")
            print(f"   Status: {collection_info['status']}")
        else:
            print(f"   ❌ Error getting collection info: {collection_info['error']}")
            return False
        
        # Test retrieval
        print("\n🔍 TESTING RETRIEVAL:")
        test_queries = [
            "insufficient balance",
            "UPI PIN reset",
            "transaction failed",
            "how to reset UPI PIN",
            "what to do when transaction fails"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            # General retrieval
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                for j, result in enumerate(results[:2]):  # Show first 2
                    print(f"     Result {j+1}: score={result.score:.3f}, source={result.source_file}")
                    print(f"     Text preview: {result.text[:80]}...")
            else:
                print(f"   ⚠️ No results found")
        
        print(f"\n✅ Simple semantic retriever test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Simple semantic retriever test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
