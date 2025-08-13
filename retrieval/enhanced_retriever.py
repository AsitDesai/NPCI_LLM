#!/usr/bin/env python3
"""
Enhanced semantic retriever supporting both JSON and TXT file formats.

This module provides semantic search and retrieval functionality
for both JSON and TXT data using sentence-transformers/all-MiniLM-L6-v2 embeddings
and Qdrant vector store with payload filtering.
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
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Result from enhanced semantic retrieval."""
    chunk_id: str
    text: str
    score: float
    payload: Dict[str, Any]
    category: str
    chunk_type: str
    source_file: str
    file_format: str  # 'json' or 'txt'


class EnhancedSemanticRetriever:
    """
    Enhanced semantic retriever for finding relevant documents from both JSON and TXT sources.
    
    This class handles query embedding and vector similarity search
    using sentence-transformers/all-MiniLM-L6-v2 and Qdrant with payload filtering.
    Supports both JSON and TXT data formats.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: Optional[str] = None):
        """
        Initialize the enhanced semantic retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            collection_name: Name of the Qdrant collection
        """
        self.model_name = model_name
        self.collection_name = collection_name or settings.vector_db_name
        
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
        
        logger.info(f"Enhanced semantic retriever initialized with model: {model_name}")
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
    
    def _create_filter(self, 
                      category_filter: Optional[str] = None,
                      type_filter: Optional[str] = None,
                      file_format_filter: Optional[str] = None) -> Optional[Filter]:
        """
        Create Qdrant filter for payload filtering.
        
        Args:
            category_filter: Filter by category
            type_filter: Filter by type
            file_format_filter: Filter by file format ('json' or 'txt')
            
        Returns:
            Qdrant Filter object or None
        """
        conditions = []
        
        if category_filter:
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category_filter)
                )
            )
        
        if type_filter:
            conditions.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=type_filter)
                )
            )
        
        if file_format_filter:
            conditions.append(
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=file_format_filter)
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                score_threshold: Optional[float] = None,
                category_filter: Optional[str] = None,
                type_filter: Optional[str] = None,
                file_format_filter: Optional[str] = None) -> List[EnhancedRetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            category_filter: Filter results by category
            type_filter: Filter results by type
            file_format_filter: Filter results by file format ('json' or 'txt')
            
        Returns:
            List of EnhancedRetrievalResult objects
        """
        try:
            start_time = time.time()
            
            # Use provided parameters or defaults
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.score_threshold
            
            logger.debug(f"Retrieving documents for query: '{query[:100]}...'")
            if category_filter:
                logger.debug(f"Category filter: {category_filter}")
            if type_filter:
                logger.debug(f"Type filter: {type_filter}")
            if file_format_filter:
                logger.debug(f"File format filter: {file_format_filter}")
            
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
            
            # Step 2: Create filter if needed
            filter_obj = self._create_filter(
                category_filter=category_filter,
                type_filter=type_filter,
                file_format_filter=file_format_filter
            )
            
            # Step 3: Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_obj,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Step 4: Convert to EnhancedRetrievalResult objects
            results = []
            for hit in search_results:
                payload = hit.payload
                
                # Determine file format from source file
                source_file = payload.get('source_file', '')
                file_format = 'txt' if source_file.endswith('.txt') else 'json'
                
                # Extract text from payload
                text = payload.get('text', '')
                if not text:
                    # Fallback: reconstruct text from individual fields
                    text_parts = []
                    for field in ['category', 'scenario', 'user_statement', 'agent_response', 'system_behavior', 'agent_guideline']:
                        if field in payload and payload[field]:
                            text_parts.append(f"{field}: {payload[field]}")
                    text = "\n\n".join(text_parts)
                
                result = EnhancedRetrievalResult(
                    chunk_id=payload.get('chunk_id', ''),
                    text=text,
                    score=hit.score,
                    payload=payload,
                    category=payload.get('category', ''),
                    chunk_type=payload.get('type', ''),
                    source_file=source_file,
                    file_format=file_format
                )
                results.append(result)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def retrieve_by_category(self, 
                           query: str, 
                           category: str,
                           top_k: Optional[int] = None) -> List[EnhancedRetrievalResult]:
        """
        Retrieve documents filtered by category.
        
        Args:
            query: User query text
            category: Category to filter by
            top_k: Number of results to return
            
        Returns:
            List of EnhancedRetrievalResult objects
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            category_filter=category
        )
    
    def retrieve_by_type(self, 
                        query: str, 
                        chunk_type: str,
                        top_k: Optional[int] = None) -> List[EnhancedRetrievalResult]:
        """
        Retrieve documents filtered by type.
        
        Args:
            query: User query text
            chunk_type: Type to filter by
            top_k: Number of results to return
            
        Returns:
            List of EnhancedRetrievalResult objects
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            type_filter=chunk_type
        )
    
    def retrieve_by_file_format(self, 
                               query: str, 
                               file_format: str,
                               top_k: Optional[int] = None) -> List[EnhancedRetrievalResult]:
        """
        Retrieve documents filtered by file format.
        
        Args:
            query: User query text
            file_format: File format to filter by ('json' or 'txt')
            top_k: Number of results to return
            
        Returns:
            List of EnhancedRetrievalResult objects
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            file_format_filter=file_format
        )
    
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
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available categories in the collection.
        
        Returns:
            List of category names
        """
        try:
            # Get all points to extract categories
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True
            )[0]
            
            categories = set()
            for point in all_points:
                category = point.payload.get('category')
                if category:
                    categories.add(category)
            
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting available categories: {e}")
            return []
    
    def get_available_types(self) -> List[str]:
        """
        Get list of available types in the collection.
        
        Returns:
            List of type names
        """
        try:
            # Get all points to extract types
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True
            )[0]
            
            types = set()
            for point in all_points:
                chunk_type = point.payload.get('type')
                if chunk_type:
                    types.add(chunk_type)
            
            return sorted(list(types))
            
        except Exception as e:
            logger.error(f"Error getting available types: {e}")
            return []
    
    def get_available_file_formats(self) -> List[str]:
        """
        Get list of available file formats in the collection.
        
        Returns:
            List of file format names
        """
        try:
            # Get all points to extract file formats
            all_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True
            )[0]
            
            formats = set()
            for point in all_points:
                source_file = point.payload.get('source_file', '')
                if source_file.endswith('.txt'):
                    formats.add('txt')
                elif source_file.endswith('.json'):
                    formats.add('json')
            
            return sorted(list(formats))
            
        except Exception as e:
            logger.error(f"Error getting available file formats: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.get_collection_info()
            categories = self.get_available_categories()
            types = self.get_available_types()
            formats = self.get_available_file_formats()
            
            return {
                "collection_info": collection_info,
                "categories": categories,
                "types": types,
                "file_formats": formats,
                "total_categories": len(categories),
                "total_types": len(types),
                "total_formats": len(formats)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}


def main():
    """Test the enhanced semantic retriever."""
    print("🧪 ENHANCED SEMANTIC RETRIEVER TEST")
    print("="*50)
    
    try:
        # Initialize retriever
        retriever = EnhancedSemanticRetriever()
        
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
        
        # Get available filters
        print("\n📊 AVAILABLE FILTERS:")
        categories = retriever.get_available_categories()
        types = retriever.get_available_types()
        formats = retriever.get_available_file_formats()
        
        print(f"   Categories ({len(categories)}): {categories[:5]}...")  # Show first 5
        print(f"   Types ({len(types)}): {types}")
        print(f"   File formats ({len(formats)}): {formats}")
        
        # Test retrieval
        print("\n🔍 TESTING RETRIEVAL:")
        test_queries = [
            "insufficient balance",
            "UPI PIN reset",
            "transaction failed"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            # General retrieval
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                for j, result in enumerate(results[:2]):  # Show first 2
                    print(f"     Result {j+1}: {result.category} - {result.chunk_type} (score: {result.score:.3f}, format: {result.file_format})")
            else:
                print(f"   ⚠️ No results found")
        
        # Test category filtering
        if categories:
            print(f"\n🔍 TESTING CATEGORY FILTERING:")
            category_filter = categories[0]
            print(f"   Category filter: {category_filter}")
            
            category_results = retriever.retrieve_by_category(
                "What should I do?", 
                category_filter,
                top_k=2
            )
            
            if category_results:
                print(f"   ✅ Found {len(category_results)} results for category '{category_filter}'")
                for result in category_results:
                    print(f"     - {result.chunk_type} (score: {result.score:.3f})")
            else:
                print(f"   ⚠️ No results for category '{category_filter}'")
        
        # Test file format filtering
        if formats:
            print(f"\n🔍 TESTING FILE FORMAT FILTERING:")
            for file_format in formats:
                print(f"   File format: {file_format}")
                
                format_results = retriever.retrieve_by_file_format(
                    "balance issue", 
                    file_format,
                    top_k=2
                )
                
                if format_results:
                    print(f"   ✅ Found {len(format_results)} results for {file_format} files")
                else:
                    print(f"   ⚠️ No results for {file_format} files")
        
        print(f"\n✅ Enhanced semantic retriever test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced semantic retriever test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
