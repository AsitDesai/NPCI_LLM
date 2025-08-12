"""
JSON-specific semantic retriever for the RAG System.

This module provides semantic search and retrieval functionality
for JSON data using sentence-transformers/all-MiniLM-L6-v2 embeddings
and Qdrant vector store with payload filtering.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class JSONRetrievalResult:
    """Result from JSON semantic retrieval."""
    chunk_id: str
    text: str
    score: float
    payload: Dict[str, Any]
    category: str
    chunk_type: str
    source_file: str


class JSONSemanticRetriever:
    """
    JSON-specific semantic retriever for finding relevant documents.
    
    This class handles query embedding and vector similarity search
    using sentence-transformers/all-MiniLM-L6-v2 and Qdrant with payload filtering.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: Optional[str] = None):
        """
        Initialize the JSON semantic retriever.
        
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
        
        logger.info(f"JSON semantic retriever initialized with model: {model_name}")
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
    
    def _create_filter(self, 
                      category_filter: Optional[str] = None,
                      type_filter: Optional[str] = None) -> Optional[Filter]:
        """
        Create Qdrant filter for payload filtering.
        
        Args:
            category_filter: Filter by category
            type_filter: Filter by type
            
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
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                score_threshold: Optional[float] = None,
                category_filter: Optional[str] = None,
                type_filter: Optional[str] = None) -> List[JSONRetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            category_filter: Filter results by category
            type_filter: Filter results by type
            
        Returns:
            List of JSONRetrievalResult objects
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
            
            # Step 1: Generate query embedding with smart error handling
            try:
                query_embedding = self.embedding_model.encode(query, convert_to_list=True, batch_size=1)
                # Ensure embedding is proper Python list
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cublas" in error_msg:
                    logger.warning(f"GPU query encoding failed ({e}), switching to CPU")
                    import torch
                    torch.cuda.empty_cache()
                    # Reinitialize model on CPU
                    self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
                    query_embedding = self.embedding_model.encode(query, convert_to_list=True, batch_size=1)
                    # Ensure embedding is proper Python list
                    if hasattr(query_embedding, 'tolist'):
                        query_embedding = query_embedding.tolist()
                else:
                    raise e
            
            # Step 2: Create filter if needed
            qdrant_filter = self._create_filter(category_filter, type_filter)
            
            # Step 3: Search for similar embeddings in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True
            )
            
            # Step 4: Convert to JSONRetrievalResult objects
            retrieval_results = []
            for result in search_results:
                payload = result.payload
                
                # Extract text from payload (it was stored in the original chunk structure)
                # We need to reconstruct the text from the concatenated fields
                text = self._reconstruct_text_from_payload(payload)
                
                retrieval_result = JSONRetrievalResult(
                    chunk_id=result.id,
                    text=text,
                    score=result.score,
                    payload=payload,
                    category=payload.get("category", "unknown"),
                    chunk_type=payload.get("type", "unknown"),
                    source_file=payload.get("source_file", "unknown")
                )
                retrieval_results.append(retrieval_result)
            
            retrieval_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(retrieval_results)} results in {retrieval_time:.3f}s")
            logger.debug(f"Top result score: {retrieval_results[0].score if retrieval_results else 0:.3f}")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _reconstruct_text_from_payload(self, payload: Dict[str, Any]) -> str:
        """
        Reconstruct the original text from payload fields.
        
        Args:
            payload: Payload from Qdrant result
            
        Returns:
            Reconstructed text string
        """
        # Define the order and fields to reconstruct
        field_order = [
            'category',
            'scenario', 
            'user_statement',
            'agent_response',
            'system_behavior',
            'agent_guideline'
        ]
        
        text_parts = []
        
        for field in field_order:
            if field in payload and payload[field]:
                value = payload[field]
                if isinstance(value, str):
                    text_parts.append(f"{field}: {value}")
                else:
                    text_parts.append(f"{field}: {str(value)}")
        
        # Join all parts with double newlines for clear separation
        return "\n\n".join(text_parts)
    
    def retrieve_by_category(self, 
                           query: str, 
                           category: str,
                           top_k: Optional[int] = None,
                           score_threshold: Optional[float] = None) -> List[JSONRetrievalResult]:
        """
        Retrieve documents filtered by category.
        
        Args:
            query: User query text
            category: Category to filter by
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of JSONRetrievalResult objects
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            category_filter=category
        )
    
    def retrieve_by_type(self, 
                        query: str, 
                        chunk_type: str,
                        top_k: Optional[int] = None,
                        score_threshold: Optional[float] = None) -> List[JSONRetrievalResult]:
        """
        Retrieve documents filtered by type.
        
        Args:
            query: User query text
            chunk_type: Type to filter by
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of JSONRetrievalResult objects
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            type_filter=chunk_type
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the Qdrant collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
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
            # Get all points and extract categories from payload
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your data size
                with_payload=True
            )
            
            categories = set()
            for point in results[0]:  # results is a tuple (points, next_page_offset)
                if point.payload and "category" in point.payload:
                    categories.add(point.payload["category"])
            
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
            # Get all points and extract types from payload
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your data size
                with_payload=True
            )
            
            types = set()
            for point in results[0]:  # results is a tuple (points, next_page_offset)
                if point.payload and "type" in point.payload:
                    types.add(point.payload["type"])
            
            return sorted(list(types))
            
        except Exception as e:
            logger.error(f"Error getting available types: {e}")
            return []


def main():
    """Test the JSON semantic retriever."""
    print("🧪 JSON SEMANTIC RETRIEVER TEST")
    print("="*50)
    
    try:
        # Initialize retriever
        retriever = JSONSemanticRetriever()
        
        # Get collection info
        collection_info = retriever.get_collection_info()
        print(f"\n📊 COLLECTION INFO:")
        for key, value in collection_info.items():
            print(f"   {key}: {value}")
        
        # Get available categories and types
        categories = retriever.get_available_categories()
        types = retriever.get_available_types()
        
        print(f"\n📂 AVAILABLE CATEGORIES:")
        for category in categories:
            print(f"   - {category}")
        
        print(f"\n🏷️  AVAILABLE TYPES:")
        for chunk_type in types:
            print(f"   - {chunk_type}")
        
        # Test retrieval
        test_query = "insufficient balance"
        print(f"\n🔍 TESTING RETRIEVAL:")
        print(f"   Query: '{test_query}'")
        
        results = retriever.retrieve(test_query, top_k=3)
        
        if results:
            print(f"\n📋 RETRIEVAL RESULTS:")
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i}:")
                print(f"     Score: {result.score:.3f}")
                print(f"     Category: {result.category}")
                print(f"     Type: {result.chunk_type}")
                print(f"     Source: {result.source_file}")
                print(f"     Text: {result.text[:200]}...")
        else:
            print("   No results found")
        
        # Test category filtering
        if categories:
            print(f"\n🔍 TESTING CATEGORY FILTERING:")
            category_filter = categories[0]
            print(f"   Category filter: {category_filter}")
            
            category_results = retriever.retrieve_by_category(test_query, category_filter, top_k=2)
            
            if category_results:
                print(f"   Found {len(category_results)} results for category '{category_filter}'")
                for i, result in enumerate(category_results, 1):
                    print(f"     {i}. {result.category} - {result.chunk_type} (score: {result.score:.3f})")
            else:
                print("   No results found for this category")
        
        print(f"\n✅ JSON semantic retriever test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"JSON semantic retriever test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

