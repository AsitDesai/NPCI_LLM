"""
Qdrant vector store integration for the RAG System.

This module provides integration with Qdrant vector database for storing
and retrieving embeddings with metadata.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaIndexQdrantStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorStoreStats:
    """Statistics about vector store operations."""
    total_points: int
    collection_name: str
    vector_dimension: int
    distance_metric: str
    storage_size: Optional[int] = None


class QdrantVectorStore:
    """
    Qdrant vector store wrapper for storing and retrieving embeddings.
    
    This class provides a simple interface for Qdrant operations
    with proper error handling and logging.
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the Qdrant vector store.
        
        Args:
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name or settings.vector_db_name
        self.vector_dimension = settings.vector_db_dimension
        self.distance_metric = settings.vector_db_metric
        
        # Initialize Qdrant client
        # Handle both local and cloud URLs
        if settings.qdrant_host.startswith(('http://', 'https://')):
            # For cloud URLs, use the url parameter (without port)
            self.client = QdrantClient(
                url=settings.qdrant_host,
                api_key=settings.qdrant_api_key,
                timeout=30.0  # Add timeout for cloud connections
            )
        else:
            # For local URLs, use host and port
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key
            )
        
        # Initialize LlamaIndex Qdrant store
        self.vector_store = LlamaIndexQdrantStore(
            client=self.client,
            collection_name=self.collection_name
        )
        
        logger.info(f"Qdrant vector store initialized: {self.collection_name}")
    
    def create_collection(self, vector_dimension: Optional[int] = None) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            vector_dimension: Dimension of vectors (uses settings if not provided)
            
        Returns:
            True if collection created successfully, False otherwise
        """
        try:
            dimension = vector_dimension or self.vector_dimension
            
            # Define vector parameters
            vector_params = VectorParams(
                size=dimension,
                distance=self._get_distance_metric()
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params
            )
            
            logger.info(f"Created Qdrant collection: {self.collection_name} ({dimension}d)")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {self.collection_name}: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the collection from Qdrant.
        
        Returns:
            True if collection deleted successfully, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return self.collection_name in collection_names
            
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def store_embeddings(self, 
                        embeddings: List[List[float]], 
                        nodes: List[TextNode],
                        batch_size: int = 100) -> bool:
        """
        Store embeddings with their metadata in Qdrant.
        
        Args:
            embeddings: List of embedding vectors
            nodes: List of TextNode objects with metadata
            batch_size: Number of points to insert in each batch
            
        Returns:
            True if all embeddings stored successfully, False otherwise
        """
        try:
            if len(embeddings) != len(nodes):
                raise ValueError("Number of embeddings must match number of nodes")
            
            # Prepare points for insertion
            points = []
            for i, (embedding, node) in enumerate(zip(embeddings, nodes)):
                # Create point with embedding and metadata
                point = PointStruct(
                    id=i,  # Simple sequential ID
                    vector=embedding,
                    payload={
                        "text": node.text,
                        "doc_id": node.metadata.get("source_document", ""),
                        "file_name": node.metadata.get("file_name", ""),
                        "file_type": node.metadata.get("file_type", ""),
                        "chunk_index": i,
                        **node.metadata  # Include all metadata
                    }
                )
                points.append(point)
            
            # Insert points in batches
            total_points = len(points)
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} points")
            
            logger.info(f"Successfully stored {total_points} embeddings in {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_similar(self, 
                      query_embedding: List[float], 
                      top_k: int = 5,
                      score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata
        """
        try:
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_result:
                # Extract text from the payload
                text = point.payload.get("text", "")
                
                # If text is empty or just "...", try to extract from _node_content
                if not text or text == "...":
                    node_content = point.payload.get("_node_content", "")
                    if node_content:
                        try:
                            import json
                            node_data = json.loads(node_content)
                            text = node_data.get("text", "")
                        except (json.JSONDecodeError, KeyError):
                            text = ""
                
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": text,
                    "metadata": {k: v for k, v in point.payload.items() if k not in ["text", "_node_content"]}
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 5,
               score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings (alias for search_similar).
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata
        """
        return self.search_similar(query_embedding, top_k, score_threshold)
    
    def get_collection_stats(self) -> Optional[VectorStoreStats]:
        """
        Get statistics about the collection.
        
        Returns:
            VectorStoreStats object or None if error
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = VectorStoreStats(
                total_points=collection_info.points_count,
                collection_name=self.collection_name,
                vector_dimension=collection_info.config.params.vectors.size,
                distance_metric=collection_info.config.params.vectors.distance.value
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def clear_collection(self) -> bool:
        """
        Clear all points from the collection.
        
        Returns:
            True if collection cleared successfully, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"all": True}
            )
            logger.info(f"Cleared all points from collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def _get_distance_metric(self) -> Distance:
        """
        Get the distance metric for vector similarity.
        
        Returns:
            Qdrant Distance enum value
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID
        }
        
        return distance_map.get(self.distance_metric.lower(), Distance.COSINE)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information for the vector store.
        
        Returns:
            Dictionary with connection details
        """
        return {
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "collection_name": self.collection_name,
            "vector_dimension": self.vector_dimension,
            "distance_metric": self.distance_metric,
            "api_key_configured": bool(settings.qdrant_api_key)
        } 