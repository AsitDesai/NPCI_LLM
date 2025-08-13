#!/usr/bin/env python3
"""
Simple Embedding Builder

This script provides a simple embedding generation pipeline that:
- Works with simple text chunks
- Generates embeddings using sentence-transformers
- Stores embeddings in Qdrant with minimal metadata
- No complex category/scenario segregation
"""

import os
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
from qdrant_client.models import Distance, VectorParams, PointStruct

from scripts.simple_data_ingestion import SimpleDataIngestion, SimpleChunk
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleEmbeddingResult:
    """Result from simple embedding generation."""
    chunk_id: str
    vector: List[float]
    text: str
    source_file: str


@dataclass
class EmbeddingStats:
    """Statistics about embedding operations."""
    total_chunks: int
    total_embeddings: int
    processing_time: float
    embedding_time: float
    upload_time: float
    model_name: str
    vector_dimension: int


class SimpleEmbeddingBuilder:
    """
    Simple embedding generation and storage.
    
    This class provides a straightforward embedding pipeline for simple text chunks
    using sentence-transformers/all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the simple embedding builder.
        
        Args:
            batch_size: Batch size for embedding generation
        """
        self.batch_size = batch_size
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.vector_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
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
                    self.embedding_model = SentenceTransformer(self.model_name, device='cuda')
                    logger.info(f"Embedding model initialized on GPU: {self.model_name}")
                except Exception as e:
                    logger.warning(f"GPU loading failed: {e}, falling back to CPU")
                    gpu_available = False
            
            if not gpu_available:
                self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
                logger.info(f"Embedding model initialized on CPU: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
        
        # Initialize Qdrant client
        self._init_qdrant_client()
        
        # Initialize simple data ingestion
        self.data_ingestion = SimpleDataIngestion()
        
        logger.info(f"Simple embedding builder initialized with model: {self.model_name}")
        logger.info(f"Vector dimension: {self.vector_dimension}, Batch size: {batch_size}")
    
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
    
    def create_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            True if collection created successfully, False otherwise
        """
        try:
            collection_name = collection_name or "simple_embeddings"
            
            # Check if collection already exists
            collections = self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Define vector parameters
            vector_params = VectorParams(
                size=self.vector_dimension,
                distance=Distance.COSINE
            )
            
            # Create collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params
            )
            
            logger.info(f"Created collection '{collection_name}' with dimension {self.vector_dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def _ensure_float_embeddings(self, embeddings):
        """Ensure embeddings are proper float lists for Qdrant."""
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        
        if isinstance(embeddings, list):
            if len(embeddings) > 0 and hasattr(embeddings[0], 'tolist'):
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            
            # Convert all values to float for Qdrant compatibility
            float_embeddings = []
            for emb in embeddings:
                float_emb = [float(val) for val in emb]
                float_embeddings.append(float_emb)
            return float_embeddings
        
        return embeddings
    
    def generate_embeddings(self, chunks: List[SimpleChunk]) -> List[SimpleEmbeddingResult]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of SimpleChunk instances
            
        Returns:
            List of SimpleEmbeddingResult instances
        """
        try:
            start_time = time.time()
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Extract texts for batch processing
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_chunks = chunks[i:i + self.batch_size]
                
                logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch_texts)} texts")
                
                # Generate embeddings for batch with smart error handling
                try:
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        convert_to_list=True, 
                        batch_size=1  # Use smaller batch size for stability
                    )
                    
                    # Ensure embeddings are proper float lists for Qdrant
                    batch_embeddings = self._ensure_float_embeddings(batch_embeddings)
                    
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "cublas" in error_msg:
                        # If GPU encoding fails, force CPU mode
                        logger.warning(f"GPU encoding failed ({e}), forcing CPU mode")
                        self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
                        batch_embeddings = self.embedding_model.encode(
                            batch_texts, 
                            convert_to_list=True, 
                            batch_size=1
                        )
                        batch_embeddings = self._ensure_float_embeddings(batch_embeddings)
                        all_embeddings.extend(batch_embeddings)
                    else:
                        raise e
            
            embedding_time = time.time() - start_time
            
            # Create SimpleEmbeddingResult objects
            results = []
            for chunk, embedding in zip(chunks, all_embeddings):
                result = SimpleEmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    vector=embedding,
                    text=chunk.text,
                    source_file=chunk.source_file
                )
                results.append(result)
            
            logger.info(f"Generated {len(results)} embeddings in {embedding_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def upload_to_qdrant(self, 
                        embedding_results: List[SimpleEmbeddingResult], 
                        collection_name: Optional[str] = None,
                        batch_size: int = 100) -> bool:
        """
        Upload embedding results to Qdrant.
        
        Args:
            embedding_results: List of SimpleEmbeddingResult instances
            collection_name: Name of the collection to upload to
            batch_size: Batch size for upload operations
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            start_time = time.time()
            
            collection_name = collection_name or "simple_embeddings"
            
            logger.info(f"Uploading {len(embedding_results)} embeddings to collection '{collection_name}'")
            
            # Create collection if it doesn't exist
            if not self.create_collection(collection_name):
                return False
            
            # Upload in batches
            for i in range(0, len(embedding_results), batch_size):
                batch_results = embedding_results[i:i + batch_size]
                
                # Create points for batch
                points = []
                for j, result in enumerate(batch_results):
                    point = PointStruct(
                        id=i + j,  # Use batch index for unique IDs
                        vector=result.vector,
                        payload={
                            "chunk_id": result.chunk_id,
                            "text": result.text,
                            "source_file": result.source_file,
                            "upload_timestamp": time.time()
                        }
                    )
                    points.append(point)
                
                # Upload batch
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                logger.debug(f"Uploaded batch {i//batch_size + 1}: {len(points)} points")
            
            upload_time = time.time() - start_time
            logger.info(f"Uploaded {len(embedding_results)} embeddings in {upload_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
            return False
    
    def process_all_files(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all text files and generate embeddings.
        
        Args:
            collection_name: Name of the collection to use
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing all text files for embedding generation")
            
            # Process files using simple data ingestion
            ingestion_results = self.data_ingestion.process_all_files()
            
            if not ingestion_results["success"]:
                return {
                    "success": False,
                    "error": ingestion_results.get("error", "Unknown error"),
                    "statistics": None
                }
            
            chunks = ingestion_results["chunks"]
            ingestion_stats = ingestion_results["statistics"]
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No valid chunks found",
                    "statistics": None
                }
            
            logger.info(f"Processing {len(chunks)} chunks for embedding generation")
            
            # Generate embeddings
            embedding_start = time.time()
            embedding_results = self.generate_embeddings(chunks)
            embedding_time = time.time() - embedding_start
            
            # Upload to Qdrant
            upload_start = time.time()
            upload_success = self.upload_to_qdrant(embedding_results, collection_name)
            upload_time = time.time() - upload_start
            
            if not upload_success:
                return {
                    "success": False,
                    "error": "Failed to upload embeddings to Qdrant",
                    "statistics": None
                }
            
            total_time = time.time() - start_time
            
            # Create statistics
            stats = EmbeddingStats(
                total_chunks=len(chunks),
                total_embeddings=len(embedding_results),
                processing_time=total_time,
                embedding_time=embedding_time,
                upload_time=upload_time,
                model_name=self.model_name,
                vector_dimension=self.vector_dimension
            )
            
            logger.info(f"Embedding generation completed successfully")
            logger.info(f"Total time: {total_time:.3f}s (embedding: {embedding_time:.3f}s, upload: {upload_time:.3f}s)")
            
            return {
                "success": True,
                "embedding_results": embedding_results,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error in embedding generation pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": None
            }


def main():
    """Test the simple embedding builder."""
    print("🧪 SIMPLE EMBEDDING BUILDER TEST")
    print("="*50)
    
    try:
        # Initialize builder
        builder = SimpleEmbeddingBuilder(batch_size=16)
        
        # Process all files
        print("\n📄 PROCESSING TEXT FILES AND GENERATING EMBEDDINGS:")
        results = builder.process_all_files()
        
        if results["success"]:
            stats = results["statistics"]
            print(f"   ✅ Processing successful")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Embeddings generated: {stats.total_embeddings}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
            print(f"   Embedding time: {stats.embedding_time:.3f}s")
            print(f"   Upload time: {stats.upload_time:.3f}s")
            print(f"   Model: {stats.model_name}")
            print(f"   Vector dimension: {stats.vector_dimension}")
            
            # Show sample embeddings
            if results["embedding_results"]:
                print(f"\n📋 SAMPLE EMBEDDINGS:")
                for i, result in enumerate(results["embedding_results"][:3]):
                    print(f"   Embedding {i+1}:")
                    print(f"     Chunk ID: {result.chunk_id}")
                    print(f"     Source: {result.source_file}")
                    print(f"     Vector dimension: {len(result.vector)}")
                    print(f"     Text preview: {result.text[:80]}...")
                    print()
        else:
            print(f"   ❌ Processing failed: {results.get('error')}")
            return False
        
        print(f"\n✅ Simple embedding builder test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Simple embedding builder test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
