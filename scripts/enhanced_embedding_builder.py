#!/usr/bin/env python3
"""
Enhanced embedding generation script supporting both JSON and TXT file formats.

This script handles the complete embedding generation pipeline:
- JSON and TXT data processing using the enhanced data ingestion pipeline
- Embedding generation with sentence-transformers/all-MiniLM-L6-v2
- Vector storage in Qdrant with payload filtering
- Batch upload optimization
- Support for multiple file formats
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

from scripts.enhanced_data_ingestion import EnhancedDataIngestion, ChunkObject
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    chunk_id: str
    vector: List[float]
    payload: Dict[str, Any]


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
    batch_size: int
    json_files: int
    txt_files: int


class EnhancedEmbeddingBuilder:
    """
    Enhanced embedding generation and storage supporting both JSON and TXT formats.
    
    This class provides a complete embedding pipeline for both JSON and TXT data
    using sentence-transformers/all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the enhanced embedding builder.
        
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
        
        # Initialize enhanced data ingestion
        self.data_ingestion = EnhancedDataIngestion()
        
        logger.info(f"Enhanced embedding builder initialized with model: {self.model_name}")
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
            collection_name = collection_name or settings.vector_db_name
            
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
    
    def generate_embeddings(self, chunks: List[ChunkObject]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of ChunkObject instances
            
        Returns:
            List of EmbeddingResult instances
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
            
            # Create EmbeddingResult objects
            results = []
            for chunk, embedding in zip(chunks, all_embeddings):
                result = EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    vector=embedding,
                    payload=chunk.payload
                )
                results.append(result)
            
            logger.info(f"Generated {len(results)} embeddings in {embedding_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def upload_to_qdrant(self, 
                        embedding_results: List[EmbeddingResult], 
                        collection_name: Optional[str] = None,
                        batch_size: int = 100) -> bool:
        """
        Upload embedding results to Qdrant.
        
        Args:
            embedding_results: List of EmbeddingResult instances
            collection_name: Name of the collection to upload to
            batch_size: Batch size for upload operations
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            start_time = time.time()
            
            collection_name = collection_name or settings.vector_db_name
            
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
                        payload=result.payload
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
    
    def process_json_files(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process JSON files and generate embeddings.
        
        Args:
            collection_name: Name of the collection to use
            
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.json'], collection_name=collection_name)
    
    def process_txt_files(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process TXT files and generate embeddings.
        
        Args:
            collection_name: Name of the collection to use
            
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.txt'], collection_name=collection_name)
    
    def process_all_files(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all supported files (JSON and TXT) and generate embeddings.
        
        Args:
            collection_name: Name of the collection to use
            
        Returns:
            Dictionary with processing results and statistics
        """
        return self.process_files(file_extensions=['.json', '.txt'], collection_name=collection_name)
    
    def process_files(self, 
                     file_extensions: List[str], 
                     collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process files with specified extensions and generate embeddings.
        
        Args:
            file_extensions: List of file extensions to process
            collection_name: Name of the collection to use
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing files with extensions: {file_extensions}")
            
            # Process files using enhanced data ingestion
            ingestion_results = self.data_ingestion.process_files(file_extensions)
            
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
                vector_dimension=self.vector_dimension,
                batch_size=self.batch_size,
                json_files=ingestion_stats.json_files,
                txt_files=ingestion_stats.txt_files
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
    """Test the enhanced embedding builder."""
    print("🧪 ENHANCED EMBEDDING BUILDER TEST")
    print("="*50)
    
    try:
        # Initialize builder
        builder = EnhancedEmbeddingBuilder(batch_size=16)
        
        # Test TXT files (since we have TXT data)
        print("\n📄 TESTING TXT FILES EMBEDDING:")
        txt_results = builder.process_txt_files()
        
        if txt_results["success"]:
            stats = txt_results["statistics"]
            print(f"   ✅ TXT embedding successful")
            print(f"   Files processed: {stats.txt_files}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Embeddings generated: {stats.total_embeddings}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
            print(f"   Embedding time: {stats.embedding_time:.3f}s")
            print(f"   Upload time: {stats.upload_time:.3f}s")
            print(f"   Model: {stats.model_name}")
            print(f"   Vector dimension: {stats.vector_dimension}")
        else:
            print(f"   ❌ TXT embedding failed: {txt_results.get('error')}")
            return False
        
        # Test all files
        print("\n📄 TESTING ALL FILES EMBEDDING:")
        all_results = builder.process_all_files()
        
        if all_results["success"]:
            stats = all_results["statistics"]
            print(f"   ✅ Combined embedding successful")
            print(f"   JSON files: {stats.json_files}")
            print(f"   TXT files: {stats.txt_files}")
            print(f"   Total chunks: {stats.total_chunks}")
            print(f"   Embeddings generated: {stats.total_embeddings}")
            print(f"   Processing time: {stats.processing_time:.3f}s")
        else:
            print(f"   ❌ Combined embedding failed: {all_results.get('error')}")
            return False
        
        print(f"\n✅ Enhanced embedding builder test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced embedding builder test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)
