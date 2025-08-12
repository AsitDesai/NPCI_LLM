#!/usr/bin/env python3
"""
JSON-specific embedding generation script using sentence-transformers/all-MiniLM-L6-v2.

This script handles the complete JSON embedding generation pipeline:
- JSON data processing using the new JSONDataIngestion pipeline
- Embedding generation with sentence-transformers/all-MiniLM-L6-v2
- Vector storage in Qdrant with payload filtering
- Batch upload optimization
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

from scripts.json_data_ingestion import JSONDataIngestion, ChunkObject
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


class JSONEmbeddingBuilder:
    """
    JSON-specific embedding generation and storage.
    
    This class provides a complete embedding pipeline for JSON data
    using sentence-transformers/all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the JSON embedding builder.
        
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
        
        # Initialize JSON data ingestion
        self.json_ingestion = JSONDataIngestion()
        
        logger.info(f"JSON embedding builder initialized with model: {self.model_name}")
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
                    batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_list=True, batch_size=1)
                    # Ensure embeddings are proper Python lists
                    if hasattr(batch_embeddings, 'tolist'):
                        batch_embeddings = batch_embeddings.tolist()
                    elif isinstance(batch_embeddings, list):
                        batch_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings]
                except Exception as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "cublas" in error_msg:
                        logger.warning(f"GPU encoding failed ({e}), switching to CPU")
                        import torch
                        torch.cuda.empty_cache()
                        # Reinitialize model on CPU
                        self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
                        batch_embeddings = self.embedding_model.encode(batch_texts, convert_to_list=True, batch_size=1)
                        # Ensure embeddings are proper Python lists
                        if hasattr(batch_embeddings, 'tolist'):
                            batch_embeddings = batch_embeddings.tolist()
                        elif isinstance(batch_embeddings, list):
                            batch_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings]
                    else:
                        raise e
                
                # Create EmbeddingResult objects
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    embedding_result = EmbeddingResult(
                        chunk_id=chunk.chunk_id,
                        vector=embedding,
                        payload=chunk.payload
                    )
                    all_embeddings.append(embedding_result)
            
            embedding_time = time.time() - start_time
            
            logger.info(f"Generated {len(all_embeddings)} embeddings in {embedding_time:.3f}s")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def upload_to_qdrant(self, 
                        embedding_results: List[EmbeddingResult], 
                        collection_name: Optional[str] = None) -> bool:
        """
        Upload embedding results to Qdrant.
        
        Args:
            embedding_results: List of EmbeddingResult instances
            collection_name: Name of the collection to upload to
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            start_time = time.time()
            
            collection_name = collection_name or settings.vector_db_name
            
            logger.info(f"Uploading {len(embedding_results)} embeddings to collection '{collection_name}'")
            
            # Prepare points for batch upload
            points = []
            
            for result in embedding_results:
                point = PointStruct(
                    id=result.chunk_id,
                    vector=result.vector,
                    payload=result.payload
                )
                points.append(point)
            
            # Upload in batches
            batch_size = 100  # Qdrant batch size
            total_uploaded = 0
            
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                
                total_uploaded += len(batch_points)
                logger.debug(f"Uploaded batch {i//batch_size + 1}: {len(batch_points)} points")
            
            upload_time = time.time() - start_time
            
            logger.info(f"Successfully uploaded {total_uploaded} embeddings in {upload_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
            return False
    
    def create_payload_indexes(self, collection_name: Optional[str] = None) -> bool:
        """
        Create payload indexes for filtering on 'type' and 'category'.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if indexes created successfully, False otherwise
        """
        try:
            collection_name = collection_name or settings.vector_db_name
            
            logger.info(f"Creating payload indexes for collection '{collection_name}'")
            
            # Create index on 'type' field
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="type",
                field_schema="keyword"
            )
            
            # Create index on 'category' field
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_schema="keyword"
            )
            
            logger.info("Payload indexes created successfully for 'type' and 'category'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating payload indexes: {e}")
            return False
    
    def build_embeddings_pipeline(self, documents_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete JSON embedding generation pipeline.
        
        Args:
            documents_dir: Directory containing JSON files
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info("Starting JSON embedding generation pipeline")
            
            # Step 1: Process JSON files
            processing_start = time.time()
            json_results = self.json_ingestion.process_json_files(documents_dir)
            processing_time = time.time() - processing_start
            
            if not json_results["success"]:
                logger.error(f"JSON processing failed: {json_results.get('error')}")
                return {
                    "success": False,
                    "error": json_results.get("error", "JSON processing failed"),
                    "statistics": {}
                }
            
            chunks = json_results["chunks"]
            if not chunks:
                logger.warning("No chunks to process")
                return {
                    "success": False,
                    "error": "No chunks to process",
                    "statistics": {}
                }
            
            # Step 2: Create Qdrant collection
            collection_start = time.time()
            collection_created = self.create_collection()
            collection_time = time.time() - collection_start
            
            if not collection_created:
                logger.error("Failed to create Qdrant collection")
                return {
                    "success": False,
                    "error": "Failed to create Qdrant collection",
                    "statistics": {}
                }
            
            # Step 3: Generate embeddings
            embedding_start = time.time()
            embedding_results = self.generate_embeddings(chunks)
            embedding_time = time.time() - embedding_start
            
            # Step 4: Upload to Qdrant
            upload_start = time.time()
            upload_success = self.upload_to_qdrant(embedding_results)
            upload_time = time.time() - upload_start
            
            if not upload_success:
                logger.error("Failed to upload embeddings to Qdrant")
                return {
                    "success": False,
                    "error": "Failed to upload embeddings to Qdrant",
                    "statistics": {}
                }
            
            # Step 5: Create payload indexes
            index_start = time.time()
            indexes_created = self.create_payload_indexes()
            index_time = time.time() - index_start
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            stats = EmbeddingStats(
                total_chunks=len(chunks),
                total_embeddings=len(embedding_results),
                processing_time=processing_time,
                embedding_time=embedding_time,
                upload_time=upload_time,
                model_name=self.model_name,
                vector_dimension=self.vector_dimension,
                batch_size=self.batch_size
            )
            
            logger.info(f"JSON embedding pipeline completed in {total_time:.3f}s")
            
            return {
                "success": True,
                "chunks": chunks,
                "embedding_results": embedding_results,
                "statistics": stats,
                "timing": {
                    "processing_time": processing_time,
                    "collection_time": collection_time,
                    "embedding_time": embedding_time,
                    "upload_time": upload_time,
                    "index_time": index_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"JSON embedding pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def get_pipeline_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed pipeline statistics.
        
        Args:
            results: Results from build_embeddings_pipeline
            
        Returns:
            Dictionary with detailed statistics
        """
        if not results.get("success", False):
            return {"error": "Pipeline failed"}
        
        stats = results["statistics"]
        timing = results.get("timing", {})
        
        # Get JSON processing stats
        json_stats = self.json_ingestion.get_processing_stats({
            "success": True,
            "chunks": results.get("chunks", [])
        })
        
        detailed_stats = {
            "model_info": {
                "name": stats.model_name,
                "dimension": stats.vector_dimension,
                "batch_size": stats.batch_size
            },
            "processing_stats": {
                "total_chunks": stats.total_chunks,
                "total_embeddings": stats.total_embeddings,
                "embedding_success_rate": stats.total_embeddings / stats.total_chunks if stats.total_chunks > 0 else 0
            },
            "timing_breakdown": {
                "json_processing": timing.get("processing_time", 0),
                "collection_creation": timing.get("collection_time", 0),
                "embedding_generation": timing.get("embedding_time", 0),
                "qdrant_upload": timing.get("upload_time", 0),
                "index_creation": timing.get("index_time", 0),
                "total_time": timing.get("total_time", 0)
            },
            "performance_metrics": {
                "embeddings_per_second": stats.total_embeddings / timing.get("embedding_time", 1) if timing.get("embedding_time", 0) > 0 else 0,
                "upload_rate": stats.total_embeddings / timing.get("upload_time", 1) if timing.get("upload_time", 0) > 0 else 0
            },
            "json_processing_details": json_stats
        }
        
        return detailed_stats


def main():
    """Run the JSON embedding generation pipeline."""
    print("🚀 JSON EMBEDDING GENERATION PIPELINE")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = JSONEmbeddingBuilder(batch_size=32)
        
        # Build embeddings
        results = pipeline.build_embeddings_pipeline()
        
        if results["success"]:
            print("\n✅ JSON EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
            
            # Display statistics
            stats = pipeline.get_pipeline_stats(results)
            
            print(f"\n📊 PIPELINE STATISTICS:")
            model_info = stats.get('model_info', {})
            print(f"   Model: {model_info.get('name', 'Unknown')}")
            print(f"   Vector dimension: {model_info.get('dimension', 0)}")
            print(f"   Batch size: {model_info.get('batch_size', 0)}")
            
            processing_stats = stats.get('processing_stats', {})
            print(f"\n📈 PROCESSING STATISTICS:")
            print(f"   Total chunks: {processing_stats.get('total_chunks', 0)}")
            print(f"   Total embeddings: {processing_stats.get('total_embeddings', 0)}")
            print(f"   Success rate: {processing_stats.get('embedding_success_rate', 0):.1%}")
            
            timing = stats.get('timing_breakdown', {})
            print(f"\n⏱️  TIMING BREAKDOWN:")
            print(f"   JSON processing: {timing.get('json_processing', 0):.3f}s")
            print(f"   Collection creation: {timing.get('collection_creation', 0):.3f}s")
            print(f"   Embedding generation: {timing.get('embedding_generation', 0):.3f}s")
            print(f"   Qdrant upload: {timing.get('qdrant_upload', 0):.3f}s")
            print(f"   Index creation: {timing.get('index_creation', 0):.3f}s")
            print(f"   Total time: {timing.get('total_time', 0):.3f}s")
            
            performance = stats.get('performance_metrics', {})
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Embeddings per second: {performance.get('embeddings_per_second', 0):.1f}")
            print(f"   Upload rate: {performance.get('upload_rate', 0):.1f} embeddings/s")
            
            # JSON processing details
            json_details = stats.get('json_processing_details', {})
            if json_details and 'category_distribution' in json_details:
                print(f"\n📂 CATEGORY DISTRIBUTION:")
                for category, count in json_details['category_distribution'].items():
                    print(f"   {category}: {count} chunks")
            
            print(f"\n🚀 Ready for retrieval and generation!")
            return True
            
        else:
            print(f"\n❌ JSON EMBEDDING GENERATION FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"JSON embedding pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

