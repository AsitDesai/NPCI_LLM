#!/usr/bin/env python3
"""
Embedding generation script using LlamaIndex.

This script handles the complete embedding generation pipeline:
- Document processing using LlamaIndex
- Embedding generation with LlamaIndex HuggingFace integration
- Vector storage in Qdrant Cloud
- Index building and persistence
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from scripts.data_ingestion import LlamaIndexDataIngestion
from embeddings.embedder import LlamaIndexEmbedder
from embeddings.vector_store import QdrantVectorStore as CustomQdrantStore
from embeddings.models import get_default_embedding_config
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


class LlamaIndexEmbeddingBuilder:
    """
    LlamaIndex-based embedding generation and storage.
    
    This class provides a complete embedding pipeline using LlamaIndex
    components for embedding generation and vector storage.
    """
    
    def __init__(self):
        """Initialize the LlamaIndex embedding builder."""
        self.data_ingestion = LlamaIndexDataIngestion()
        self.embedder = LlamaIndexEmbedder(get_default_embedding_config())
        self.vector_store = CustomQdrantStore()
        
        logger.info("LlamaIndex embedding builder initialized")
    
    def create_llama_index_embedding_model(self) -> HuggingFaceEmbedding:
        """
        Create LlamaIndex HuggingFace embedding model.
        
        Returns:
            LlamaIndex HuggingFaceEmbedding instance
        """
        try:
            embedder = HuggingFaceEmbedding(
                model_name=settings.embedding_model_name,
                cache_folder=settings.llama_index_cache_dir,
                trust_remote_code=True,
                device="cpu"
            )
            
            logger.info(f"Created LlamaIndex embedding model: {settings.embedding_model_name}")
            return embedder
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex embedding model: {e}")
            raise
    
    def create_llama_index_vector_store(self) -> QdrantVectorStore:
        """
        Create LlamaIndex Qdrant vector store.
        
        Returns:
            LlamaIndex QdrantVectorStore instance
        """
        try:
            from qdrant_client import QdrantClient
            
            # Initialize Qdrant client
            if settings.qdrant_host.startswith(('http://', 'https://')):
                client = QdrantClient(
                    url=settings.qdrant_host,
                    api_key=settings.qdrant_api_key,
                    timeout=30.0
                )
            else:
                client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key
                )
            
            # Create LlamaIndex vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=settings.vector_db_name
            )
            
            logger.info(f"Created LlamaIndex vector store: {settings.vector_db_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex vector store: {e}")
            raise
    
    def build_embeddings_llama_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """
        Build embeddings using LlamaIndex VectorStoreIndex.
        
        Args:
            nodes: List of LlamaIndex TextNode objects
            
        Returns:
            LlamaIndex VectorStoreIndex
        """
        try:
            logger.info(f"Building embeddings for {len(nodes)} nodes using LlamaIndex")
            
            # Create embedding model
            embed_model = self.create_llama_index_embedding_model()
            
            # Create vector store
            vector_store = self.create_llama_index_vector_store()
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            # Build index
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )
            
            logger.info(f"Successfully built LlamaIndex embeddings for {len(nodes)} nodes")
            return index
            
        except Exception as e:
            logger.error(f"Error building LlamaIndex embeddings: {e}")
            raise
    
    def persist_index(self, index: VectorStoreIndex, persist_dir: Optional[str] = None) -> bool:
        """
        Persist the LlamaIndex to disk.
        
        Args:
            index: LlamaIndex VectorStoreIndex
            persist_dir: Directory to persist the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            persist_dir = persist_dir or settings.llama_index_persist_dir
            
            # Ensure persist directory exists
            os.makedirs(persist_dir, exist_ok=True)
            
            # Persist index
            index.storage_context.persist(persist_dir=persist_dir)
            
            logger.info(f"Persisted LlamaIndex to: {persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting LlamaIndex: {e}")
            return False
    
    def load_persisted_index(self, persist_dir: Optional[str] = None) -> Optional[VectorStoreIndex]:
        """
        Load a persisted LlamaIndex from disk.
        
        Args:
            persist_dir: Directory containing the persisted index
            
        Returns:
            LlamaIndex VectorStoreIndex if successful, None otherwise
        """
        try:
            persist_dir = persist_dir or settings.llama_index_persist_dir
            
            if not os.path.exists(persist_dir):
                logger.warning(f"Persist directory does not exist: {persist_dir}")
                return None
            
            # Load storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir
            )
            
            # Load index
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.create_llama_index_vector_store(),
                storage_context=storage_context
            )
            
            logger.info(f"Loaded persisted LlamaIndex from: {persist_dir}")
            return index
            
        except Exception as e:
            logger.error(f"Error loading persisted LlamaIndex: {e}")
            return None
    
    def build_embeddings_pipeline(self, documents_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete embedding generation pipeline using LlamaIndex.
        
        Args:
            documents_dir: Directory containing documents
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LlamaIndex embedding generation pipeline")
            
            # Step 1: Process documents
            doc_start = time.time()
            doc_results = self.data_ingestion.process_documents_pipeline(documents_dir)
            doc_time = time.time() - doc_start
            
            if not doc_results["success"]:
                logger.error("Document processing failed, stopping pipeline")
                return {
                    "success": False,
                    "error": doc_results.get("error", "Document processing failed"),
                    "statistics": {}
                }
            
            nodes = doc_results["nodes"]
            
            # Step 2: Build embeddings
            embed_start = time.time()
            index = self.build_embeddings_llama_index(nodes)
            embed_time = time.time() - embed_start
            
            # Step 3: Persist index
            persist_start = time.time()
            persist_success = self.persist_index(index)
            persist_time = time.time() - persist_start
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            doc_stats = doc_results["statistics"]
            
            statistics = {
                **doc_stats,
                "embedding_time": embed_time,
                "persist_time": persist_time,
                "total_pipeline_time": total_time,
                "embedding_success": True,
                "persist_success": persist_success,
                "index_created": True,
                "nodes_embedded": len(nodes)
            }
            
            logger.info(f"Embedding generation pipeline completed in {total_time:.3f}s")
            logger.info(f"Statistics: {statistics}")
            
            return {
                "success": True,
                "index": index,
                "nodes": nodes,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"Embedding generation pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def validate_embeddings(self, index: VectorStoreIndex) -> Dict[str, Any]:
        """
        Validate the generated embeddings.
        
        Args:
            index: LlamaIndex VectorStoreIndex
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Test query to validate embeddings without using LLM
            test_query = "test query for validation"
            
            # Create retriever instead of query engine to avoid LLM dependency
            retriever = index.as_retriever(similarity_top_k=3)
            
            # Test retrieval
            retrieved_nodes = retriever.retrieve(test_query)
            
            validation_results = {
                "index_valid": True,
                "retriever_created": True,
                "test_query_successful": True,
                "nodes_retrieved": len(retrieved_nodes),
                "has_embeddings": all(hasattr(node, 'embedding') and node.embedding is not None for node in retrieved_nodes) if retrieved_nodes else False
            }
            
            logger.info("Embedding validation successful")
            return validation_results
            
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return {
                "index_valid": False,
                "error": str(e)
            }
    
    def get_embedding_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed embedding statistics.
        
        Args:
            results: Results from build_embeddings_pipeline
            
        Returns:
            Dictionary with detailed statistics
        """
        if not results.get("success", False):
            return {"error": "Embedding generation failed"}
        
        stats = results["statistics"]
        
        # Additional embedding statistics
        nodes = results.get("nodes", [])
        
        # Calculate embedding statistics
        total_embedding_chars = sum(len(node.text) for node in nodes)
        avg_embedding_chars = total_embedding_chars / len(nodes) if nodes else 0
        
        # Performance metrics
        embedding_performance = {
            "nodes_per_second": stats["nodes_embedded"] / stats["embedding_time"] if stats["embedding_time"] > 0 else 0,
            "chars_per_second": total_embedding_chars / stats["embedding_time"] if stats["embedding_time"] > 0 else 0,
            "total_pipeline_efficiency": stats["nodes_embedded"] / stats["total_pipeline_time"] if stats["total_pipeline_time"] > 0 else 0
        }
        
        detailed_stats = {
            **stats,
            "embedding_stats": {
                "total_embedding_chars": total_embedding_chars,
                "avg_embedding_chars": avg_embedding_chars,
                "embedding_model": settings.embedding_model_name,
                "vector_dimension": settings.embedding_model_dimension
            },
            "performance_metrics": embedding_performance,
            "storage_info": {
                "vector_db_name": settings.vector_db_name,
                "vector_db_dimension": settings.vector_db_dimension,
                "vector_db_metric": settings.vector_db_metric,
                "persist_directory": settings.llama_index_persist_dir
            }
        }
        
        return detailed_stats


def main():
    """Run the LlamaIndex embedding generation pipeline."""
    print("🚀 LLAMAINDEX EMBEDDING GENERATION PIPELINE")
    print("="*60)
    
    try:
        # Initialize pipeline
        builder = LlamaIndexEmbeddingBuilder()
        
        # Build embeddings
        results = builder.build_embeddings_pipeline()
        
        if results["success"]:
            print("\n✅ EMBEDDING GENERATION COMPLETED SUCCESSFULLY!")
            
            # Validate embeddings
            print("\n🔍 Validating embeddings...")
            validation = builder.validate_embeddings(results["index"])
            
            if validation["index_valid"]:
                print("✅ Embedding validation successful")
            else:
                print(f"⚠️ Embedding validation failed: {validation.get('error', 'Unknown error')}")
            
            # Display statistics
            stats = builder.get_embedding_stats(results)
            
            print(f"\n📊 EMBEDDING STATISTICS:")
            print(f"   Documents processed: {stats['documents_loaded']}")
            print(f"   Nodes created: {stats['chunks_created']}")
            print(f"   Nodes embedded: {stats['nodes_embedded']}")
            print(f"   Total pipeline time: {stats['total_pipeline_time']:.3f}s")
            print(f"   Embedding time: {stats['embedding_time']:.3f}s")
            print(f"   Persist time: {stats['persist_time']:.3f}s")
            
            # Embedding model info
            embed_stats = stats.get('embedding_stats', {})
            print(f"\n🤖 EMBEDDING MODEL INFO:")
            print(f"   Model: {embed_stats.get('embedding_model', 'Unknown')}")
            print(f"   Dimension: {embed_stats.get('vector_dimension', 'Unknown')}")
            print(f"   Total characters: {embed_stats.get('total_embedding_chars', 0):,}")
            print(f"   Average chars per node: {embed_stats.get('avg_embedding_chars', 0):.0f}")
            
            # Performance metrics
            perf = stats.get('performance_metrics', {})
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Nodes per second: {perf.get('nodes_per_second', 0):.1f}")
            print(f"   Characters per second: {perf.get('chars_per_second', 0):.0f}")
            print(f"   Total pipeline efficiency: {perf.get('total_pipeline_efficiency', 0):.1f} nodes/s")
            
            # Storage info
            storage = stats.get('storage_info', {})
            print(f"\n💾 STORAGE INFO:")
            print(f"   Vector DB: {storage.get('vector_db_name', 'Unknown')}")
            print(f"   Dimension: {storage.get('vector_db_dimension', 'Unknown')}")
            print(f"   Metric: {storage.get('vector_db_metric', 'Unknown')}")
            print(f"   Persist directory: {storage.get('persist_directory', 'Unknown')}")
            
            print(f"\n🚀 Ready for retrieval and generation!")
            return True
            
        else:
            print(f"\n❌ EMBEDDING GENERATION FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Embedding generation pipeline failed: {e}")
        print(f"\n❌ PIPELINE FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 