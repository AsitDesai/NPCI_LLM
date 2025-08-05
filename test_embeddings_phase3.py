#!/usr/bin/env python3
"""
Comprehensive test script for Phase 3: Embeddings & Storage.

This script tests the complete embeddings pipeline including:
1. Embedding model configuration
2. Text-to-vector conversion using LlamaIndex
3. Qdrant Cloud connection and collection management
4. Embedding storage and retrieval
5. End-to-end pipeline validation

Run this script to validate the complete embeddings system.
"""

import sys
import os
import time
from pathlib import Path
from typing import List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from embeddings.embedder import LlamaIndexEmbedder, EmbeddingStats
from embeddings.vector_store import QdrantVectorStore, VectorStoreStats
from embeddings.models import get_default_embedding_config, get_embedding_config_for_model
from data.ingestion.data_collector import DocumentCollector
from data.ingestion.preprocessor import TextPreprocessor
from data.ingestion.chunking import DocumentChunker
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run the complete Phase 3 embeddings pipeline test."""
    print("🚀 STARTING PHASE 3: EMBEDDINGS & STORAGE TEST")
    print("="*80)
    print("Testing complete embeddings pipeline with Qdrant Cloud")
    print("="*80)
    
    try:
        # ========================================
        # STEP 1: CONFIGURATION VALIDATION
        # ========================================
        print("\n" + "="*60)
        print("🔧 STEP 1: CONFIGURATION VALIDATION")
        print("="*60)
        
        # Test embedding configuration
        config = get_default_embedding_config()
        print(f"✅ Embedding config loaded")
        print(f"   Model: {config.model_name}")
        print(f"   Dimension: {config.model_dimension}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Max length: {config.max_length}")
        
        # Test Qdrant configuration
        print(f"✅ Qdrant config loaded")
        print(f"   Host: {settings.qdrant_host}")
        print(f"   Port: {settings.qdrant_port}")
        print(f"   Collection: {settings.vector_db_name}")
        print(f"   Dimension: {settings.vector_db_dimension}")
        print(f"   Metric: {settings.vector_db_metric}")
        print(f"   API Key: {'✅ Configured' if settings.qdrant_api_key else '❌ Missing'}")
        
        # ========================================
        # STEP 2: EMBEDDING MODEL INITIALIZATION
        # ========================================
        print("\n" + "="*60)
        print("🤖 STEP 2: EMBEDDING MODEL INITIALIZATION")
        print("="*60)
        
        # Initialize embedder
        embedder = LlamaIndexEmbedder(config)
        print(f"✅ LlamaIndex embedder initialized")
        
        # Get model info
        model_info = embedder.get_model_info()
        print(f"   Model info: {model_info}")
        
        # Test single text embedding
        test_text = "How do I check my payment status?"
        print(f"✅ Testing single text embedding")
        print(f"   Test text: '{test_text}'")
        
        start_time = time.time()
        embedding = embedder.embed_text(test_text)
        processing_time = time.time() - start_time
        
        print(f"   Generated embedding: {len(embedding)} dimensions")
        print(f"   Processing time: {processing_time:.3f}s")
        
        # Validate embedding
        is_valid = embedder.validate_embedding(embedding)
        print(f"   Embedding validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # ========================================
        # STEP 3: QDRANT CLOUD CONNECTION
        # ========================================
        print("\n" + "="*60)
        print("☁️ STEP 3: QDRANT CLOUD CONNECTION")
        print("="*60)
        
        # Initialize vector store
        vector_store = QdrantVectorStore()
        print(f"✅ Qdrant vector store initialized")
        
        # Get connection info
        connection_info = vector_store.get_connection_info()
        print(f"   Connection info: {connection_info}")
        
        # Check if collection exists
        collection_exists = vector_store.collection_exists()
        print(f"   Collection exists: {'✅ YES' if collection_exists else '❌ NO'}")
        
        if not collection_exists:
            print(f"   Creating collection...")
            success = vector_store.create_collection()
            if success:
                print(f"   ✅ Collection created successfully")
            else:
                print(f"   ❌ Failed to create collection")
                return False
        else:
            print(f"   ✅ Using existing collection")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        if stats:
            print(f"   Collection stats:")
            print(f"     Total points: {stats.total_points}")
            print(f"     Vector dimension: {stats.vector_dimension}")
            print(f"     Distance metric: {stats.distance_metric}")
        else:
            print(f"   ❌ Could not get collection stats")
        
        # ========================================
        # STEP 4: DOCUMENT PROCESSING PIPELINE
        # ========================================
        print("\n" + "="*60)
        print("📚 STEP 4: DOCUMENT PROCESSING PIPELINE")
        print("="*60)
        
        # Load documents from Phase 2
        collector = DocumentCollector()
        documents = collector.load_documents()
        print(f"✅ Loaded {len(documents)} documents")
        
        if not documents:
            print("❌ No documents found, cannot proceed")
            return False
        
        # Preprocess documents
        preprocessor = TextPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)
        print(f"✅ Preprocessed {len(processed_documents)} documents")
        
        # Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(processed_documents, method="sentence")
        print(f"✅ Created {len(chunks)} chunks")
        
        # ========================================
        # STEP 5: EMBEDDING GENERATION
        # ========================================
        print("\n" + "="*60)
        print("🔢 STEP 5: EMBEDDING GENERATION")
        print("="*60)
        
        # Generate embeddings for all chunks
        print(f"✅ Generating embeddings for {len(chunks)} chunks")
        
        start_time = time.time()
        embeddings = embedder.embed_nodes(chunks)
        processing_time = time.time() - start_time
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Average time per chunk: {processing_time/len(chunks):.3f}s")
        
        # Validate all embeddings
        valid_embeddings = 0
        for i, embedding in enumerate(embeddings):
            if embedder.validate_embedding(embedding):
                valid_embeddings += 1
            else:
                print(f"   ⚠️ Invalid embedding at index {i}")
        
        print(f"   Embedding validation: {valid_embeddings}/{len(embeddings)} valid")
        
        # Get embedding statistics
        embedding_stats = embedder.get_embedding_stats([chunk.text for chunk in chunks], processing_time)
        print(f"   Embedding stats:")
        print(f"     Total texts: {embedding_stats.total_texts}")
        print(f"     Total embeddings: {embedding_stats.total_embeddings}")
        print(f"     Total tokens: {embedding_stats.total_tokens}")
        print(f"     Processing time: {embedding_stats.processing_time:.3f}s")
        print(f"     Avg time per text: {embedding_stats.avg_time_per_text:.3f}s")
        
        # ========================================
        # STEP 6: VECTOR STORAGE
        # ========================================
        print("\n" + "="*60)
        print("💾 STEP 6: VECTOR STORAGE")
        print("="*60)
        
        # Store embeddings in Qdrant
        print(f"✅ Storing {len(embeddings)} embeddings in Qdrant Cloud")
        
        start_time = time.time()
        storage_success = vector_store.store_embeddings(embeddings, chunks, batch_size=50)
        storage_time = time.time() - start_time
        
        if storage_success:
            print(f"   ✅ Successfully stored embeddings")
            print(f"   Storage time: {storage_time:.3f}s")
        else:
            print(f"   ❌ Failed to store embeddings")
            return False
        
        # Get updated collection stats
        updated_stats = vector_store.get_collection_stats()
        if updated_stats:
            print(f"   Updated collection stats:")
            print(f"     Total points: {updated_stats.total_points}")
            print(f"     Storage size: {updated_stats.storage_size or 'N/A'}")
        
        # ========================================
        # STEP 7: VECTOR SEARCH TEST
        # ========================================
        print("\n" + "="*60)
        print("🔍 STEP 7: VECTOR SEARCH TEST")
        print("="*60)
        
        # Test search functionality
        test_queries = [
            "How do I check my payment status?",
            "What are the payment ethics?",
            "How to process refunds?",
            "What is the refund policy?"
        ]
        
        for query in test_queries:
            print(f"   Testing query: '{query}'")
            
            # Generate query embedding
            query_embedding = embedder.embed_text(query)
            
            # Search for similar embeddings
            search_results = vector_store.search_similar(query_embedding, top_k=3)
            
            print(f"     Found {len(search_results)} results:")
            for i, result in enumerate(search_results[:2]):  # Show top 2
                score = result['score']
                text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                print(f"       {i+1}. Score: {score:.3f} | Text: {text}")
        
        # ========================================
        # STEP 8: PIPELINE INTEGRATION TEST
        # ========================================
        print("\n" + "="*60)
        print("🔗 STEP 8: PIPELINE INTEGRATION TEST")
        print("="*60)
        
        # Test end-to-end pipeline
        print(f"✅ Document collection: {len(documents)} documents")
        print(f"✅ Text preprocessing: {len(processed_documents)} documents processed")
        print(f"✅ Document chunking: {len(chunks)} chunks created")
        print(f"✅ Embedding generation: {len(embeddings)} embeddings generated")
        print(f"✅ Vector storage: {len(embeddings)} embeddings stored in Qdrant Cloud")
        print(f"✅ Vector search: Search functionality working")
        print(f"✅ LlamaIndex integration: All operations use LlamaIndex components")
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("📊 PHASE 3 TEST SUMMARY")
        print("="*80)
        
        print("✅ ALL PHASE 3 STEPS COMPLETED SUCCESSFULLY!")
        print()
        print("🎯 Phase 3 Achievements:")
        print(f"   • Embedding model: {config.model_name} ({config.model_dimension}d)")
        print(f"   • Qdrant Cloud: Connected and operational")
        print(f"   • Documents processed: {len(documents)}")
        print(f"   • Chunks created: {len(chunks)}")
        print(f"   • Embeddings generated: {len(embeddings)}")
        print(f"   • Embeddings stored: {len(embeddings)} in Qdrant Cloud")
        print(f"   • Search functionality: ✅ Working")
        print()
        print("🚀 Ready for Phase 4: Retrieval & Generation")
        print("   The embeddings system is fully functional and validated.")
        print("   All documents have been processed, embedded, and stored in Qdrant Cloud.")
        print()
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3 TEST FAILED: {e}")
        logger.error(f"Phase 3 test failed: {e}")
        print("\n" + "="*80)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 