#!/usr/bin/env python3
"""
Test script for LlamaIndex Integration.

This script tests the complete LlamaIndex integration including:
1. Environment setup and validation
2. Document processing pipeline
3. Embedding generation and storage
4. End-to-end LlamaIndex workflow validation

Run this script to validate the complete LlamaIndex integration.
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.setup import setup_environment
from scripts.data_ingestion import LlamaIndexDataIngestion
from scripts.build_embeddings import LlamaIndexEmbeddingBuilder
from retrieval.retriever import SemanticRetriever
from retrieval.context_builder import ContextBuilder
from embeddings.embedder import LlamaIndexEmbedder
from embeddings.vector_store import QdrantVectorStore
from embeddings.models import get_default_embedding_config
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


def test_environment_setup():
    """Test environment setup and validation."""
    print("\n" + "="*60)
    print("🔧 STEP 1: ENVIRONMENT SETUP TEST")
    print("="*60)
    
    try:
        success = setup_environment()
        if success:
            print("✅ Environment setup completed successfully")
            return True
        else:
            print("❌ Environment setup failed")
            return False
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False


def test_data_ingestion():
    """Test LlamaIndex data ingestion pipeline."""
    print("\n" + "="*60)
    print("📄 STEP 2: DATA INGESTION TEST")
    print("="*60)
    
    try:
        # Initialize pipeline
        ingestion = LlamaIndexDataIngestion()
        
        # Process documents
        results = ingestion.process_documents_pipeline()
        
        if results["success"]:
            stats = ingestion.get_processing_stats(results)
            
            print("✅ Data ingestion completed successfully")
            print(f"   Documents loaded: {stats['documents_loaded']}")
            print(f"   Chunks created: {stats['chunks_created']}")
            print(f"   Processing time: {stats['total_time']:.3f}s")
            
            return results
        else:
            print(f"❌ Data ingestion failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return None


def test_embedding_generation():
    """Test LlamaIndex embedding generation."""
    print("\n" + "="*60)
    print("🤖 STEP 3: EMBEDDING GENERATION TEST")
    print("="*60)
    
    try:
        # Initialize builder
        builder = LlamaIndexEmbeddingBuilder()
        
        # Build embeddings
        results = builder.build_embeddings_pipeline()
        
        if results["success"]:
            stats = builder.get_embedding_stats(results)
            
            print("✅ Embedding generation completed successfully")
            print(f"   Nodes embedded: {stats['nodes_embedded']}")
            print(f"   Embedding time: {stats['embedding_time']:.3f}s")
            print(f"   Total pipeline time: {stats['total_pipeline_time']:.3f}s")
            
            # Validate embeddings
            validation = builder.validate_embeddings(results["index"])
            if validation["index_valid"]:
                print("✅ Embedding validation successful")
            else:
                print(f"⚠️ Embedding validation failed: {validation.get('error', 'Unknown error')}")
            
            return results
        else:
            print(f"❌ Embedding generation failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return None


def test_retrieval_integration():
    """Test retrieval integration with LlamaIndex."""
    print("\n" + "="*60)
    print("🔍 STEP 4: RETRIEVAL INTEGRATION TEST")
    print("="*60)
    
    try:
        # Initialize components
        embedder = LlamaIndexEmbedder(get_default_embedding_config())
        vector_store = QdrantVectorStore()
        retriever = SemanticRetriever(embedder, vector_store)
        context_builder = ContextBuilder()
        
        # Test queries
        test_queries = [
            "How do I check my payment status?",
            "What are the payment ethics?",
            "How to process refunds?"
        ]
        
        all_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"   Testing query {i}: '{query}'")
            
            # Perform retrieval
            start_time = time.time()
            results = retriever.retrieve(query, top_k=3)
            retrieval_time = time.time() - start_time
            
            if results:
                print(f"     ✅ Retrieved {len(results)} results in {retrieval_time:.3f}s")
                print(f"     Top score: {results[0].score:.3f}")
                
                # Build context
                context_info = context_builder.build_context(results, query)
                print(f"     Context built: {context_info.total_tokens} tokens")
                
                all_results.append(results)
            else:
                print(f"     ❌ No results retrieved")
                all_results.append([])
        
        print(f"✅ Retrieval integration test completed")
        print(f"   Successful queries: {sum(1 for r in all_results if r)}/{len(test_queries)}")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Retrieval integration test failed: {e}")
        return None


def test_end_to_end_workflow():
    """Test complete end-to-end LlamaIndex workflow."""
    print("\n" + "="*60)
    print("🔄 STEP 5: END-TO-END WORKFLOW TEST")
    print("="*60)
    
    try:
        # Test complete workflow
        test_query = "How do I check my payment status?"
        print(f"   Testing complete workflow with query: '{test_query}'")
        
        # Initialize components
        embedder = LlamaIndexEmbedder(get_default_embedding_config())
        vector_store = QdrantVectorStore()
        retriever = SemanticRetriever(embedder, vector_store)
        context_builder = ContextBuilder()
        
        # Step 1: Retrieve
        start_time = time.time()
        results = retriever.retrieve(test_query, top_k=5)
        retrieval_time = time.time() - start_time
        
        if not results:
            print("     ❌ No results retrieved")
            return False
        
        # Step 2: Build context
        context_info = context_builder.build_context(results, test_query)
        build_time = time.time() - start_time - retrieval_time
        
        total_time = time.time() - start_time
        
        print(f"     ✅ Complete workflow completed in {total_time:.3f}s")
        print(f"     Retrieval: {retrieval_time:.3f}s")
        print(f"     Context building: {build_time:.3f}s")
        print(f"     Final context: {context_info.total_tokens} tokens, {context_info.num_chunks} chunks")
        
        # Show sample results
        if results:
            top_result = results[0]
            print(f"     Top result: {top_result.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        return False


def main():
    """Run the complete LlamaIndex integration test."""
    print("🚀 LLAMAINDEX INTEGRATION TEST")
    print("="*80)
    print("Testing complete LlamaIndex integration workflow")
    print("="*80)
    
    test_results = {
        "environment_setup": False,
        "data_ingestion": False,
        "embedding_generation": False,
        "retrieval_integration": False,
        "end_to_end_workflow": False
    }
    
    try:
        # Step 1: Environment Setup
        test_results["environment_setup"] = test_environment_setup()
        
        # Step 2: Data Ingestion
        if test_results["environment_setup"]:
            ingestion_results = test_data_ingestion()
            test_results["data_ingestion"] = ingestion_results is not None
        else:
            print("⏭️ Skipping data ingestion (environment setup failed)")
        
        # Step 3: Embedding Generation
        if test_results["data_ingestion"]:
            embedding_results = test_embedding_generation()
            test_results["embedding_generation"] = embedding_results is not None
        else:
            print("⏭️ Skipping embedding generation (data ingestion failed)")
        
        # Step 4: Retrieval Integration
        if test_results["embedding_generation"]:
            retrieval_results = test_retrieval_integration()
            test_results["retrieval_integration"] = retrieval_results is not None
        else:
            print("⏭️ Skipping retrieval integration (embedding generation failed)")
        
        # Step 5: End-to-End Workflow
        if test_results["retrieval_integration"]:
            test_results["end_to_end_workflow"] = test_end_to_end_workflow()
        else:
            print("⏭️ Skipping end-to-end workflow (retrieval integration failed)")
        
        # Final Summary
        print("\n" + "="*80)
        print("📊 LLAMAINDEX INTEGRATION TEST SUMMARY")
        print("="*80)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"✅ PASSED: {passed_tests}/{total_tests} tests")
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL LLAMAINDEX INTEGRATION TESTS PASSED!")
            print("🚀 LlamaIndex integration is fully functional")
            print("📈 Ready for production use")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} TESTS FAILED")
            print("🔧 Please check the logs above for issues")
        
        print("\n" + "="*80)
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"LlamaIndex integration test failed: {e}")
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        print("\n" + "="*80)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 