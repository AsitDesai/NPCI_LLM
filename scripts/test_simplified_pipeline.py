#!/usr/bin/env python3
"""
Simplified Pipeline Test

This script tests the entire simplified pipeline:
1. Simple data ingestion with increased overlap
2. Simple embedding generation
3. Simple retrieval
4. End-to-end functionality
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simple_data_ingestion import SimpleDataIngestion
from scripts.simple_embedding_builder import SimpleEmbeddingBuilder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from simple_retriever import SimpleSemanticRetriever
from config.logging import get_logger

logger = get_logger(__name__)


def test_simplified_pipeline():
    """Test the entire simplified pipeline."""
    print("🧪 SIMPLIFIED PIPELINE TEST")
    print("="*60)
    
    try:
        # Step 1: Test Data Ingestion
        print("\n📄 STEP 1: DATA INGESTION TEST")
        print("-" * 40)
        
        ingestion = SimpleDataIngestion(chunk_size=500, chunk_overlap=200)
        ingestion_results = ingestion.process_all_files()
        
        if not ingestion_results["success"]:
            print(f"❌ Data ingestion failed: {ingestion_results.get('error')}")
            return False
        
        chunks = ingestion_results["chunks"]
        stats = ingestion_results["statistics"]
        
        print(f"✅ Data ingestion successful")
        print(f"   Files processed: {stats.files_processed}")
        print(f"   Total chunks: {stats.total_chunks}")
        print(f"   Valid chunks: {stats.valid_chunks}")
        print(f"   Processing time: {stats.processing_time:.3f}s")
        print(f"   Average tokens per chunk: {stats.avg_tokens_per_chunk:.1f}")
        
        # Show sample chunks
        if chunks:
            print(f"\n📋 SAMPLE CHUNKS:")
            for i, chunk in enumerate(chunks[:2]):
                print(f"   Chunk {i+1}:")
                print(f"     ID: {chunk.chunk_id}")
                print(f"     Source: {chunk.source_file}")
                print(f"     Tokens: {chunk.token_count}")
                print(f"     Text preview: {chunk.text[:100]}...")
                print()
        
        # Step 2: Test Embedding Generation
        print("\n🔢 STEP 2: EMBEDDING GENERATION TEST")
        print("-" * 40)
        
        builder = SimpleEmbeddingBuilder(batch_size=16)
        embedding_results = builder.process_all_files()
        
        if not embedding_results["success"]:
            print(f"❌ Embedding generation failed: {embedding_results.get('error')}")
            return False
        
        embedding_stats = embedding_results["statistics"]
        
        print(f"✅ Embedding generation successful")
        print(f"   Total chunks: {embedding_stats.total_chunks}")
        print(f"   Embeddings generated: {embedding_stats.total_embeddings}")
        print(f"   Processing time: {embedding_stats.processing_time:.3f}s")
        print(f"   Embedding time: {embedding_stats.embedding_time:.3f}s")
        print(f"   Upload time: {embedding_stats.upload_time:.3f}s")
        print(f"   Model: {embedding_stats.model_name}")
        print(f"   Vector dimension: {embedding_stats.vector_dimension}")
        
        # Step 3: Test Retrieval
        print("\n🔍 STEP 3: RETRIEVAL TEST")
        print("-" * 40)
        
        retriever = SimpleSemanticRetriever()
        
        # Get collection info
        collection_info = retriever.get_collection_info()
        if "error" not in collection_info:
            print(f"✅ Collection info retrieved")
            print(f"   Collection: {collection_info['collection_name']}")
            print(f"   Vector size: {collection_info['vector_size']}")
            print(f"   Points count: {collection_info['points_count']}")
            print(f"   Status: {collection_info['status']}")
        else:
            print(f"❌ Error getting collection info: {collection_info['error']}")
            return False
        
        # Test retrieval with various queries
        test_queries = [
            "insufficient balance",
            "UPI PIN reset",
            "transaction failed",
            "how to reset UPI PIN",
            "what to do when transaction fails"
        ]
        
        print(f"\n🔍 TESTING RETRIEVAL WITH VARIOUS QUERIES:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                for j, result in enumerate(results[:2]):  # Show first 2
                    print(f"     Result {j+1}: score={result.score:.3f}, source={result.source_file}")
                    print(f"     Text preview: {result.text[:80]}...")
            else:
                print(f"   ⚠️ No results found")
        
        # Step 4: Test End-to-End Performance
        print("\n⚡ STEP 4: END-TO-END PERFORMANCE TEST")
        print("-" * 40)
        
        # Test a complex query with timing
        complex_query = "My UPI transaction failed due to insufficient balance, what should I do?"
        
        print(f"Testing complex query: '{complex_query}'")
        
        start_time = time.time()
        results = retriever.retrieve(complex_query, top_k=5)
        retrieval_time = time.time() - start_time
        
        if results:
            print(f"✅ Complex query successful")
            print(f"   Retrieval time: {retrieval_time:.3f}s")
            print(f"   Found {len(results)} results")
            
            # Show best result
            best_result = results[0]
            print(f"   Best result (score: {best_result.score:.3f}):")
            print(f"   Source: {best_result.source_file}")
            print(f"   Text: {best_result.text[:200]}...")
        else:
            print(f"❌ Complex query failed - no results")
            return False
        
        # Final validation
        success_criteria = [
            len(chunks) > 0,
            embedding_stats.total_embeddings > 0,
            collection_info['points_count'] > 0,
            len(results) > 0
        ]
        
        all_passed = all(success_criteria)
        
        print(f"\n🎯 PIPELINE VALIDATION:")
        print(f"   ✅ Chunks created: {len(chunks) > 0}")
        print(f"   ✅ Embeddings generated: {embedding_stats.total_embeddings > 0}")
        print(f"   ✅ Data in collection: {collection_info['points_count'] > 0}")
        print(f"   ✅ Retrieval working: {len(results) > 0}")
        
        if all_passed:
            print(f"\n🎉 SIMPLIFIED PIPELINE TEST PASSED!")
            print(f"   Ready for LangGraph integration")
            return True
        else:
            print(f"\n❌ SIMPLIFIED PIPELINE TEST FAILED!")
            print(f"   Some validation criteria not met")
            return False
            
    except Exception as e:
        logger.error(f"Simplified pipeline test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


def main():
    """Run the simplified pipeline test."""
    success = test_simplified_pipeline()
    
    if success:
        print(f"\n✅ SIMPLIFIED PIPELINE COMPLETED SUCCESSFULLY")
        print(f"   Next step: LangGraph integration")
        return True
    else:
        print(f"\n❌ SIMPLIFIED PIPELINE FAILED")
        print(f"   Please fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
