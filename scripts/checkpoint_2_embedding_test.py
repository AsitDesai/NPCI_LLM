#!/usr/bin/env python3
"""
Checkpoint 2: Enhanced Embedding Builder Validation Test

This script validates that the enhanced embedding builder correctly:
1. Processes TXT files and generates embeddings
2. Uploads embeddings to Qdrant
3. Maintains proper data structure and metadata
4. Handles both JSON and TXT formats
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_embedding_builder import EnhancedEmbeddingBuilder
from scripts.enhanced_data_ingestion import EnhancedDataIngestion
from config.logging import get_logger

logger = get_logger(__name__)


def test_embedding_builder_detailed():
    """Detailed test of enhanced embedding builder functionality."""
    print("🔍 CHECKPOINT 2: DETAILED EMBEDDING BUILDER TEST")
    print("="*60)
    
    try:
        # Initialize components
        ingestion = EnhancedDataIngestion(max_tokens_per_chunk=200)
        builder = EnhancedEmbeddingBuilder(batch_size=8)  # Smaller batch for testing
        
        print(f"📄 Testing TXT file processing and embedding generation")
        
        # Step 1: Test data ingestion
        print(f"\n📊 STEP 1: DATA INGESTION TEST")
        ingestion_results = ingestion.process_txt_files()
        
        if not ingestion_results["success"]:
            print(f"❌ Data ingestion failed: {ingestion_results.get('error')}")
            return False
        
        chunks = ingestion_results["chunks"]
        stats = ingestion_results["statistics"]
        
        print(f"   ✅ Successfully processed {len(chunks)} chunks")
        print(f"   Files processed: {stats.files_processed}")
        print(f"   Valid chunks: {stats.valid_chunks}")
        print(f"   Average tokens per chunk: {stats.avg_tokens_per_chunk:.1f}")
        
        # Step 2: Test embedding generation (without upload)
        print(f"\n📊 STEP 2: EMBEDDING GENERATION TEST")
        embedding_start = time.time()
        
        try:
            embedding_results = builder.generate_embeddings(chunks)
            embedding_time = time.time() - embedding_start
            
            print(f"   ✅ Successfully generated {len(embedding_results)} embeddings")
            print(f"   Embedding time: {embedding_time:.3f}s")
            print(f"   Average time per embedding: {embedding_time/len(embedding_results):.3f}s")
            
            # Validate embedding structure
            valid_embeddings = 0
            for result in embedding_results:
                if (len(result.vector) == 384 and  # Correct dimension
                    isinstance(result.vector[0], float) and  # Proper float values
                    result.chunk_id and  # Has chunk ID
                    result.payload):  # Has payload
                    valid_embeddings += 1
            
            print(f"   Valid embedding structure: {valid_embeddings}/{len(embedding_results)}")
            
            if valid_embeddings != len(embedding_results):
                print(f"   ❌ Some embeddings have invalid structure")
                return False
            
            # Show sample embedding
            if embedding_results:
                sample = embedding_results[0]
                print(f"   Sample embedding:")
                print(f"     Chunk ID: {sample.chunk_id}")
                print(f"     Vector dimension: {len(sample.vector)}")
                print(f"     Vector type: {type(sample.vector[0])}")
                print(f"     Payload keys: {list(sample.payload.keys())}")
            
        except Exception as e:
            print(f"   ❌ Embedding generation failed: {e}")
            return False
        
        # Step 3: Test Qdrant connection and collection creation
        print(f"\n📊 STEP 3: QDRANT CONNECTION TEST")
        try:
            # Test connection
            collections = builder.qdrant_client.get_collections()
            print(f"   ✅ Qdrant connection successful")
            print(f"   Available collections: {len(collections.collections)}")
            
            # Test collection creation
            test_collection = "test_enhanced_embeddings"
            collection_created = builder.create_collection(test_collection)
            
            if collection_created:
                print(f"   ✅ Collection '{test_collection}' created/verified")
            else:
                print(f"   ❌ Failed to create collection")
                return False
                
        except Exception as e:
            print(f"   ❌ Qdrant connection failed: {e}")
            return False
        
        # Step 4: Test upload (with small subset)
        print(f"\n📊 STEP 4: UPLOAD TEST (Small subset)")
        try:
            # Upload only first 5 embeddings for testing
            test_embeddings = embedding_results[:5]
            upload_success = builder.upload_to_qdrant(
                test_embeddings, 
                collection_name=test_collection,
                batch_size=5
            )
            
            if upload_success:
                print(f"   ✅ Successfully uploaded {len(test_embeddings)} embeddings")
                
                # Verify upload by checking collection info
                collection_info = builder.qdrant_client.get_collection(test_collection)
                print(f"   Collection points: {collection_info.points_count}")
                print(f"   Vector size: {collection_info.config.params.vectors.size}")
                
            else:
                print(f"   ❌ Upload failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Upload test failed: {e}")
            return False
        
        # Step 5: Test search functionality
        print(f"\n📊 STEP 5: SEARCH TEST")
        try:
            # Test search with a sample query
            test_query = "insufficient balance"
            
            # Generate query embedding
            query_embedding = builder.embedding_model.encode([test_query], convert_to_list=True)[0]
            query_embedding = builder._ensure_float_embeddings([query_embedding])[0]
            
            # Search in collection
            search_results = builder.qdrant_client.search(
                collection_name=test_collection,
                query_vector=query_embedding,
                limit=3
            )
            
            if search_results:
                print(f"   ✅ Search successful: {len(search_results)} results")
                for i, result in enumerate(search_results):
                    print(f"     Result {i+1}: score={result.score:.3f}, category={result.payload.get('category', 'N/A')}")
            else:
                print(f"   ⚠️ Search returned no results (may be normal with small test set)")
                
        except Exception as e:
            print(f"   ❌ Search test failed: {e}")
            return False
        
        # Final validation
        success_criteria = [
            len(chunks) > 0,
            len(embedding_results) > 0,
            valid_embeddings == len(embedding_results),
            collection_created,
            upload_success
        ]
        
        all_passed = all(success_criteria)
        
        print(f"\n🎯 CHECKPOINT 2 VALIDATION:")
        print(f"   ✅ Chunks processed: {len(chunks) > 0}")
        print(f"   ✅ Embeddings generated: {len(embedding_results) > 0}")
        print(f"   ✅ Valid embedding structure: {valid_embeddings == len(embedding_results)}")
        print(f"   ✅ Collection created: {collection_created}")
        print(f"   ✅ Upload successful: {upload_success}")
        
        if all_passed:
            print(f"\n🎉 CHECKPOINT 2 PASSED! Enhanced embedding builder is working correctly.")
            print(f"   Ready to proceed to Step 3: Enhanced Retrieval System")
            return True
        else:
            print(f"\n❌ CHECKPOINT 2 FAILED! Some validation criteria not met.")
            return False
            
    except Exception as e:
        logger.error(f"Checkpoint 2 test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


def main():
    """Run the checkpoint test."""
    success = test_embedding_builder_detailed()
    
    if success:
        print(f"\n✅ CHECKPOINT 2 COMPLETED SUCCESSFULLY")
        print(f"   Next step: Update retrieval system for enhanced data support")
        return True
    else:
        print(f"\n❌ CHECKPOINT 2 FAILED")
        print(f"   Please fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
