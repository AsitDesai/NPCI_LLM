"""
Debug script to test context building step specifically
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from embeddings.embedder import LlamaIndexEmbedder
from embeddings.models import get_default_embedding_config
from embeddings.vector_store import QdrantVectorStore
from retrieval.retriever import SemanticRetriever, RetrievalResult
from retrieval.context_builder import ContextBuilder

def debug_context_building():
    """Debug the context building step specifically."""
    print("🔍 DEBUGGING CONTEXT BUILDING")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = LlamaIndexEmbedder(get_default_embedding_config())
    vector_store = QdrantVectorStore()
    retriever = SemanticRetriever(embedder, vector_store)
    context_builder = ContextBuilder()
    
    print("✅ Components initialized")
    
    # Test query that was failing
    test_query = "what is the refund policy"
    print(f"\n2. Testing query: '{test_query}'")
    
    # Step 1: Retrieve documents
    print("\n3. Retrieving documents...")
    try:
        results = retriever.retrieve(test_query, top_k=3)
        print(f"✅ Retrieved {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"\n📄 Result {i+1}:")
            print(f"   Score: {result.score:.3f}")
            print(f"   Text: {result.text[:200]}...")
            print(f"   Source: {result.source_document}")
            print(f"   Metadata: {result.metadata}")
            
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return
    
    # Step 2: Build context
    print("\n4. Building context...")
    try:
        context_info = context_builder.build_context(results, test_query)
        print(f"✅ Context built successfully")
        print(f"   Context length: {len(context_info.context)} characters")
        print(f"   Total tokens: {context_info.total_tokens}")
        print(f"   Num chunks: {context_info.num_chunks}")
        print(f"   Sources: {context_info.sources}")
        
        print(f"\n📝 CONTEXT SENT TO MODEL:")
        print("=" * 60)
        print(context_info.context)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error building context: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test with different queries
    print("\n5. Testing with different queries...")
    test_queries = [
        "how can i check my payment status",
        "which browsers can i use for this service",
        "what is the refund policy",
        "how do i reset my password"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        try:
            results = retriever.retrieve(query, top_k=3)
            context_info = context_builder.build_context(results, query)
            
            if context_info.context.strip():
                print(f"✅ Context built: {len(context_info.context)} chars")
            else:
                print(f"❌ Empty context")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_context_building() 