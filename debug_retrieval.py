"""
Debug script to test retrieval and see what context is being sent to the model
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
from retrieval.retriever import SemanticRetriever
from retrieval.context_builder import ContextBuilder

def debug_retrieval():
    """Debug the retrieval process step by step."""
    print("🔍 DEBUGGING RAG RETRIEVAL")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = LlamaIndexEmbedder(get_default_embedding_config())
    vector_store = QdrantVectorStore()
    retriever = SemanticRetriever(embedder, vector_store)
    context_builder = ContextBuilder()
    
    print("✅ Components initialized")
    
    # Test query
    test_query = "how can i check my payment status"
    print(f"\n2. Testing query: '{test_query}'")
    
    # Step 1: Retrieve documents
    print("\n3. Retrieving documents...")
    try:
        retrieved_docs = retriever.retrieve(test_query)
        print(f"✅ Retrieved {len(retrieved_docs)} documents")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"\n📄 Document {i+1}:")
            print(f"   Text: {doc.text[:200]}...")
            print(f"   Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return
    
    # Step 2: Build context
    print("\n4. Building context...")
    try:
        context = context_builder.build_context(retrieved_docs)
        print(f"✅ Context built ({len(context)} characters)")
        print(f"\n📝 CONTEXT SENT TO MODEL:")
        print("=" * 60)
        print(context)
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error building context: {e}")
        return
    
    # Step 3: Check what's in the vector store
    print("\n5. Checking vector store contents...")
    try:
        # Get collection info
        collection_info = vector_store.client.get_collection("rag_embeddings")
        print(f"✅ Collection points: {collection_info.points_count}")
        
        # Try to get all points
        all_points = vector_store.client.scroll(
            collection_name="rag_embeddings",
            limit=10
        )
        print(f"✅ Found {len(all_points[0])} points in collection")
        
        for i, point in enumerate(all_points[0]):
            print(f"\n🔍 Point {i+1}:")
            print(f"   ID: {point.id}")
            print(f"   Payload: {point.payload}")
            if hasattr(point, 'vector'):
                print(f"   Vector dimension: {len(point.vector)}")
            
    except Exception as e:
        print(f"❌ Error checking vector store: {e}")
        return
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_retrieval() 