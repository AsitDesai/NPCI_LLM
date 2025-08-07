"""
Debug script to test text extraction from LlamaIndex Document objects
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

def debug_text_extraction():
    """Debug text extraction from LlamaIndex Document objects."""
    print("🔍 DEBUGGING TEXT EXTRACTION")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = LlamaIndexEmbedder(get_default_embedding_config())
    vector_store = QdrantVectorStore()
    retriever = SemanticRetriever(embedder, vector_store)
    
    print("✅ Components initialized")
    
    # Test query
    test_query = "what is the refund policy"
    print(f"\n2. Testing query: '{test_query}'")
    
    # Step 1: Get raw LlamaIndex results
    print("\n3. Getting raw LlamaIndex results...")
    try:
        # Get the raw retriever (LlamaIndex)
        from retrieval.retriever import SemanticRetriever
        raw_retriever = SemanticRetriever(embedder, vector_store)
        
        # This will return LlamaIndex Document objects
        raw_results = raw_retriever.retrieve(test_query, top_k=3)
        print(f"✅ Retrieved {len(raw_results)} raw results")
        
        for i, doc in enumerate(raw_results):
            print(f"\n📄 Raw Document {i+1}:")
            print(f"   Type: {type(doc)}")
            print(f"   Has text attribute: {hasattr(doc, 'text')}")
            print(f"   Has metadata attribute: {hasattr(doc, 'metadata')}")
            
            if hasattr(doc, 'text'):
                text = doc.text
                print(f"   Text length: {len(text)}")
                print(f"   Text preview: {text[:200]}...")
                print(f"   Text is empty: {not text.strip()}")
            else:
                print(f"   No text attribute, doc content: {str(doc)[:200]}...")
            
            if hasattr(doc, 'metadata'):
                print(f"   Metadata keys: {list(doc.metadata.keys())}")
            else:
                print(f"   No metadata attribute")
                
    except Exception as e:
        print(f"❌ Error getting raw results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Test text extraction logic
    print("\n4. Testing text extraction logic...")
    try:
        for i, doc in enumerate(raw_results):
            print(f"\n🔍 Testing extraction for Document {i+1}:")
            
            # Test different extraction methods
            methods = [
                ("doc.text", lambda d: d.text if hasattr(d, 'text') else None),
                ("str(doc)", lambda d: str(d)),
                ("getattr(doc, 'text', '')", lambda d: getattr(d, 'text', '')),
                ("doc.text if hasattr(doc, 'text') else str(doc)", lambda d: d.text if hasattr(d, 'text') else str(d))
            ]
            
            for method_name, method_func in methods:
                try:
                    extracted_text = method_func(doc)
                    print(f"   {method_name}: {len(extracted_text) if extracted_text else 0} chars")
                    if extracted_text:
                        print(f"      Preview: {extracted_text[:100]}...")
                    else:
                        print(f"      Result: EMPTY")
                except Exception as e:
                    print(f"   {method_name}: ERROR - {e}")
                    
    except Exception as e:
        print(f"❌ Error testing extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test the conversion to RetrievalResult
    print("\n5. Testing conversion to RetrievalResult...")
    try:
        from retrieval.retriever import RetrievalResult
        
        for i, doc in enumerate(raw_results):
            print(f"\n🔍 Converting Document {i+1} to RetrievalResult:")
            
            # Try different text extraction methods
            text_methods = [
                ("doc.text", lambda d: d.text if hasattr(d, 'text') else ''),
                ("str(doc)", lambda d: str(d)),
                ("getattr", lambda d: getattr(d, 'text', ''))
            ]
            
            for method_name, text_func in text_methods:
                try:
                    extracted_text = text_func(doc)
                    
                    # Create RetrievalResult
                    retrieval_result = RetrievalResult(
                        text=extracted_text,
                        score=0.8,
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                        source_document=doc.metadata.get('file_name', 'unknown') if hasattr(doc, 'metadata') else 'unknown',
                        chunk_index=doc.metadata.get('chunk_index', i) if hasattr(doc, 'metadata') else i
                    )
                    
                    print(f"   {method_name}: {len(retrieval_result.text)} chars")
                    if retrieval_result.text.strip():
                        print(f"      Preview: {retrieval_result.text[:100]}...")
                    else:
                        print(f"      Result: EMPTY")
                        
                except Exception as e:
                    print(f"   {method_name}: ERROR - {e}")
                    
    except Exception as e:
        print(f"❌ Error testing conversion: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    debug_text_extraction() 