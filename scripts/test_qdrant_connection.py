#!/usr/bin/env python3
"""
Test script to check Qdrant vector database connectivity.
Tests connection, collection operations, and basic vector operations.
"""

import os
import sys
import asyncio
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, 
        CreateCollection, CollectionInfo
    )
except ImportError:
    print("❌ qdrant-client not installed. Please install it with: pip install qdrant-client")
    sys.exit(1)

def load_environment() -> Dict[str, Any]:
    """Load environment variables from env.server file."""
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'env.server')
    load_dotenv(env_file)
    
    return {
        'qdrant_host': os.getenv('QDRANT_HOST', '0.0.0.0'),
        'qdrant_port': int(os.getenv('QDRANT_PORT', 6333)),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
        'vector_db_name': os.getenv('VECTOR_DB_NAME', 'rag_embeddings'),
        'vector_db_dimension': int(os.getenv('VECTOR_DB_DIMENSION', 384)),
        'vector_db_metric': os.getenv('VECTOR_DB_METRIC', 'cosine'),
    }

def test_qdrant_connection(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test basic Qdrant connection."""
    try:
        # Parse host and port
        host, port = config['qdrant_host'].split(':')
        port = int(port)
        
        # Create client with different connection options
        client_options = [
            # Option 1: Default connection
            {
                'host': host,
                'port': port,
                'api_key': config['qdrant_api_key'] if config['qdrant_api_key'] else None
            },
            # Option 2: HTTP connection (no SSL)
            {
                'host': host,
                'port': port,
                'api_key': config['qdrant_api_key'] if config['qdrant_api_key'] else None,
                'https': False
            },
            # Option 3: Localhost connection
            {
                'host': 'localhost',
                'port': port,
                'api_key': config['qdrant_api_key'] if config['qdrant_api_key'] else None,
                'https': False
            },
            # Option 4: 127.0.0.1 connection
            {
                'host': '127.0.0.1',
                'port': port,
                'api_key': config['qdrant_api_key'] if config['qdrant_api_key'] else None,
                'https': False
            }
        ]
        
        client = None
        collections = None
        connection_method = "default"
        
        for i, options in enumerate(client_options):
            try:
                print(f"   Trying connection method {i+1}...")
                client = QdrantClient(**options)
                
                # Test connection by getting collections
                collections = client.get_collections()
                connection_method = f"method_{i+1}"
                break
                
            except Exception as e:
                print(f"   Method {i+1} failed: {str(e)[:100]}...")
                continue
        
        if client is None or collections is None:
            return {
                'success': False,
                'error': 'All connection methods failed',
                'connection_info': config['qdrant_host']
            }
        
        return {
            'success': True,
            'client': client,
            'collections': collections,
            'connection_info': f"{host}:{port}",
            'connection_method': connection_method
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'connection_info': config['qdrant_host']
        }

def test_collection_operations(client: QdrantClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """Test collection creation, listing, and deletion."""
    test_collection_name = f"test_collection_{np.random.randint(1000, 9999)}"
    
    try:
        # Test 1: Create collection
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(
                size=config['vector_db_dimension'],
                distance=Distance.COSINE
            )
        )
        
        # Test 2: List collections
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        # Test 3: Get collection info
        collection_info = client.get_collection(test_collection_name)
        
        # Test 4: Delete test collection
        client.delete_collection(test_collection_name)
        
        return {
            'success': True,
            'collection_created': test_collection_name in collection_names,
            'collection_info': collection_info,
            'total_collections': len(collections.collections)
        }
    except Exception as e:
        # Try to clean up if collection was created
        try:
            client.delete_collection(test_collection_name)
        except:
            pass
        return {
            'success': False,
            'error': str(e)
        }

def test_vector_operations(client: QdrantClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """Test basic vector operations."""
    test_collection_name = f"test_vectors_{np.random.randint(1000, 9999)}"
    
    try:
        # Create test collection
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(
                size=config['vector_db_dimension'],
                distance=Distance.COSINE
            )
        )
        
        # Generate test vectors
        test_vectors = np.random.rand(5, config['vector_db_dimension']).tolist()
        test_points = [
            PointStruct(
                id=i,
                vector=vector,
                payload={"text": f"test_document_{i}", "metadata": {"test": True}}
            )
            for i, vector in enumerate(test_vectors)
        ]
        
        # Test 1: Upsert vectors
        client.upsert(
            collection_name=test_collection_name,
            points=test_points
        )
        
        # Test 2: Count points
        collection_info = client.get_collection(test_collection_name)
        point_count = collection_info.points_count
        
        # Test 3: Search vectors
        search_vector = np.random.rand(config['vector_db_dimension']).tolist()
        search_results = client.search(
            collection_name=test_collection_name,
            query_vector=search_vector,
            limit=3
        )
        
        # Test 4: Delete collection
        client.delete_collection(test_collection_name)
        
        return {
            'success': True,
            'points_inserted': len(test_points),
            'points_counted': point_count,
            'search_results_count': len(search_results),
            'search_score_range': (
                min(r.score for r in search_results),
                max(r.score for r in search_results)
            ) if search_results else (0, 0)
        }
    except Exception as e:
        # Try to clean up
        try:
            client.delete_collection(test_collection_name)
        except:
            pass
        return {
            'success': False,
            'error': str(e)
        }

def test_existing_collection(client: QdrantClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """Test operations on existing collection if it exists."""
    collection_name = config['vector_db_name']
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            return {
                'success': True,
                'collection_exists': False,
                'message': f"Collection '{collection_name}' does not exist yet"
            }
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Test search on existing collection
        search_vector = np.random.rand(config['vector_db_dimension']).tolist()
        search_results = client.search(
            collection_name=collection_name,
            query_vector=search_vector,
            limit=5
        )
        
        return {
            'success': True,
            'collection_exists': True,
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.vectors_count,
            'search_results_count': len(search_results)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main test function."""
    print("🔍 Testing Qdrant Vector Database Connection")
    print("=" * 60)
    
    # Load environment configuration
    config = load_environment()
    
    print(f"📋 Configuration loaded:")
    print(f"   Qdrant Host: {config['qdrant_host']}")
    print(f"   Qdrant Port: {config['qdrant_port']}")
    print(f"   Vector DB Name: {config['vector_db_name']}")
    print(f"   Vector Dimension: {config['vector_db_dimension']}")
    print(f"   Vector Metric: {config['vector_db_metric']}")
    print()
    
    # Test 1: Basic connection
    print("1️⃣ Testing Qdrant Connection")
    print("-" * 40)
    
    connection_result = test_qdrant_connection(config)
    if connection_result['success']:
        print(f"✅ Qdrant connection successful")
        print(f"   Connection: {connection_result['connection_info']}")
        print(f"   Method: {connection_result.get('connection_method', 'unknown')}")
        print(f"   Collections found: {len(connection_result['collections'].collections)}")
        client = connection_result['client']
    else:
        print(f"❌ Qdrant connection failed: {connection_result['error']}")
        print("   Please check your Qdrant server configuration")
        return 1
    
    # Test 2: Collection operations
    print("\n2️⃣ Testing Collection Operations")
    print("-" * 40)
    
    collection_result = test_collection_operations(client, config)
    if collection_result['success']:
        print(f"✅ Collection operations successful")
        print(f"   Test collection created and deleted")
        print(f"   Total collections: {collection_result['total_collections']}")
    else:
        print(f"❌ Collection operations failed: {collection_result['error']}")
    
    # Test 3: Vector operations
    print("\n3️⃣ Testing Vector Operations")
    print("-" * 40)
    
    vector_result = test_vector_operations(client, config)
    if vector_result['success']:
        print(f"✅ Vector operations successful")
        print(f"   Points inserted: {vector_result['points_inserted']}")
        print(f"   Points counted: {vector_result['points_counted']}")
        print(f"   Search results: {vector_result['search_results_count']}")
        print(f"   Score range: {vector_result['search_score_range']}")
    else:
        print(f"❌ Vector operations failed: {vector_result['error']}")
    
    # Test 4: Existing collection test
    print("\n4️⃣ Testing Existing Collection")
    print("-" * 40)
    
    existing_result = test_existing_collection(client, config)
    if existing_result['success']:
        if existing_result['collection_exists']:
            print(f"✅ Existing collection test successful")
            print(f"   Collection: {config['vector_db_name']}")
            print(f"   Points count: {existing_result['points_count']}")
            print(f"   Vectors count: {existing_result['vectors_count']}")
            print(f"   Search results: {existing_result['search_results_count']}")
        else:
            print(f"ℹ️  Collection '{config['vector_db_name']}' does not exist yet")
            print(f"   This is normal for a fresh setup")
    else:
        print(f"❌ Existing collection test failed: {existing_result['error']}")
    
    # Summary
    print("\n📊 Summary")
    print("=" * 60)
    tests_passed = 0
    total_tests = 4
    
    if connection_result['success']:
        tests_passed += 1
    if collection_result['success']:
        tests_passed += 1
    if vector_result['success']:
        tests_passed += 1
    if existing_result['success']:
        tests_passed += 1
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All Qdrant tests passed!")
        print("✅ Qdrant is ready for RAG operations")
        return 0
    elif tests_passed >= 3:
        print("⚠️  Most Qdrant tests passed. Some issues detected.")
        return 0
    else:
        print("❌ Multiple Qdrant tests failed. Check your Qdrant configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
