#!/usr/bin/env python3
"""
JSON RAG Pipeline Debug Script

This script provides step-by-step debugging of the JSON RAG pipeline:
1. Environment and dependencies check
2. Configuration validation
3. JSON data file validation
4. Data ingestion component test
5. Embedding model test
6. Qdrant connection test
7. Vector storage test
8. Retrieval system test
9. Mistral API test
10. Complete pipeline integration test
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


class JSONRAGDebugger:
    """Comprehensive debugger for JSON RAG pipeline."""
    
    def __init__(self):
        self.debug_results = {}
        self.current_step = 0
        self.total_steps = 10
    
    def print_step_header(self, step_name: str):
        """Print step header with progress."""
        self.current_step += 1
        print(f"\n{'='*80}")
        print(f"🔍 STEP {self.current_step}/{self.total_steps}: {step_name}")
        print(f"{'='*80}")
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"⚠️  {message}")
    
    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️  {message}")
    
    def step_1_environment_check(self) -> bool:
        """Step 1: Check environment and dependencies."""
        self.print_step_header("ENVIRONMENT AND DEPENDENCIES CHECK")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major >= 3 and python_version.minor >= 8:
                self.print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            else:
                self.print_error(f"Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+")
                return False
            
            # Check required packages
            required_packages = [
                'sentence_transformers',
                'qdrant_client',
                'mistralai',
                'torch',
                'transformers',
                'pydantic',
                'pydantic_settings'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                    self.print_success(f"✓ {package}")
                except ImportError:
                    missing_packages.append(package)
                    self.print_error(f"✗ {package} - NOT FOUND")
            
            if missing_packages:
                self.print_error(f"Missing packages: {', '.join(missing_packages)}")
                self.print_info("Install with: pip install " + " ".join(missing_packages))
                return False
            
            # Check project structure
            required_files = [
                'scripts/json_data_ingestion.py',
                'scripts/build_json_embeddings.py',
                'retrieval/json_retriever.py',
                'generation/json_rag_pipeline.py',
                'config/settings.py',
                'config/logging.py'
            ]
            
            missing_files = []
            for file_path in required_files:
                if Path(file_path).exists():
                    self.print_success(f"✓ {file_path}")
                else:
                    missing_files.append(file_path)
                    self.print_error(f"✗ {file_path} - NOT FOUND")
            
            if missing_files:
                self.print_error(f"Missing files: {', '.join(missing_files)}")
                return False
            
            self.debug_results['environment'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Environment check failed: {e}")
            return False
    
    def step_2_configuration_check(self) -> bool:
        """Step 2: Validate configuration settings."""
        self.print_step_header("CONFIGURATION VALIDATION")
        
        try:
            # Check essential settings
            essential_settings = {
                'embedding_model_name': settings.embedding_model_name,
                'embedding_model_dimension': settings.embedding_model_dimension,
                'vector_db_name': settings.vector_db_name,
                'vector_db_dimension': settings.vector_db_dimension,
                'qdrant_host': settings.qdrant_host,
                'qdrant_port': settings.qdrant_port,
                'reference_documents_dir': settings.reference_documents_dir
            }
            
            for key, value in essential_settings.items():
                if value:
                    self.print_success(f"✓ {key}: {value}")
                else:
                    self.print_error(f"✗ {key}: NOT SET")
                    return False
            
            # Check API keys
            if settings.mistral_api_key:
                self.print_success("✓ MISTRAL_API_KEY: SET")
            else:
                self.print_warning("⚠️ MISTRAL_API_KEY: NOT SET (will skip Mistral tests)")
            
            if settings.qdrant_api_key:
                self.print_success("✓ QDRANT_API_KEY: SET")
            else:
                self.print_warning("⚠️ QDRANT_API_KEY: NOT SET")
            
            # Validate model configuration
            if settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2":
                self.print_success("✓ Correct embedding model configured")
            else:
                self.print_warning(f"⚠️ Unexpected embedding model: {settings.embedding_model_name}")
            
            if settings.embedding_model_dimension == 384:
                self.print_success("✓ Correct vector dimension configured")
            else:
                self.print_warning(f"⚠️ Unexpected vector dimension: {settings.embedding_model_dimension}")
            
            self.debug_results['configuration'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Configuration check failed: {e}")
            return False
    
    def step_3_json_data_check(self) -> bool:
        """Step 3: Validate JSON data files."""
        self.print_step_header("JSON DATA FILE VALIDATION")
        
        try:
            documents_dir = Path(settings.reference_documents_dir)
            
            if not documents_dir.exists():
                self.print_error(f"Documents directory does not exist: {documents_dir}")
                return False
            
            self.print_success(f"✓ Documents directory exists: {documents_dir}")
            
            # Find JSON files
            json_files = list(documents_dir.glob("*.json"))
            
            if not json_files:
                self.print_error("No JSON files found in documents directory")
                return False
            
            self.print_success(f"✓ Found {len(json_files)} JSON file(s)")
            
            # Validate each JSON file
            valid_files = []
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, list):
                        self.print_error(f"✗ {json_file.name}: Not an array")
                        continue
                    
                    if len(data) == 0:
                        self.print_warning(f"⚠️ {json_file.name}: Empty array")
                        continue
                    
                    # Check first object structure
                    first_obj = data[0]
                    required_fields = ['category', 'type']
                    optional_fields = ['scenario', 'user_statement', 'agent_response', 'system_behavior', 'agent_guideline']
                    
                    missing_required = [field for field in required_fields if field not in first_obj]
                    if missing_required:
                        self.print_error(f"✗ {json_file.name}: Missing required fields: {missing_required}")
                        continue
                    
                    self.print_success(f"✓ {json_file.name}: {len(data)} objects, valid structure")
                    valid_files.append(json_file)
                    
                    # Show sample data
                    sample_obj = data[0]
                    self.print_info(f"   Sample object keys: {list(sample_obj.keys())}")
                    
                except json.JSONDecodeError as e:
                    self.print_error(f"✗ {json_file.name}: Invalid JSON - {e}")
                except Exception as e:
                    self.print_error(f"✗ {json_file.name}: Error - {e}")
            
            if not valid_files:
                self.print_error("No valid JSON files found")
                return False
            
            self.debug_results['json_data'] = True
            self.debug_results['valid_files'] = [f.name for f in valid_files]
            return True
            
        except Exception as e:
            self.print_error(f"JSON data check failed: {e}")
            return False
    
    def step_4_data_ingestion_test(self) -> bool:
        """Step 4: Test data ingestion component."""
        self.print_step_header("DATA INGESTION COMPONENT TEST")
        
        try:
            from scripts.json_data_ingestion import JSONDataIngestion
            
            # Initialize ingestion pipeline
            pipeline = JSONDataIngestion(max_tokens_per_chunk=200)
            self.print_success("✓ JSONDataIngestion initialized")
            
            # Process JSON files
            results = pipeline.process_json_files()
            
            if not results["success"]:
                self.print_error(f"Data ingestion failed: {results.get('error')}")
                return False
            
            chunks = results["chunks"]
            stats = results["statistics"]
            
            self.print_success(f"✓ Data ingestion completed")
            self.print_info(f"   Files processed: {stats.files_processed}")
            self.print_info(f"   Total chunks: {stats.total_chunks}")
            self.print_info(f"   Valid chunks: {stats.valid_chunks}")
            self.print_info(f"   Oversized chunks: {stats.oversized_chunks}")
            self.print_info(f"   Processing time: {stats.processing_time:.3f}s")
            self.print_info(f"   Average tokens per chunk: {stats.avg_tokens_per_chunk:.1f}")
            
            # Validate chunk structure
            if chunks:
                sample_chunk = chunks[0]
                required_attrs = ['chunk_id', 'text', 'payload', 'token_count']
                
                for attr in required_attrs:
                    if hasattr(sample_chunk, attr):
                        self.print_success(f"✓ Chunk has {attr}")
                    else:
                        self.print_error(f"✗ Chunk missing {attr}")
                        return False
                
                # Check token count
                if sample_chunk.token_count <= 200:
                    self.print_success(f"✓ Token count within limit: {sample_chunk.token_count}")
                else:
                    self.print_error(f"✗ Token count exceeds limit: {sample_chunk.token_count}")
                    return False
                
                # Check text length
                if len(sample_chunk.text) > 0:
                    self.print_success(f"✓ Text content present: {len(sample_chunk.text)} chars")
                else:
                    self.print_error("✗ No text content")
                    return False
            
            self.debug_results['data_ingestion'] = True
            self.debug_results['chunks_count'] = len(chunks)
            return True
            
        except Exception as e:
            self.print_error(f"Data ingestion test failed: {e}")
            return False
    
    def _ensure_float_embeddings(self, embeddings):
        """Ensure embeddings are proper float lists for Qdrant."""
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        
        if isinstance(embeddings, list):
            if len(embeddings) > 0 and hasattr(embeddings[0], 'tolist'):
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            
            # Convert all values to float for Qdrant compatibility
            float_embeddings = []
            for emb in embeddings:
                float_emb = [float(val) for val in emb]
                float_embeddings.append(float_emb)
            return float_embeddings
        
        return embeddings

    def step_5_embedding_model_test(self) -> bool:
        """Step 5: Test embedding model."""
        self.print_step_header("EMBEDDING MODEL TEST")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Initialize model with CUDA memory management
            model_name = settings.embedding_model_name
            
            # Smart device selection based on GPU memory availability
            gpu_available = False
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory
                free_gb = free_memory / 1024**3
                
                # Consider GPU available if we have at least 2GB free
                gpu_available = free_gb > 2.0
                self.print_info(f"GPU memory: {free_gb:.1f}GB free, {allocated_memory/1024**3:.1f}GB allocated")
            
            if gpu_available:
                try:
                    torch.cuda.empty_cache()
                    model = SentenceTransformer(model_name, device='cuda')
                    self.print_success(f"✓ Model loaded on GPU: {model_name}")
                except Exception as e:
                    self.print_warning(f"⚠️ GPU loading failed: {e}, falling back to CPU")
                    gpu_available = False
            
            if not gpu_available:
                model = SentenceTransformer(model_name, device='cpu')
                self.print_success(f"✓ Model loaded on CPU: {model_name}")
            
            # Test embedding generation with smaller batch size
            test_texts = [
                "This is a test sentence for embedding generation.",
                "Another test sentence to verify the model works correctly."
            ]
            
            start_time = time.time()
            try:
                embeddings = model.encode(test_texts, convert_to_list=True, batch_size=1)
                # Ensure embeddings are proper float lists for Qdrant
                embeddings = self._ensure_float_embeddings(embeddings)
            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cublas" in error_msg:
                    # If GPU encoding fails, force CPU mode
                    self.print_warning(f"⚠️ GPU encoding failed ({e}), forcing CPU mode")
                    model = SentenceTransformer(model_name, device='cpu')
                    embeddings = model.encode(test_texts, convert_to_list=True, batch_size=1)
                    # Ensure embeddings are proper float lists for Qdrant
                    embeddings = self._ensure_float_embeddings(embeddings)
                else:
                    raise e
            
            generation_time = time.time() - start_time
            
            self.print_success(f"✓ Embeddings generated in {generation_time:.3f}s")
            
            # Validate embeddings
            if len(embeddings) == len(test_texts):
                self.print_success(f"✓ Correct number of embeddings: {len(embeddings)}")
            else:
                self.print_error(f"✗ Wrong number of embeddings: {len(embeddings)}")
                return False
            
            # Check embedding dimension
            embedding_dim = len(embeddings[0])
            expected_dim = settings.embedding_model_dimension
            
            if embedding_dim == expected_dim:
                self.print_success(f"✓ Correct embedding dimension: {embedding_dim}")
            else:
                self.print_error(f"✗ Wrong embedding dimension: {embedding_dim}, expected: {expected_dim}")
                return False
            
            # Check embedding values - ensure they are proper lists of floats for Qdrant
            valid_embeddings = True
            for emb in embeddings:
                if not isinstance(emb, list):
                    valid_embeddings = False
                    break
                for val in emb:
                    # Qdrant expects float values, not ints or bools
                    if not isinstance(val, float):
                        valid_embeddings = False
                        break
                if not valid_embeddings:
                    break
            
            if valid_embeddings:
                self.print_success("✓ Embeddings are valid float lists for Qdrant")
            else:
                # Debug: print the actual format
                self.print_error("✗ Invalid embedding format for Qdrant")
                self.print_info(f"   First embedding type: {type(embeddings[0])}")
                if embeddings and len(embeddings) > 0:
                    self.print_info(f"   First embedding length: {len(embeddings[0])}")
                    if len(embeddings[0]) > 0:
                        self.print_info(f"   First value type: {type(embeddings[0][0])}")
                        self.print_info(f"   First value: {embeddings[0][0]}")
                        # Try to convert to proper format
                        try:
                            float_embeddings = [[float(val) for val in emb] for emb in embeddings]
                            self.print_info("   Converted to float format successfully")
                            embeddings = float_embeddings
                            valid_embeddings = True
                        except Exception as conv_e:
                            self.print_error(f"   Failed to convert to float format: {conv_e}")
                return False
            
            self.debug_results['embedding_model'] = True
            self.debug_results['embedding_dimension'] = embedding_dim
            return True
            
        except Exception as e:
            self.print_error(f"Embedding model test failed: {e}")
            return False
    
    def step_6_qdrant_connection_test(self) -> bool:
        """Step 6: Test Qdrant connection."""
        self.print_step_header("QDRANT CONNECTION TEST")
        
        try:
            from qdrant_client import QdrantClient
            
            # Initialize client for local Qdrant
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                https=False,
                timeout=10.0
            )
            self.print_info(f"ℹ️ Connecting to local Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
            
            # Test connection
            try:
                collections = client.get_collections()
                self.print_success("✓ Qdrant connection successful")
                self.print_info(f"   Available collections: {len(collections.collections)}")
                
                for collection in collections.collections:
                    self.print_info(f"   - {collection.name}")
                
            except Exception as e:
                self.print_error(f"✗ Qdrant connection failed: {e}")
                self.print_info("   Make sure Qdrant is running and accessible")
                return False
            
            self.debug_results['qdrant_connection'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Qdrant connection test failed: {e}")
            return False
    
    def step_7_vector_storage_test(self) -> bool:
        """Step 7: Test vector storage operations."""
        self.print_step_header("VECTOR STORAGE TEST")
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            from sentence_transformers import SentenceTransformer
            
            # Initialize components for local Qdrant
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                https=False
            )
            
            # Initialize model with CUDA memory management
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    model = SentenceTransformer(settings.embedding_model_name, device='cuda')
                else:
                    model = SentenceTransformer(settings.embedding_model_name, device='cpu')
            except Exception as e:
                if "out of memory" in str(e).lower():
                    import torch
                    torch.cuda.empty_cache()
                    model = SentenceTransformer(settings.embedding_model_name, device='cpu')
                else:
                    raise e
            test_collection = "answers_collection"
            
            # Create test collection
            try:
                vector_params = VectorParams(
                    size=settings.embedding_model_dimension,
                    distance=Distance.COSINE
                )
                
                client.create_collection(
                    collection_name=test_collection,
                    vectors_config=vector_params
                )
                self.print_success(f"✓ Test collection created: {test_collection}")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    self.print_info(f"ℹ️ Test collection already exists: {test_collection}")
                else:
                    self.print_error(f"✗ Failed to create test collection: {e}")
                    return False
            
            # Test vector upload with real data
            from scripts.json_data_ingestion import JSONDataIngestion
            
            # Get real data for testing
            ingestion = JSONDataIngestion(max_tokens_per_chunk=200)
            results = ingestion.process_json_files()
            
            if results["success"] and results["chunks"]:
                # Use first 2 chunks from real data
                real_chunks = results["chunks"][:2]
                test_texts = [chunk.text for chunk in real_chunks]
                
                try:
                    test_embeddings = model.encode(test_texts, convert_to_list=True, batch_size=1)
                    # Ensure embeddings are proper float lists for Qdrant
                    test_embeddings = self._ensure_float_embeddings(test_embeddings)
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        # Force CPU mode if still out of memory
                        model = SentenceTransformer(settings.embedding_model_name, device='cpu')
                        test_embeddings = model.encode(test_texts, convert_to_list=True, batch_size=1)
                        # Ensure embeddings are proper float lists for Qdrant
                        test_embeddings = self._ensure_float_embeddings(test_embeddings)
                    else:
                        raise e
                
                points = []
                for i, (chunk, embedding) in enumerate(zip(real_chunks, test_embeddings)):
                    point = PointStruct(
                        id=i,
                        vector=embedding,
                        payload=chunk.payload  # Use real payload with real categories/types
                    )
                    points.append(point)
                
                self.print_info(f"ℹ️ Using real data: {len(real_chunks)} chunks")
            else:
                # Fallback to test data if real data fails
                test_texts = ["Test sentence 1", "Test sentence 2"]
                test_embeddings = model.encode(test_texts, convert_to_list=True, batch_size=1)
                test_embeddings = self._ensure_float_embeddings(test_embeddings)
                
                points = []
                for i, (text, embedding) in enumerate(zip(test_texts, test_embeddings)):
                    point = PointStruct(
                        id=i,
                        vector=embedding,
                        payload={
                            "text": text, 
                            "index": i,
                            "category": "test_category",
                            "type": "test_type"
                        }
                    )
                    points.append(point)
            
            client.upsert(
                collection_name=test_collection,
                points=points
            )
            self.print_success(f"✓ Test vectors uploaded: {len(points)} points")
            
            # Test vector search
            query_text = "Test sentence"
            try:
                query_embedding = model.encode([query_text], convert_to_list=True, batch_size=1)[0]
                # Ensure embedding is proper float list for Qdrant
                query_embedding = self._ensure_float_embeddings([query_embedding])[0]
            except Exception as e:
                if "out of memory" in str(e).lower():
                    # Force CPU mode if still out of memory
                    model = SentenceTransformer(settings.embedding_model_name, device='cpu')
                    query_embedding = model.encode([query_text], convert_to_list=True, batch_size=1)[0]
                    # Ensure embedding is proper float list for Qdrant
                    query_embedding = self._ensure_float_embeddings([query_embedding])[0]
                else:
                    raise e
            
            search_results = client.search(
                collection_name=test_collection,
                query_vector=query_embedding,
                limit=2
            )
            
            if search_results:
                self.print_success(f"✓ Vector search successful: {len(search_results)} results")
                for i, result in enumerate(search_results):
                    self.print_info(f"   Result {i+1}: score={result.score:.3f}, payload={result.payload}")
            else:
                self.print_error("✗ Vector search returned no results")
                return False
            
            # Note: Not cleaning up test collection to preserve it for step 8
            self.print_info(f"ℹ️ Test collection '{test_collection}' preserved for retrieval testing")
            
            self.debug_results['vector_storage'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Vector storage test failed: {e}")
            return False
    
    def step_8_retrieval_system_test(self) -> bool:
        """Step 8: Test retrieval system."""
        self.print_step_header("RETRIEVAL SYSTEM TEST")
        
        try:
            from retrieval.json_retriever import JSONSemanticRetriever
            
            # Initialize retriever
            retriever = JSONSemanticRetriever()
            self.print_success("✓ JSONSemanticRetriever initialized")
            
            # Test collection info
            collection_info = retriever.get_collection_info()
            if "error" not in collection_info:
                self.print_success("✓ Collection info retrieved")
                self.print_info(f"   Collection: {collection_info.get('collection_name')}")
                self.print_info(f"   Vector size: {collection_info.get('vector_size')}")
                self.print_info(f"   Points count: {collection_info.get('points_count')}")
            else:
                self.print_warning(f"⚠️ Collection info error: {collection_info.get('error')}")
                self.print_info("   This is expected if no data has been ingested yet")
            
            # Test available filters (these might fail if collection doesn't exist, which is OK)
            try:
                categories = retriever.get_available_categories()
                if categories:
                    self.print_success(f"✓ Available categories: {len(categories)}")
                    self.print_info(f"   Categories: {categories[:3]}...")  # Show first 3
                else:
                    self.print_warning("⚠️ No categories found")
            except Exception as e:
                self.print_warning(f"⚠️ Could not get categories: {e}")
            
            try:
                types = retriever.get_available_types()
                if types:
                    self.print_success(f"✓ Available types: {len(types)}")
                    self.print_info(f"   Types: {types}")
                else:
                    self.print_warning("⚠️ No types found")
            except Exception as e:
                self.print_warning(f"⚠️ Could not get types: {e}")
            
            # Test retrieval (only if we have data)
            if collection_info.get('points_count', 0) > 0:
                # Use a query that should match the real UPI decline data
                test_query = "balance insufficient funds"
                results = retriever.retrieve(test_query, top_k=3)
                
                if results:
                    self.print_success(f"✓ Retrieval successful: {len(results)} results")
                    for i, result in enumerate(results[:2]):  # Show first 2
                        self.print_info(f"   Result {i+1}: {result.category} - {result.chunk_type} (score: {result.score:.3f})")
                else:
                    self.print_warning("⚠️ No retrieval results (may be normal if no data)")
            else:
                self.print_info("ℹ️ Skipping retrieval test (no data in collection)")
            
            self.debug_results['retrieval_system'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Retrieval system test failed: {e}")
            return False
    
    def step_9_mistral_api_test(self) -> bool:
        """Step 9: Test Mistral API."""
        self.print_step_header("MISTRAL API TEST")
        
        if not settings.mistral_api_key:
            self.print_warning("⚠️ MISTRAL_API_KEY not set, skipping Mistral test")
            self.debug_results['mistral_api'] = False
            return True
        
        try:
            import mistralai
            from mistralai import Mistral, UserMessage
            
            # Initialize client
            client = Mistral(api_key=settings.mistral_api_key)
            self.print_success("✓ Mistral client initialized")
            
            # Test simple completion
            messages = [
                UserMessage(content="Hello, this is a test message. Please respond with 'Test successful'.")
            ]
            
            start_time = time.time()
            response = client.chat.complete(
                model=settings.mistral_model,
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            generation_time = time.time() - start_time
            
            if response and response.choices:
                answer = response.choices[0].message.content
                self.print_success(f"✓ Mistral API test successful")
                self.print_info(f"   Response: {answer[:100]}...")
                self.print_info(f"   Generation time: {generation_time:.3f}s")
                self.print_info(f"   Model used: {settings.mistral_model}")
            else:
                self.print_error("✗ Mistral API returned no response")
                return False
            
            self.debug_results['mistral_api'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Mistral API test failed: {e}")
            self.debug_results['mistral_api'] = False
            return False
    
    def step_10_integration_test(self) -> bool:
        """Step 10: Test complete pipeline integration."""
        self.print_step_header("COMPLETE PIPELINE INTEGRATION TEST")
        
        try:
            # Only test if we have critical components working
            required_components = ['data_ingestion', 'qdrant_connection']
            missing_components = [comp for comp in required_components if not self.debug_results.get(comp, False)]
            
            if missing_components:
                self.print_warning(f"⚠️ Skipping integration test - missing components: {missing_components}")
                return True
            
            # Note: embedding_model test failure is not critical if vector_storage test passed
            if not self.debug_results.get('embedding_model', False) and self.debug_results.get('vector_storage', False):
                self.print_info("ℹ️ Embedding model test failed but vector storage works - continuing with integration test")
            
            # Test with a small subset
            from scripts.json_data_ingestion import JSONDataIngestion
            from scripts.build_json_embeddings import JSONEmbeddingBuilder
            
            # Process a small amount of data
            ingestion = JSONDataIngestion(max_tokens_per_chunk=200)
            results = ingestion.process_json_files()
            
            if not results["success"] or not results["chunks"]:
                self.print_warning("⚠️ No data to test integration with")
                return True
            
            # Take only first few chunks for quick test
            test_chunks = results["chunks"][:5]
            self.print_info(f"ℹ️ Testing with {len(test_chunks)} chunks")
            
            # Test embedding generation
            embedding_builder = JSONEmbeddingBuilder(batch_size=2)
            
            # Test embedding generation only (skip upload for debug)
            try:
                embeddings = embedding_builder.generate_embeddings(test_chunks)
                self.print_success(f"✓ Integration test successful")
                self.print_info(f"   Generated {len(embeddings)} embeddings")
                self.print_info(f"   Vector dimension: {len(embeddings[0].vector)}")
            except Exception as e:
                self.print_error(f"✗ Integration test failed: {e}")
                return False
            
            self.debug_results['integration'] = True
            return True
            
        except Exception as e:
            self.print_error(f"Integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all debug tests."""
        print("🚀 JSON RAG PIPELINE DEBUG SCRIPT")
        print("="*80)
        print("This script will test all components of the JSON RAG pipeline")
        print("to ensure everything is working correctly before running the full pipeline.")
        print("="*80)
        
        tests = [
            ("Environment Check", self.step_1_environment_check),
            ("Configuration Validation", self.step_2_configuration_check),
            ("JSON Data Validation", self.step_3_json_data_check),
            ("Data Ingestion Test", self.step_4_data_ingestion_test),
            ("Embedding Model Test", self.step_5_embedding_model_test),
            ("Qdrant Connection Test", self.step_6_qdrant_connection_test),
            ("Vector Storage Test", self.step_7_vector_storage_test),
            ("Retrieval System Test", self.step_8_retrieval_system_test),
            ("Mistral API Test", self.step_9_mistral_api_test),
            ("Integration Test", self.step_10_integration_test)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                else:
                    print(f"\n❌ {test_name} FAILED")
            except Exception as e:
                print(f"\n❌ {test_name} FAILED WITH EXCEPTION: {e}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 DEBUG SUMMARY")
        print(f"{'='*80}")
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The JSON RAG pipeline is ready to use.")
            print("💡 You can now run: python run_json_rag_pipeline.py")
        elif passed_tests >= total_tests - 1:  # Allow one failure (e.g., Mistral API)
            print("\n✅ MOST TESTS PASSED! The pipeline should work with some limitations.")
            print("⚠️  Check the failed tests above for details.")
        else:
            print("\n❌ MULTIPLE TESTS FAILED! Please fix the issues before running the pipeline.")
        
        # Print detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for key, value in self.debug_results.items():
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"   {key}: {status}")
        
        return passed_tests >= total_tests - 1  # Allow one failure


def main():
    """Main debug function."""
    debugger = JSONRAGDebugger()
    success = debugger.run_all_tests()
    
    if not success:
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify your .env file has the correct configuration")
        print("3. Ensure Qdrant is running (local or cloud)")
        print("4. Check that your JSON files are in the correct format")
        print("5. Verify your API keys are set correctly")
        print("6. Check the logs for detailed error messages")
    
    return success


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
