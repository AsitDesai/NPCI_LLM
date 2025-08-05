#!/usr/bin/env python3
"""
Environment setup script for LlamaIndex integration.

This script sets up the environment for the RAG system including:
- Directory structure creation
- LlamaIndex cache and persist directories
- Environment validation
- Dependencies check
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


def create_directory_structure() -> Dict[str, bool]:
    """
    Create the complete directory structure for the RAG system.
    
    Returns:
        Dictionary with directory creation status
    """
    directories = {
        "reference_documents": settings.reference_documents_dir,
        "storage": settings.storage_dir,
        "embeddings": settings.embeddings_dir,
        "uploads": settings.upload_dir,
        "logs": os.path.dirname(settings.log_file),
        "llama_index_cache": settings.llama_index_cache_dir,
        "llama_index_persist": settings.llama_index_persist_dir,
    }
    
    results = {}
    
    for name, path in directories.items():
        try:
            if path:
                os.makedirs(path, exist_ok=True)
                results[name] = True
                logger.info(f"✅ Created directory: {name} -> {path}")
            else:
                results[name] = False
                logger.warning(f"⚠️ Skipped directory: {name} (empty path)")
        except Exception as e:
            results[name] = False
            logger.error(f"❌ Failed to create directory {name}: {e}")
    
    return results


def validate_llama_index_environment() -> Dict[str, Any]:
    """
    Validate LlamaIndex environment and dependencies.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "llama_index_import": False,
        "embedding_model": False,
        "vector_store": False,
        "cache_dir": False,
        "persist_dir": False
    }
    
    try:
        # Test LlamaIndex imports
        import llama_index
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        
        validation_results["llama_index_import"] = True
        # LlamaIndex doesn't have __version__ attribute in newer versions
        try:
            version = llama_index.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ LlamaIndex version: {version}")
        
        # Test embedding model
        try:
            embedder = HuggingFaceEmbedding(
                model_name=settings.embedding_model_name,
                cache_folder=settings.llama_index_cache_dir
            )
            validation_results["embedding_model"] = True
            logger.info(f"✅ Embedding model: {settings.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ Embedding model validation failed: {e}")
        
        # Test vector store
        try:
            from qdrant_client import QdrantClient
            validation_results["vector_store"] = True
            logger.info("✅ Qdrant client import successful")
        except Exception as e:
            logger.error(f"❌ Vector store validation failed: {e}")
        
        # Test cache directory
        if os.path.exists(settings.llama_index_cache_dir):
            validation_results["cache_dir"] = True
            logger.info(f"✅ Cache directory exists: {settings.llama_index_cache_dir}")
        else:
            logger.warning(f"⚠️ Cache directory missing: {settings.llama_index_cache_dir}")
        
        # Test persist directory
        if os.path.exists(settings.llama_index_persist_dir):
            validation_results["persist_dir"] = True
            logger.info(f"✅ Persist directory exists: {settings.llama_index_persist_dir}")
        else:
            logger.warning(f"⚠️ Persist directory missing: {settings.llama_index_persist_dir}")
        
    except ImportError as e:
        logger.error(f"❌ LlamaIndex import failed: {e}")
    
    return validation_results


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dictionary with dependency check results
    """
    dependencies = {
        "llama_index": False,
        "qdrant_client": False,
        "sentence_transformers": False,
        "pydantic_settings": False,
        "structlog": False,
        "fastapi": False,
        "uvicorn": False
    }
    
    try:
        import llama_index
        dependencies["llama_index"] = True
        # LlamaIndex doesn't have __version__ attribute in newer versions
        try:
            version = llama_index.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ llama_index: {version}")
    except ImportError:
        logger.error("❌ llama_index: Not installed")
    
    try:
        import qdrant_client
        dependencies["qdrant_client"] = True
        try:
            version = qdrant_client.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ qdrant_client: {version}")
    except ImportError:
        logger.error("❌ qdrant_client: Not installed")
    
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
        try:
            version = sentence_transformers.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ sentence_transformers: {version}")
    except ImportError:
        logger.error("❌ sentence_transformers: Not installed")
    
    try:
        import pydantic_settings
        dependencies["pydantic_settings"] = True
        logger.info("✅ pydantic_settings: Installed")
    except ImportError:
        logger.error("❌ pydantic_settings: Not installed")
    
    try:
        import structlog
        dependencies["structlog"] = True
        try:
            version = structlog.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ structlog: {version}")
    except ImportError:
        logger.error("❌ structlog: Not installed")
    
    try:
        import fastapi
        dependencies["fastapi"] = True
        try:
            version = fastapi.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ fastapi: {version}")
    except ImportError:
        logger.error("❌ fastapi: Not installed")
    
    try:
        import uvicorn
        dependencies["uvicorn"] = True
        try:
            version = uvicorn.__version__
        except AttributeError:
            version = "unknown"
        logger.info(f"✅ uvicorn: {version}")
    except ImportError:
        logger.error("❌ uvicorn: Not installed")
    
    return dependencies


def setup_environment() -> bool:
    """
    Complete environment setup for LlamaIndex integration.
    
    Returns:
        True if setup successful, False otherwise
    """
    print("🚀 SETTING UP LLAMAINDEX ENVIRONMENT")
    print("="*60)
    
    try:
        # Step 1: Create directory structure
        print("\n📁 Creating directory structure...")
        dir_results = create_directory_structure()
        
        # Step 2: Check dependencies
        print("\n📦 Checking dependencies...")
        dep_results = check_dependencies()
        
        # Step 3: Validate LlamaIndex environment
        print("\n🔧 Validating LlamaIndex environment...")
        env_results = validate_llama_index_environment()
        
        # Summary
        print("\n" + "="*60)
        print("📊 SETUP SUMMARY")
        print("="*60)
        
        # Directory results
        dir_success = sum(dir_results.values())
        dir_total = len(dir_results)
        print(f"📁 Directories: {dir_success}/{dir_total} created successfully")
        
        # Dependency results
        dep_success = sum(dep_results.values())
        dep_total = len(dep_results)
        print(f"📦 Dependencies: {dep_success}/{dep_total} installed")
        
        # Environment results
        env_success = sum(env_results.values())
        env_total = len(env_results)
        print(f"🔧 Environment: {env_success}/{env_total} validated")
        
        # Overall success
        overall_success = (dir_success == dir_total and 
                          dep_success == dep_total and 
                          env_success == env_total)
        
        if overall_success:
            print("\n✅ ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!")
            print("🚀 Ready for LlamaIndex integration")
            return True
        else:
            print("\n⚠️ ENVIRONMENT SETUP COMPLETED WITH WARNINGS")
            print("Please check the logs above for issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment setup failed: {e}")
        print(f"\n❌ ENVIRONMENT SETUP FAILED: {e}")
        return False


if __name__ == "__main__":
    success = setup_environment()
    if not success:
        sys.exit(1) 