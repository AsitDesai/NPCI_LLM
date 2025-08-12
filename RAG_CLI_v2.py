#!/usr/bin/env python3
"""
RAG CLI v2 - Interactive Query Interface with Server Endpoint Integration

This script provides an interactive command-line interface for querying the JSON RAG pipeline
with support for both server endpoint and Mistral API fallback.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging import get_logger
from generation.json_rag_pipeline import JSONRAGPipeline, RAGResponse

logger = get_logger(__name__)


class RAGCLI:
    """Interactive CLI for JSON RAG pipeline."""
    
    def __init__(self):
        """Initialize the RAG CLI."""
        self.rag_pipeline = None
        self.available_categories = []
        self.available_types = []
        
    def initialize_pipeline(self):
        """Initialize the RAG pipeline."""
        try:
            print("🚀 Initializing RAG Pipeline...")
            self.rag_pipeline = JSONRAGPipeline()
            print("✅ RAG Pipeline initialized successfully")
            
            # Get available filters
            self._load_available_filters()
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            return False
        
        return True
    
    def _load_available_filters(self):
        """Load available categories and types for filtering."""
        try:
            print("📂 Loading available filters...")
            
            # Get categories and types from retriever
            self.available_categories = self.rag_pipeline.retriever.get_available_categories()
            self.available_types = self.rag_pipeline.retriever.get_available_types()
            
            print(f"✅ Loaded {len(self.available_categories)} categories and {len(self.available_types)} types")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load filters: {e}")
            self.available_categories = []
            self.available_types = []
    
    def show_available_filters(self):
        """Display available categories and types."""
        print("\n" + "="*60)
        print("📂 AVAILABLE FILTERS")
        print("="*60)
        
        if self.available_categories:
            print("\n📊 CATEGORIES:")
            for i, category in enumerate(self.available_categories, 1):
                print(f"   {i}. {category}")
        else:
            print("\n📊 CATEGORIES: None available")
        
        if self.available_types:
            print("\n🏷️  TYPES:")
            for i, type_name in enumerate(self.available_types, 1):
                print(f"   {i}. {type_name}")
        else:
            print("\n🏷️  TYPES: None available")
        
        print("="*60)
    
    def get_user_filters(self) -> tuple[Optional[str], Optional[str]]:
        """Get category and type filters from user."""
        # Always use no filters as requested
        return None, None
    
    def _show_categories(self):
        """Show available categories."""
        print("\n📊 Available Categories:")
        for i, category in enumerate(self.available_categories, 1):
            print(f"   {i}. {category}")
    
    def _show_types(self):
        """Show available types."""
        print("\n🏷️  Available Types:")
        for i, type_name in enumerate(self.available_types, 1):
            print(f"   {i}. {type_name}")
    
    def _get_category_choice(self) -> Optional[str]:
        """Get category choice from user."""
        try:
            choice = input(f"\nSelect category (1-{len(self.available_categories)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(self.available_categories):
                return self.available_categories[index]
            else:
                print("⚠️ Invalid category choice")
                return None
        except (ValueError, IndexError):
            print("⚠️ Invalid category choice")
            return None
    
    def _get_type_choice(self) -> Optional[str]:
        """Get type choice from user."""
        try:
            choice = input(f"\nSelect type (1-{len(self.available_types)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(self.available_types):
                return self.available_types[index]
            else:
                print("⚠️ Invalid type choice")
                return None
        except (ValueError, IndexError):
            print("⚠️ Invalid type choice")
            return None
    
    def process_query(self, query: str, category_filter: Optional[str] = None, type_filter: Optional[str] = None):
        """Process a single query."""
        try:
            print(f"\n🔍 Processing query: '{query}'")
            if category_filter:
                print(f"   Category filter: {category_filter}")
            if type_filter:
                print(f"   Type filter: {type_filter}")
            
            start_time = time.time()
            
            # Get response from RAG pipeline
            response = self.rag_pipeline.answer_query(
                query=query,
                category_filter=category_filter,
                type_filter=type_filter,
                top_k=5
            )
            
            total_time = time.time() - start_time
            
            # Display results
            self._display_response(response, total_time)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Error processing query: {e}")
    
    def _display_response(self, response: RAGResponse, total_time: float):
        """Display the RAG response."""
        print("\n" + "="*80)
        print("📋 RAG RESPONSE")
        print("="*80)
        
        # Answer
        print(f"\n💬 ANSWER:")
        print(f"{response.answer}")
        
        # Metadata
        print(f"\n📊 METADATA:")
        print(f"   Model used: {response.model_used}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Retrieval time: {response.retrieval_time:.3f}s")
        print(f"   Generation time: {response.generation_time:.3f}s")
        print(f"   Documents retrieved: {len(response.retrieved_documents)}")
        
        # Retrieved documents
        if response.retrieved_documents:
            print(f"\n📚 RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(response.retrieved_documents, 1):
                print(f"\n   {i}. Score: {doc.score:.3f}")
                print(f"      Category: {doc.category}")
                print(f"      Type: {doc.chunk_type}")
                print(f"      Text: {doc.text[:200]}...")
        
        print("="*80)
    
    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n🎯 INTERACTIVE MODE")
        print("Type 'quit' or 'exit' to stop, 'help' for commands")
        print("Type 'clear' to clear screen")
        
        while True:
            try:
                print("\n" + "-"*60)
                query = input("❓ Enter your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'clear':
                    print("\n" * 50)
                    continue
                
                # Process query with no filters
                self.process_query(query, None, None)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*60)
        print("📖 HELP")
        print("="*60)
        print("Commands:")
        print("  quit/exit/q - Exit the program")
        print("  help - Show this help")
        print("  clear - Clear the screen")
        print("\nUsage:")
        print("  1. Enter your question")
        print("  2. View the generated answer with metadata")
        print("  3. All queries use no filters (search all data)")
        print("="*60)
    
    def run(self):
        """Main CLI runner."""
        print("🚀 JSON RAG CLI v2 - Server Endpoint Integration")
        print("="*60)
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            return
        
        # Start interactive mode
        self.interactive_mode()


def main():
    """Main function."""
    try:
        cli = RAGCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error in CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
