"""
RAG System CLI Test

This script provides an interactive CLI to test the complete RAG pipeline,
integrating retrieval and generation systems.
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generation.prompt_templates import PromptTemplates, PromptStyle
from generation.generator import ResponseGenerator, MockResponseGenerator
from generation.post_processor import PostProcessor
from retrieval.retriever import SemanticRetriever
from retrieval.context_builder import ContextBuilder
from embeddings.embedder import LlamaIndexEmbedder
from embeddings.models import get_default_embedding_config
from embeddings.vector_store import QdrantVectorStore


class RAGCLI:
    """
    Interactive CLI for testing the complete RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the RAG CLI."""
        print("🚀 Initializing RAG System...")
        
        try:
            # Initialize components
            self.embedder = LlamaIndexEmbedder(get_default_embedding_config())
            self.vector_store = QdrantVectorStore()
            self.retriever = SemanticRetriever(self.embedder, self.vector_store)
            self.context_builder = ContextBuilder()
            
            self.prompt_templates = PromptTemplates()
            
            # Try to use real generator, fallback to mock
            try:
                self.generator = ResponseGenerator()
                print("✅ Real Mistral generator initialized")
            except Exception as e:
                print(f"⚠️ Real Mistral generator failed: {e}")
                print("🔄 Using mock generator for testing")
                from generation.generator import MockResponseGenerator
                self.generator = MockResponseGenerator()
            
            self.post_processor = PostProcessor()
            
            print("✅ RAG System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG System: {e}")
            raise
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve context for the query."""
        print(f"\n🔍 Retrieving context for: '{query}'")
        
        try:
            start_time = time.time()
            results = self.retriever.retrieve(query, top_k=3)
            retrieval_time = time.time() - start_time
            
            if not results:
                print("⚠️ No retrieval results found, using fallback context")
                return "Based on available information, I'll provide a general response."
            
            # Build context
            context_info = self.context_builder.build_context(results, query)
            context = context_info.context
            context_time = time.time() - start_time - retrieval_time
            
            print(f"✅ Retrieved {len(results)} results in {retrieval_time:.3f}s")
            print(f"✅ Built context in {context_time:.3f}s")
            print(f"📏 Context length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return "Error retrieving context. Using fallback."
    
    def build_prompt(self, query: str, context: str, style: str = "concise") -> str:
        """Build a prompt for generation."""
        print(f"\n📝 Building prompt with {style} style...")
        
        try:
            prompt = self.prompt_templates.format_prompt(
                query=query,
                context=context,
                style=PromptStyle(style)
            )
            
            print(f"✅ Prompt built ({len(prompt)} characters)")
            return prompt
            
        except Exception as e:
            print(f"❌ Error building prompt: {e}")
            raise
    
    def generate_response(self, prompt: str, style: str = "concise") -> str:
        """Generate a response using the LLM."""
        print(f"\n🤖 Generating response with {style} style...")
        
        try:
            start_time = time.time()
            response = self.generator.generate(prompt, style=style)
            generation_time = time.time() - start_time
            
            print(f"✅ Response generated in {generation_time:.3f}s")
            print(f"📏 Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def process_response(self, response: str) -> dict:
        """Process and enhance the response."""
        print(f"\n🔧 Processing response...")
        
        try:
            processed = self.post_processor.process(response)
            
            print(f"✅ Response processed")
            print(f"📊 Confidence: {processed['confidence']:.2f}")
            print(f"🎭 Tone: {processed['metadata']['tone']}")
            print(f"📝 Word count: {processed['metadata']['word_count']}")
            
            return processed
            
        except Exception as e:
            print(f"❌ Error processing response: {e}")
            return {"response": response, "confidence": 0.0}
    
    def display_response(self, processed_response: dict):
        """Display the final response."""
        print(f"\n{'='*60}")
        print("🎯 FINAL RESPONSE")
        print(f"{'='*60}")
        
        display = self.post_processor.format_for_display(processed_response)
        print(display)
        
        print(f"\n{'='*60}")
    
    def run_query(self, query: str, style: str = "concise"):
        """Run a complete RAG query."""
        print(f"\n{'='*60}")
        print(f"🔍 PROCESSING QUERY: '{query}'")
        print(f"🎨 STYLE: {style}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Retrieve context
            context = self.retrieve_context(query)
            
            # Step 2: Build prompt
            prompt = self.build_prompt(query, context, style)
            
            # Step 3: Generate response
            response = self.generate_response(prompt, style)
            
            # Step 4: Process response
            processed = self.process_response(response)
            
            # Step 5: Display result
            self.display_response(processed)
            
            return processed
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return None
    
    def interactive_mode(self):
        """Run interactive mode for testing."""
        print(f"\n{'='*60}")
        print("🎮 INTERACTIVE RAG TESTING")
        print(f"{'='*60}")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for available commands")
        print(f"{'='*60}")
        
        while True:
            try:
                query = input("\n🔍 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if not query:
                    print("⚠️ Please enter a query")
                    continue
                
                # Get style preference
                style = input("🎨 Enter style (concise/detailed/technical/friendly/professional) [concise]: ").strip()
                if not style:
                    style = "concise"
                
                # Run the query
                self.run_query(query, style)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information."""
        print(f"\n{'='*60}")
        print("📖 HELP - RAG SYSTEM COMMANDS")
        print(f"{'='*60}")
        print("Commands:")
        print("  quit/exit/q - Exit the program")
        print("  help - Show this help message")
        print("\nStyles:")
        print("  concise - Brief, direct answers")
        print("  detailed - Comprehensive explanations")
        print("  technical - Technical, precise responses")
        print("  friendly - Warm, approachable tone")
        print("  professional - Formal, structured responses")
        print(f"{'='*60}")


def main():
    """Main function to run the RAG CLI."""
    print("🚀 RAG SYSTEM CLI TEST")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        rag_cli = RAGCLI()
        
        # Run interactive mode
        rag_cli.interactive_mode()
        
    except Exception as e:
        print(f"❌ Failed to start RAG CLI: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 