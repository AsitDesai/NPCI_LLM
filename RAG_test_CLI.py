"""
RAG System CLI

Simple CLI for testing the RAG pipeline.
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
from embeddings.vector_store import QdrantVectorStore


class RAGCLI:
    """Simple RAG CLI for testing."""
    
    def __init__(self):
        """Initialize the RAG system."""
        print("🚀 Initializing RAG System...")
        
        try:
            # Initialize components
            self.embedder = LlamaIndexEmbedder()
            self.vector_store = QdrantVectorStore()
            self.retriever = SemanticRetriever(self.embedder, self.vector_store)
            self.context_builder = ContextBuilder()
            self.prompt_templates = PromptTemplates()
            
            # Initialize generator
            try:
                self.generator = ResponseGenerator()
                print("✅ Real Mistral generator initialized")
            except Exception as e:
                print(f"⚠️ Using mock generator: {e}")
                self.generator = MockResponseGenerator()
            
            self.post_processor = PostProcessor()
            print("✅ RAG System ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve context using semantic retrieval."""
        print(f"\n🔍 Retrieving context for: '{query}'")
        
        try:
            start_time = time.time()
            results = self.retriever.retrieve(query, top_k=3)
            retrieval_time = time.time() - start_time
            
            if not results:
                print("⚠️ No results found")
                return "Based on available information, I'll provide a general response."
            
            # Build context
            context_info = self.context_builder.build_context(results, query)
            context = context_info.context
            
            print(f"✅ Retrieved {len(results)} results in {retrieval_time:.3f}s")
            print(f"📏 Context length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return "Based on available information, I'll provide a general response."
    
    def build_prompt(self, query: str, context: str, style: str = "concise") -> str:
        """Build prompt."""
        try:
            if style == "concise":
                prompt_style = PromptStyle.CONCISE
            elif style == "detailed":
                prompt_style = PromptStyle.DETAILED
            elif style == "technical":
                prompt_style = PromptStyle.TECHNICAL
            else:
                prompt_style = PromptStyle.CONCISE
            
            prompt = self.prompt_templates.format_prompt(
                query=query,
                context=context,
                style=prompt_style
            )
            
            print(f"✅ Built {style} prompt ({len(prompt)} characters)")
            return prompt
            
        except Exception as e:
            print(f"❌ Error building prompt: {e}")
            return f"Query: {query}\nContext: {context}\nPlease provide a helpful response."
    
    def generate_response(self, prompt: str) -> str:
        """Generate response."""
        try:
            start_time = time.time()
            response = self.generator.generate(prompt)
            generation_time = time.time() - start_time
            
            print(f"✅ Generated response in {generation_time:.3f}s")
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def process_response(self, response: str) -> dict:
        """Process response."""
        try:
            processed = self.post_processor.process(response)
            print("✅ Response processed")
            return processed
            
        except Exception as e:
            print(f"❌ Error processing response: {e}")
            return {
                "original_response": response,
                "processed_response": response,
                "confidence": 0.5
            }
    
    def display_response(self, processed_response: dict):
        """Display response."""
        print(f"\n{'='*60}")
        print("🎯 RAG RESPONSE")
        print(f"{'='*60}")
        
        if isinstance(processed_response, dict):
            # Handle different possible response keys
            response = (processed_response.get('processed_response') or 
                       processed_response.get('response') or 
                       processed_response.get('original_response') or 
                       'No response')
            confidence = processed_response.get('confidence', 0.5)
            
            print(f"📝 Response: {response}")
            print(f"🎯 Confidence: {confidence:.2f}")
        else:
            print(f"📝 Response: {processed_response}")
        
        print(f"{'='*60}")
    
    def run_query(self, query: str, style: str = "concise"):
        """Run a complete RAG query."""
        print(f"\n{'='*60}")
        print(f"🔍 RAG QUERY: '{query}'")
        print(f"🎨 STYLE: {style}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Context retrieval
            context = self.retrieve_context(query)
            
            # Step 2: Build prompt
            prompt = self.build_prompt(query, context, style)
            
            # Step 3: Generate response
            response = self.generate_response(prompt)
            
            # Step 4: Process response
            processed = self.process_response(response)
            
            # Step 5: Display result
            self.display_response(processed)
            
            return processed
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return None
    
    def interactive_mode(self):
        """Run interactive mode."""
        print("\n🎮 RAG INTERACTIVE MODE")
        print("="*60)
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                query = input("\n🤖 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Run query
                self.run_query(query)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Run the RAG CLI."""
    print("🚀 RAG SYSTEM CLI")
    print("="*60)
    
    try:
        # Initialize RAG CLI
        rag_cli = RAGCLI()
        
        # Check if command line arguments provided
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            rag_cli.run_query(query)
        else:
            # Run interactive mode
            rag_cli.interactive_mode()
            
    except Exception as e:
        print(f"❌ Failed to start RAG CLI: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 