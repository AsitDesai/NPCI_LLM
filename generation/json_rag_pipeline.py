"""
JSON-specific RAG pipeline for question answering.

This module provides a complete RAG pipeline that:
- Uses JSON semantic retriever for document retrieval
- Sends query + retrieved payloads to Mistral 24B for answer generation
- Handles context building and response generation
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from retrieval.json_retriever import JSONSemanticRetriever, JSONRetrievalResult
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    query: str
    retrieved_documents: List[JSONRetrievalResult]
    generation_time: float
    retrieval_time: float
    total_time: float
    model_used: str


class JSONRAGPipeline:
    """
    JSON-specific RAG pipeline for question answering.
    
    This class provides a complete RAG pipeline that integrates
    JSON retrieval with Mistral 24B for answer generation.
    """
    
    def __init__(self, 
                 retriever: Optional[JSONSemanticRetriever] = None,
                 top_k: int = 5):
        """
        Initialize the JSON RAG pipeline.
        
        Args:
            retriever: JSON semantic retriever instance
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever or JSONSemanticRetriever()
        self.top_k = top_k
        self.mistral_model = settings.mistral_model
        
        # Initialize Mistral client
        self._init_mistral_client()
        
        logger.info(f"JSON RAG pipeline initialized with top_k={top_k}")
        logger.info(f"Mistral model: {self.mistral_model}")
    
    def _init_mistral_client(self):
        """Initialize Mistral client."""
        try:
            from mistralai import Mistral, UserMessage, AssistantMessage
            
            if not settings.mistral_api_key:
                raise ValueError("Mistral API key not configured")
            
            self.mistral_client = Mistral(api_key=settings.mistral_api_key)
            self.UserMessage = UserMessage
            self.AssistantMessage = AssistantMessage
            
            logger.info("Mistral client initialized successfully")
            
        except ImportError:
            logger.error("mistralai package not installed. Please install it with: pip install mistralai")
            raise
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {e}")
            raise
    
    def _build_context_from_retrieved_docs(self, 
                                         retrieved_docs: List[JSONRetrievalResult]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format each document with its metadata
            doc_context = f"Document {i}:\n"
            doc_context += f"Category: {doc.category}\n"
            doc_context += f"Type: {doc.chunk_type}\n"
            doc_context += f"Source: {doc.source_file}\n"
            doc_context += f"Relevance Score: {doc.score:.3f}\n"
            doc_context += f"Content:\n{doc.text}\n"
            doc_context += "-" * 50
            
            context_parts.append(doc_context)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt for Mistral 24B.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Please provide accurate, helpful, and concise answers based on the information given.

Context:
{context}

Question: {query}

Please provide a clear and helpful answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        return prompt
    
    def _generate_answer_with_mistral(self, prompt: str) -> str:
        """
        Generate answer using Mistral 24B.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated answer
        """
        try:
            # Create chat message
            messages = [
                self.UserMessage(content=prompt)
            ]
            
            # Generate response
            response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1  # Low temperature for more focused responses
            )
            
            answer = response.choices[0].message.content
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer with Mistral: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_query(self, 
                    query: str,
                    category_filter: Optional[str] = None,
                    type_filter: Optional[str] = None,
                    top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a query using the RAG pipeline.
        
        Args:
            query: User query
            category_filter: Optional category filter
            type_filter: Optional type filter
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query[:100]}...'")
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(
                query=query,
                top_k=top_k or self.top_k,
                category_filter=category_filter,
                type_filter=type_filter
            )
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                logger.warning("No documents retrieved for query")
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    query=query,
                    retrieved_documents=[],
                    generation_time=0.0,
                    retrieval_time=retrieval_time,
                    total_time=time.time() - start_time,
                    model_used=self.mistral_model
                )
            
            # Step 2: Build context from retrieved documents
            context = self._build_context_from_retrieved_docs(retrieved_docs)
            
            # Step 3: Build prompt
            prompt = self._build_prompt(query, context)
            
            # Step 4: Generate answer with Mistral 24B
            generation_start = time.time()
            answer = self._generate_answer_with_mistral(prompt)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            logger.info(f"Generated answer in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")
            
            return RAGResponse(
                answer=answer,
                query=query,
                retrieved_documents=retrieved_docs,
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=total_time,
                model_used=self.mistral_model
            )
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return RAGResponse(
                answer=f"Error processing your query: {str(e)}",
                query=query,
                retrieved_documents=[],
                generation_time=0.0,
                retrieval_time=0.0,
                total_time=time.time() - start_time,
                model_used=self.mistral_model
            )
    
    def answer_query_by_category(self, 
                               query: str, 
                               category: str,
                               top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a query filtered by category.
        
        Args:
            query: User query
            category: Category to filter by
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        return self.answer_query(
            query=query,
            category_filter=category,
            top_k=top_k
        )
    
    def answer_query_by_type(self, 
                           query: str, 
                           chunk_type: str,
                           top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a query filtered by type.
        
        Args:
            query: User query
            chunk_type: Type to filter by
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        return self.answer_query(
            query=query,
            type_filter=chunk_type,
            top_k=top_k
        )
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Get available categories and types for filtering.
        
        Returns:
            Dictionary with available categories and types
        """
        categories = self.retriever.get_available_categories()
        types = self.retriever.get_available_types()
        
        return {
            "categories": categories,
            "types": types
        }


def main():
    """Test the JSON RAG pipeline."""
    print("🧪 JSON RAG PIPELINE TEST")
    print("="*50)
    
    try:
        # Initialize RAG pipeline
        pipeline = JSONRAGPipeline(top_k=3)
        
        # Get available filters
        filters = pipeline.get_available_filters()
        print(f"\n📂 AVAILABLE FILTERS:")
        print(f"   Categories: {filters['categories']}")
        print(f"   Types: {filters['types']}")
        
        # Test queries
        test_queries = [
            "What should I do if I have insufficient balance?",
            "How do I reset my UPI PIN?",
            "Can I retry a failed transaction?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 TEST QUERY {i}:")
            print(f"   Query: '{query}'")
            
            # Get answer
            response = pipeline.answer_query(query)
            
            print(f"\n📋 RESPONSE:")
            print(f"   Answer: {response.answer}")
            print(f"   Retrieved documents: {len(response.retrieved_documents)}")
            print(f"   Retrieval time: {response.retrieval_time:.3f}s")
            print(f"   Generation time: {response.generation_time:.3f}s")
            print(f"   Total time: {response.total_time:.3f}s")
            
            # Show top retrieved document
            if response.retrieved_documents:
                top_doc = response.retrieved_documents[0]
                print(f"   Top document: {top_doc.category} - {top_doc.chunk_type} (score: {top_doc.score:.3f})")
        
        # Test category filtering
        if filters['categories']:
            print(f"\n🔍 TESTING CATEGORY FILTERING:")
            category_filter = filters['categories'][0]
            print(f"   Category filter: {category_filter}")
            
            category_response = pipeline.answer_query_by_category(
                "What should I do if my transaction fails?", 
                category_filter
            )
            
            print(f"   Answer: {category_response.answer[:200]}...")
            print(f"   Retrieved documents: {len(category_response.retrieved_documents)}")
        
        print(f"\n✅ JSON RAG pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"JSON RAG pipeline test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if not success:
        sys.exit(1)

