"""
Prompt Templates for RAG System

This module provides structured prompt templates for different use cases
and response styles using Mistral AI.
"""

from enum import Enum
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class PromptStyle(Enum):
    """Enumeration of available prompt styles."""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"


class PromptTemplates:
    """
    Manages prompt templates for different use cases and styles.
    
    This class provides structured prompts that can be customized
    based on the desired response style and context.
    """
    
    def __init__(self):
        """Initialize prompt templates."""
        logger.info("Initializing prompt templates")
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load all prompt templates."""
        return {
            "rag_base": self._get_rag_base_template(),
            "rag_concise": self._get_rag_concise_template(),
            "rag_detailed": self._get_rag_detailed_template(),
            "rag_technical": self._get_rag_technical_template(),
            "rag_friendly": self._get_rag_friendly_template(),
            "rag_professional": self._get_rag_professional_template(),
        }
    
    def get_template(self, style: PromptStyle = PromptStyle.CONCISE) -> str:
        """
        Get a prompt template for the specified style.
        
        Args:
            style: The desired prompt style
            
        Returns:
            The prompt template string
        """
        template_key = f"rag_{style.value}"
        if template_key not in self._templates:
            logger.warning(f"Template {template_key} not found, using concise")
            template_key = "rag_concise"
        
        return self._templates[template_key]
    
    def format_prompt(self, query: str, context: str, style: PromptStyle = PromptStyle.CONCISE) -> str:
        """
        Format a prompt for the LLM with specific instructions to reduce hallucination.
        
        Args:
            query: User query
            context: Retrieved context
            style: Response style
            
        Returns:
            Formatted prompt
        """
        base_instruction = """You are a precise assistant. Answer the user's question based ONLY on the provided context. Provide direct, factual answers without explanations or additional commentary.

CRITICAL RULES:
- Answer ONLY using information from the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Be direct and concise
- Do not add explanations, disclaimers, or additional information
- Do not mention what you can or cannot do
- Focus only on the specific answer requested

Context: {context}

Question: {query}

Answer:"""

        if style == PromptStyle.CONCISE:
            instruction = base_instruction + "\n\nProvide a concise, direct answer."
        elif style == PromptStyle.DETAILED:
            instruction = base_instruction + "\n\nProvide a detailed explanation with examples."
        elif style == PromptStyle.TECHNICAL:
            instruction = base_instruction + "\n\nProvide a technical, precise answer with specific steps."
        else:
            instruction = base_instruction
        
        prompt = instruction.format(context=context, query=query)
        
        logger.info(f"Formatted prompt with style: {style.value}")
        return prompt
    
    def _get_rag_base_template(self) -> str:
        """Get the base RAG template."""
        return """Based on the following context, please answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
    
    def _get_rag_concise_template(self) -> str:
        """Get the concise RAG template."""
        return """You are a helpful assistant. Provide a clear, concise answer based on the given context.

Context:
{context}

Question: {query}

Provide a brief, direct answer:"""
    
    def _get_rag_detailed_template(self) -> str:
        """Get the detailed RAG template."""
        return """You are a knowledgeable assistant. Provide a comprehensive, detailed answer based on the given context.

Context:
{context}

Question: {query}

Please provide a thorough explanation with relevant details:"""
    
    def _get_rag_technical_template(self) -> str:
        """Get the technical RAG template."""
        return """You are a technical expert. Provide a precise, technical answer based on the given context.

Context:
{context}

Question: {query}

Provide a technical, accurate response with specific details:"""
    
    def _get_rag_friendly_template(self) -> str:
        """Get the friendly RAG template."""
        return """You are a friendly and helpful assistant. Provide a warm, approachable answer based on the given context.

Context:
{context}

Question: {query}

Please provide a friendly, helpful response:"""
    
    def _get_rag_professional_template(self) -> str:
        """Get the professional RAG template."""
        return """You are a professional assistant. Provide a formal, well-structured answer based on the given context.

Context:
{context}

Question: {query}

Please provide a professional, structured response:""" 