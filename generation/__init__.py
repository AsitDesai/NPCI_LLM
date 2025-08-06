"""
Generation Module for RAG System

This module handles response generation using Mistral AI as the base model.
It includes prompt engineering, response generation, and post-processing.

Components:
- PromptTemplates: Structured prompt templates for different use cases
- ResponseGenerator: Main generation class using Mistral AI
- PostProcessor: Response formatting and enhancement
"""

from .prompt_templates import PromptTemplates, PromptStyle
from .generator import ResponseGenerator, MistralResponseGenerator
from .post_processor import PostProcessor

__all__ = [
    "PromptTemplates",
    "PromptStyle", 
    "ResponseGenerator",
    "MistralResponseGenerator",
    "PostProcessor"
] 