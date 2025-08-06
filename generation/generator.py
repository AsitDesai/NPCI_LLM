"""
Response Generator using Mistral AI

This module handles response generation using Mistral AI as the base model.
It includes the main ResponseGenerator class and MistralResponseGenerator implementation.
"""

import os
from typing import Optional, Dict, Any
import structlog
from mistralai import Mistral, UserMessage, AssistantMessage
from dotenv import load_dotenv
load_dotenv()
logger = structlog.get_logger(__name__)


class ResponseGenerator:
    """
    Main response generator class.
    
    This class provides a unified interface for generating responses
    using different LLM backends, with Mistral AI as the primary choice.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        logger.info("Initializing response generator")
        self.mistral_generator = MistralResponseGenerator()
    
    def generate(self, prompt: str, style: str = "concise", **kwargs) -> str:
        """
        Generate a response with the specified style.
        
        Args:
            prompt: The formatted prompt
            style: The response style (concise, detailed, etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        try:
            # Map style to temperature for better control
            temperature_map = {
                "concise": 0.3,
                "detailed": 0.7,
                "technical": 0.2,
                "friendly": 0.8,
                "professional": 0.5
            }
            
            temperature = temperature_map.get(style.lower(), 0.7)
            
            return self.mistral_generator.generate_response(
                prompt, 
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


class MistralResponseGenerator:
    """
    Mistral AI response generator implementation.
    
    This class handles direct interaction with the Mistral AI API
    for generating responses.
    """
    
    def __init__(self):
        """Initialize the Mistral AI generator."""
        logger.info("Initializing Mistral AI response generator")
        
        # Get API key from environment
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            logger.warning("MISTRAL_API_KEY not found in environment variables")
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        # Initialize Mistral client
        try:
            self.client = Mistral(api_key=self.api_key)
            logger.info("Mistral AI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral AI client: {e}")
            raise
    
    def generate_response(self, 
                        prompt: str, 
                        max_tokens: Optional[int] = None,
                        temperature: float = 0.7,
                        stream: bool = False) -> str:
        """
        Generate a response from the Mistral AI model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (None for default)
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response
            
        Returns:
            Generated response string
        """
        try:
            # Create messages for Mistral
            messages = [UserMessage(content=prompt)]
            
            # Generate response
            if stream:
                response = self._generate_streaming_response(messages, max_tokens, temperature)
            else:
                response = self._generate_complete_response(messages, max_tokens, temperature)
            
            logger.info(f"Generated response with {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error generating Mistral response: {e}")
            raise
    
    def _generate_complete_response(self, 
                                  messages: list, 
                                  max_tokens: Optional[int],
                                  temperature: float) -> str:
        """Generate a complete response (non-streaming)."""
        try:
            # Set default max_tokens if not provided
            if max_tokens is None:
                max_tokens = 1000
            
            # Generate response
            response = self.client.chat.complete(
                model="mistral-small-latest",  # Using the latest small model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("No response content received from Mistral")
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Error in complete response generation: {e}")
            raise
    
    def _generate_streaming_response(self, 
                                   messages: list, 
                                   max_tokens: Optional[int],
                                   temperature: float) -> str:
        """Generate a streaming response."""
        try:
            # Set default max_tokens if not provided
            if max_tokens is None:
                max_tokens = 1000
            
            # Generate streaming response
            stream = self.client.chat.stream(
                model="mistral-small-latest",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Collect the full response
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in streaming response generation: {e}")
            raise


# Mock classes for testing without API key
class MockMistralResponseGenerator:
    """Mock implementation for testing without API key."""
    
    def __init__(self):
        logger.info("Initializing mock Mistral response generator")
    
    def generate_response(self, 
                        prompt: str, 
                        max_tokens: Optional[int] = None,
                        temperature: float = 0.7,
                        stream: bool = False) -> str:
        """Generate a mock response for testing."""
        logger.info(f"Mock response generated for prompt: {prompt[:50]}...")
        
        # Return a mock response based on the prompt
        if "payment" in prompt.lower():
            return "Based on the available information, you can check your payment status by logging into your account portal or contacting customer service."
        elif "refund" in prompt.lower():
            return "Refunds are processed within 5-7 business days. Please contact support for specific refund requests."
        else:
            return "This is a mock response for testing purposes. In production, this would be generated by Mistral AI."


class MockResponseGenerator:
    """Mock response generator for testing."""
    
    def __init__(self):
        logger.info("Initializing mock response generator")
        self.mistral_generator = MockMistralResponseGenerator()
    
    def generate(self, prompt: str, style: str = "concise", **kwargs) -> str:
        """Generate a mock response."""
        return self.mistral_generator.generate_response(prompt, **kwargs) 