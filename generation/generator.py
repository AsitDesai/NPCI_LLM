"""
Response Generator using Mistral AI

This module handles response generation using Mistral AI as the base model.
It includes the main ResponseGenerator class and MistralResponseGenerator implementation.
"""

import os
import requests
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
    using the configured server model endpoint.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        logger.info("Initializing response generator")
        
        # Try to initialize server generator first
        try:
            self.server_generator = ServerResponseGenerator()
            self.generator_type = "server"
            logger.info("✅ Using server model endpoint for generation")
        except Exception as e:
            logger.error(f"❌ Failed to initialize server generator: {e}")
            # Fallback to Mistral if server fails
            try:
                self.mistral_generator = MistralResponseGenerator()
                self.generator_type = "mistral"
                logger.info("⚠️  Falling back to Mistral AI for generation")
            except Exception as e2:
                logger.error(f"❌ Failed to initialize Mistral generator: {e2}")
                # Final fallback to mock generator
                self.mock_generator = MockResponseGenerator()
                self.generator_type = "mock"
                logger.info("⚠️  Using mock generator as final fallback")
    
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
            
            # Use the appropriate generator based on what was initialized
            if hasattr(self, 'server_generator') and self.generator_type == "server":
                return self.server_generator.generate_response(
                    prompt, 
                    temperature=temperature
                )
            elif hasattr(self, 'mistral_generator') and self.generator_type == "mistral":
                return self.mistral_generator.generate_response(
                    prompt, 
                    temperature=temperature
                )
            else:
                return self.mock_generator.generate(
                    prompt, 
                    style=style
                )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


class ServerResponseGenerator:
    """
    Server endpoint response generator implementation.
    
    This class handles interaction with the configured server endpoint
    for generating responses.
    """
    
    def __init__(self):
        """Initialize the server generator."""
        logger.info("Initializing server response generator")
        
        # Get server endpoint from environment
        self.server_endpoint = os.getenv("SERVER_MODEL_ENDPOINT")
        self.api_key = os.getenv("SERVER_MODEL_API_KEY")
        
        if not self.server_endpoint:
            raise ValueError("SERVER_MODEL_ENDPOINT environment variable is required")
        
        logger.info(f"Server endpoint configured: {self.server_endpoint}")
    
    def generate_response(self, 
                        prompt: str, 
                        max_tokens: Optional[int] = None,
                        temperature: float = 0.7,
                        stream: bool = False) -> str:
        """
        Generate a response from the server endpoint.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (None for default)
            temperature: Generation temperature (0.0 to 1.0)
            stream: Whether to use streaming (not supported by server endpoint)
            
        Returns:
            Generated response string
        """
        try:
            # Prepare the request payload
            payload = {
                "model": "NPCI_Greviance",  # Using the actual model from the server endpoint
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            # Set headers
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "your_server_model_api_key_here":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make the request to the chat completions endpoint
            chat_endpoint = f"{self.server_endpoint}/v1/chat/completions"
            logger.info(f"Sending request to chat endpoint: {chat_endpoint}")
            response = requests.post(
                chat_endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response content (handle different response formats)
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                elif "response" in result:
                    content = result["response"]
                elif "content" in result:
                    content = result["content"]
                else:
                    content = str(result)
                
                logger.info(f"Generated response with {len(content)} characters")
                return content
            else:
                logger.error(f"Server endpoint error: {response.status_code} - {response.text}")
                raise Exception(f"Server endpoint returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error generating server response: {e}")
            raise


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