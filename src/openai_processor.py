"""
OpenAI Model Processor

This module provides a simple function to interact with OpenAI models.
"""

from openai import OpenAI
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import os
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelParams:
    """Data class for model parameters following PEP guidelines."""
    
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# Global OpenAI client instance
_openai_client = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def process_openai_request(
    prompt: str,
    system_prompt: str,
    model_name: str,
    params: Optional[ModelParams] = None
) -> Dict[str, Any]:
    """
    Process a single OpenAI request.
    
    Args:
        prompt (str): User prompt to send to the model
        system_prompt (str): System prompt defining model behavior
        model_name (str): OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        params (Optional[ModelParams]): Model parameters, defaults to ModelParams()
    
    Returns:
        Dict[str, Any]: Response from OpenAI API containing content and metadata
        
    Raises:
        Exception: If API call fails
    """
    if params is None:
        params = ModelParams()
    
    try:
        # Use singleton OpenAI client (efficient!)
        client = get_openai_client()
        
        # Prepare messages for OpenAI API
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": model_name,
            "usage": response.usage,
            "finish_reason": response.choices[0].finish_reason
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise


def process_openai_request_streaming(
    prompt: str,
    system_prompt: str,
    model_name: str,
    params: Optional[ModelParams] = None
):
    """
    Process OpenAI request with real-time streaming output.
    
    Args:
        prompt (str): User prompt to send to the model
        system_prompt (str): System prompt defining model behavior
        model_name (str): OpenAI model name
        params (Optional[ModelParams]): Model parameters
    
    Yields:
        str: Individual characters/tokens as they arrive in real-time
        
    Returns:
        str: Complete response for transcript
    """
    if params is None:
        params = ModelParams()
    
    try:
        # Use singleton OpenAI client
        client = get_openai_client()
        
        # Prepare messages for OpenAI API
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        
        # Make streaming API call
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            stream=True
        )
        
        full_response = ""
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # Yield each character/token immediately (real-time streaming)
                yield content
        
        # Return complete response for transcript
        return full_response
        
    except Exception as e:
        logger.error(f"Error in streaming request: {e}")
        raise


def setup_openai_client() -> None:
    """
    Setup OpenAI client using environment variable.
    Requires OPENAI_API_KEY in .env file or environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Set environment variable for OpenAI client
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OpenAI client configured successfully")


# Example usage function
def example_usage() -> None:
    """Example demonstrating how to use the OpenAI processor function."""
    
    # Setup (uses OPENAI_API_KEY from .env file)
    setup_openai_client()
    
    # Define parameters
    params = ModelParams(temperature=0.7, max_tokens=500)
    
    # Single request
    try:
        result = process_openai_request(
            prompt="What is machine learning?",
            system_prompt="You are a helpful AI assistant.",
            model_name="gpt-3.5-turbo",
            params=params
        )
        print(f"Response: {result['content']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage() 