"""
Custom Pipecat service that integrates the multi-agent legal intake system.
This service extends OpenAI's LLM service but routes requests through our agent system.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional, List, Dict, Any

from pipecat.frames.frames import (
    Frame, 
    LLMFullResponseEndFrame, 
    LLMFullResponseStartFrame, 
    LLMTextFrame, 
    TranscriptionFrame,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.frame_processor import FrameDirection

from .agent_system import main_system_streaming, set_debug_mode, initialize_system

class PipecatAgentService(OpenAILLMService):
    """
    Custom Pipecat LLM service that integrates the multi-agent legal intake system.
    
    This service extends OpenAI's LLM service but routes completion requests through
    our sophisticated agent system instead of directly to OpenAI.
    """
    
    def __init__(
        self,
        *,
        api_key: str = "dummy",  # Not used since we override completion
        model: str = "gpt-4",
        debug_mode: bool = False,
        buck_data_file: str = "src/sample_buck_data.json",
        **kwargs,
    ):
        """
        Initialize the Pipecat Agent Service.
        
        Args:
            api_key: Dummy API key (not used since we override completion)
            model: The model name to use for the agent system
            debug_mode: Whether to enable debug logging
            buck_data_file: Path to the buck data configuration file
            **kwargs: Additional arguments passed to the parent class
        """
        # Initialize parent with dummy API key since we override completion
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        # Initialize the agent system
        try:
            # Initialize the global agent system
            initialize_system()
            logging.info(f"✅ Agent system initialized with model: {model}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize agent system: {e}")
            raise
        
        # Set debug mode
        set_debug_mode(debug_mode)
        
        # Track configuration
        self._model_name = model
        self._buck_data_file = buck_data_file
        
        logging.info(f"🚀 PipecatAgentService initialized successfully")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the Pipecat pipeline."""
        
        if isinstance(frame, TranscriptionFrame):
            # Handle transcription directly from STT
            # Pipecat's turn detection ensures this is only called when user is done speaking
            transcription = frame.text.strip()
            
            if transcription:
                logging.info(f"🎤 Processing complete user message: {transcription}")
                
                # Process through agent system and stream response
                try:
                    await self.push_frame(LLMFullResponseStartFrame())
                    
                    # Get streaming response from agent system
                    chunks = main_system_streaming(transcription)
                    
                    for chunk in chunks:
                        if chunk and chunk.strip():
                            await self.push_frame(LLMTextFrame(chunk))
                            # Small delay to prevent overwhelming the pipeline
                            await asyncio.sleep(0.01)
                    
                    await self.push_frame(LLMFullResponseEndFrame())
                    
                except Exception as e:
                    logging.error(f"❌ Error processing transcription: {e}")
                    error_message = "I apologize, I encountered an error processing your request."
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(LLMTextFrame(error_message))
                    await self.push_frame(LLMFullResponseEndFrame())
            
            return
        
        # Pass through other frame types to parent
        await super().process_frame(frame, direction)

    async def _stream_chat_completions(
        self,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Override the chat completion streaming to use our agent system.
        This method is called when the service is used with context aggregation.
        
        Args:
            context: List of conversation messages
            **kwargs: Additional arguments (ignored)
            
        Yields:
            Text chunks from our agent system
        """
        try:
            # Extract the last user message
            user_input = ""
            for message in reversed(context):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        user_input = content.strip()
                        break
                    elif isinstance(content, list):
                        # Handle multi-modal content
                        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                        user_input = " ".join(text_parts).strip()
                        break
            
            if not user_input:
                logging.warning("⚠️ No user input found in context")
                yield "I didn't receive any input to process."
                return
            
            logging.info(f"🎤 Processing message through agent system: {user_input[:100]}...")
            
            # Get response from agent system
            chunks = main_system_streaming(user_input)
            
            # Stream the response
            for chunk in chunks:
                if chunk and chunk.strip():
                    yield chunk
                    # Add small delay to simulate realistic streaming
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logging.error(f"❌ Error in agent system streaming: {e}")
            yield "I apologize, I encountered an error processing your request." 