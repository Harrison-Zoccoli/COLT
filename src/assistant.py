"""
LiveKit general assistant.
Handles the core conversation logic and integrates with existing tools.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from livekit.agents import Agent
from livekit.plugins import elevenlabs
from src.LLMTools import LLMTools, tools

logger = logging.getLogger(__name__)


class IntakeAssistant(Agent):
    """
    LiveKit Agent for intake conversations.
    Integrates existing intake logic with LiveKit's framework.
    """
    
    def __init__(self, tts: elevenlabs.TTS, instructions: str, firm_id: str = "default"):
        super().__init__(
            instructions=instructions,
            tts=tts
        )
        self.firm_id = firm_id #ignore firm stuff it will come later
        self.transcript = [] #initilizing empty trancript to handle calls transcription shih
        
    async def on_enter(self) -> None:
        """Called when the agent session starts."""
        logger.info("BotBoy session started")
        
        # Initialize LLM tools for the session
        # call control id is placehiolder for now will get repl;ac ed latrer wityh the actual calls info
        self.llm_tools = LLMTools(
            call_control_id="livekit_session",
            caller_number="console_user",
            pipeline_task=None
        )
        
    async def on_exit(self) -> None:
        """Called when the agent session ends."""
        logger.info(" intake assistant session ended")
        
        # Send transcript to Stallion (if configured)
        await self._send_transcript_to_stallion()
        
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Called when the user has finished speaking, and the LLM is about to respond."""
        logger.info(f"User turn completed: {new_message.content}")
        
        # Add to transcript
        self.transcript.append({"role": "user", "content": new_message.content})
        
        # Process any function calls that might be in the response
        # This will be handled by the LLM node automatically
        
    async def _send_transcript_to_stallion(self) -> None:
        """Send transcript to Stallion system."""
        if not self.transcript:
            return
            
        # Format transcript
        transcript_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.transcript 
            if msg['role'] in ('user', 'assistant')
        ])
        
        # Send to Stallion (if configured) placehiolder for post call later on
        stallion_url = "http://localhost:8000"  # Configure as needed
        try:
            import aiohttp
            payload = {
                "transcript": transcript_text,
                "call_control_id": "livekit_session",
                "direction": "console",
                "firm_id": self.firm_id
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(f"{stallion_url}/", json=payload)
                logger.info("Transcript sent to Stallion")
                
        except Exception as e:
            logger.error(f"Failed to send transcript to Stallion: {e}") 