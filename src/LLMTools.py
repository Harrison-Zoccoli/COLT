"""
LLM Tools for the Telnyx bot - handles function calling capabilities
"""
import os
import aiohttp
from loguru import logger
from typing import Dict, Any


class LLMTools:
    def __init__(self, call_control_id: str, caller_number: str, pipeline_task=None):
        print(f"LLMTools initialized with call_control_id: {call_control_id} and caller_number: {caller_number}")
        self.call_control_id = call_control_id
        self.caller_number = caller_number
        self.pipeline_task = pipeline_task  # For terminating the AI pipeline
        self.telnyx_api_key = os.getenv("TELNYX_API_KEY", "")
        self.transfer_completed = False  # Flag to track if transfer was completed
        self.shouldLLM = True  # Flag to control if LLM should process
        self.hangup_queue = []  # Queue for pending hangup operations
        self.forward_queue = []  # Queue for pending forward operations
    
    async def normalForwarding(self, params) -> str:
        """
        Forward the call to a lawyer using Telnyx Call Control API.
        This will create a new call leg to the lawyer and bridge it to the original call.
        Args:
            params: FunctionCallParams object from PipeCat, expects params.arguments['forward_to_number']
        Returns:
            str: Success or error message
        """
        forward_to_number = params.arguments["forward_to_number"]
        print(f"normalForwarding called with caller_number: {self.caller_number}, forwarding to: {forward_to_number}")
        
        # For LiveKit console mode, just log the forward operation idc abt it for now
        if self.call_control_id.startswith("livekit_"):
            logger.info(f"LiveKit mode: Would forward to {forward_to_number}")
            return f"Forward queued - would connect to lawyer at {forward_to_number}"
        
        # Original Telnyx logic for phone calls
        if not self.telnyx_api_key:
            logger.error("TELNYX_API_KEY not found in environment variables")
            return "Error: Telnyx API key not configured"
        
        # Queue the forward operation instead of executing immediately
        self.forward_queue.append(lambda: self._execute_forward(forward_to_number))
        logger.info(f"Forward queued. Queue length: {len(self.forward_queue)}")
        return "Forward queued - will execute after current speech finishes"

    async def hangupCall(self, params) -> str:
        """
        Hang up the current call using Telnyx Call Control API.
        WARNING: This will immediately end the call. Only use this when the user explicitly requests to end the call
        or when the conversation is clearly finished and the user wants to hang up.
        Args:
            params: FunctionCallParams object from PipeCat
        Returns:
            str: Success or error message
        """
        print(f"hangupCall called for call_control_id: {self.call_control_id}")
        
        # For LiveKit console mode, just log the hangup operation idc abt it for now
        if self.call_control_id.startswith("livekit_"):
            logger.info("LiveKit mode: Would hang up call")
            return "Hangup queued - would end the conversation"
        
        # Original Telnyx logic for phone calls
        if not self.telnyx_api_key:
            logger.error("TELNYX_API_KEY not found in environment variables")
            return "Error: Telnyx API key not configured"
        
        # Queue the hangup operation instead of executing immediately
        if params.arguments.get("bypass_goodbye", True):
            await self._execute_hangup()
            return "Hangup Triggered"
        else:
            self.hangup_queue.append(self._execute_hangup)
            logger.info(f"Hangup queued. Queue length: {len(self.hangup_queue)}")
            return "Hangup queued - will execute after current speech finishes"
    
    async def _execute_hangup(self):
        """
        Execute the actual hangup operation.
        """
        print(f"Executing hangup for call_control_id: {self.call_control_id}")
        
        # Telnyx Call Control API endpoint for hanging up calls
        url = f"https://api.telnyx.com/v2/calls/{self.call_control_id}/actions/hangup"
        
        headers = {
            "Authorization": f"Bearer {self.telnyx_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as response:
                    response_text = await response.text()
                    print(f"Hangup API response: {response.status} - {response_text}")
                    if response.status in (200, 202):  # Accept both 200 and 202 as success
                        logger.info(f"Call {self.call_control_id} successfully hung up")
                        # Set flag to indicate call ended - AI will stop processing audio
                        self.shouldLLM = False  # Disable LLM processing
                        logger.info("Call hung up - AI will stop processing audio")
                    else:
                        logger.error(f"Failed to hang up call {self.call_control_id}. Status: {response.status}, Response: {response_text}")
        except Exception as e:
            logger.error(f"Exception occurred while hanging up call {self.call_control_id}: {str(e)}")
    
    async def execute_queued_hangup(self):
        """
        Execute the queued hangup operation after TTS has finished.
        """
        logger.info(f"execute_queued_hangup called. Queue length: {len(self.hangup_queue)}")
        if not self.hangup_queue:
            logger.info("No hangup in queue")
            return
            
        hangup_func = self.hangup_queue.pop(0)
        logger.info("Executing queued hangup function")
        await hangup_func()
    
    async def _execute_forward(self, forward_to_number: str):
        """
        Execute the actual forward operation.
        """
        print(f"Executing forward for call_control_id: {self.call_control_id} to {forward_to_number}")
        
        # Telnyx Call Control API endpoint for transferring calls
        url = f"https://api.telnyx.com/v2/calls/{self.call_control_id}/actions/transfer"
        
        headers = {
            "Authorization": f"Bearer {self.telnyx_api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload for call transfer (with timeout and time limit)
        payload = {
            "to": forward_to_number,
            "from": self.caller_number,
            "timeout_secs": 30,
            "time_limit_secs": 3600
        }
        print(f"Payload: {payload}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    print(f"Transfer API response: {response.status} - {response_text}")
                    if response.status in (200, 202):  # Accept both 200 and 202 as success
                        logger.info(f"Call {self.call_control_id} successfully forwarded to {forward_to_number}")
                        # Set flag to indicate transfer completed - AI will stop processing audio
                        self.transfer_completed = True
                        self.shouldLLM = False  # Disable LLM processing
                        logger.info("Transfer completed - AI will stop processing audio but connection remains active")
                    else:
                        logger.error(f"Failed to forward call {self.call_control_id}. Status: {response.status}, Response: {response_text}")
        except Exception as e:
            logger.error(f"Exception occurred while forwarding call {self.call_control_id}: {str(e)}")
    
    async def execute_queued_forward(self):
        """
        Execute the queued forward operation after TTS has finished.
        """
        logger.info(f"execute_queued_forward called. Queue length: {len(self.forward_queue)}")
        if not self.forward_queue:
            logger.info("No forward in queue")
            return
            
        forward_func = self.forward_queue.pop(0)
        logger.info("Executing queued forward function")
        await forward_func()


# Tool definitions for OpenAI function calling
# for now not n eeded cus ya
tools = [
    {
        "type": "function",
        "function": {
            "name": "normalForwarding",
            "description": "Forward the current call to a lawyer. Use this when the user requests to speak with a lawyer or when they need assistance that requires human intervention. IMPORTANT: Only call this function AFTER you have already spoken a message to the user indicating that you will forward them (e.g., 'Alright, let me get you connected to our car accident attorney right now' or 'Perfect, I'll forward you to the right attorney now'). Never call this function without first generating a spoken message about the forward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_to_number": {
                        "type": "string",
                        "description": "The phone number to forward the call to (include country code, e.g., +1234567890)"
                    }
                },
                "required": ["forward_to_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hangupCall",
            "description": (
                "Hang up the current call. IMPORTANT: Only call this function AFTER the conversation has ended and the caller has no intention of continuing the conversation. "
                "Never call this function before speaking a goodbye message. This function should not be triggered when a caller has told you to wait for them. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bypass_goodbye": {
                        "type": "boolean",
                        "description": "DO NOT SET THIS TO TRUE. This is only for emergency situations when the user has not been responding at all for several messages and has not indicated to wait or that they'll be back."
                    }
                },
                "required": []
            }
        }
    }
] 