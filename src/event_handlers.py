"""
Event handlers for the voice bot.
Handles Daily room events and call state management.
"""

import asyncio
from loguru import logger
import aiohttp


class EventHandlers:
    """Manages all event handlers for the voice bot."""
    
    def __init__(self, transport, task, context, context_aggregator, 

                    llm_tools, firm_id, session_id, call_control_id,
                 task_cancelled, on_call_end_func, check_in_detector=None, intro_message=None):
        self.transport = transport
        self.task = task
        self.context = context
        self.context_aggregator = context_aggregator
        self.llm_tools = llm_tools
        self.firm_id = firm_id
        self.session_id = session_id
        self.call_control_id = call_control_id
        self.task_cancelled = task_cancelled
        self.on_call_end_func = on_call_end_func
        self.check_in_detector = check_in_detector
        self.intro_message = intro_message
        
    def setup_handlers(self):
        """Setup all event handlers."""
        self._setup_client_connected_handler()
        self._setup_participant_left_handler()
        self._setup_dialin_handlers()
    
    def _setup_client_connected_handler(self):
        """Setup handler for when client connects."""
        @self.transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            """Kick off the conversation on connected."""
            # Skip intro message if None or empty
            if self.intro_message:
                content1 = self.intro_message
                self.context.messages.append({"role": "system", "content": content1})
                context_frame = self.context_aggregator.user().get_context_frame()
                await self.task.queue_frames([context_frame])
    
    def _setup_participant_left_handler(self):
        """Setup handler for when participant leaves."""
        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            # logger.info(f"Participant left: {participant['id']}, reason: {reason}")
            
            # Gather transcript from context aggregator or pipeline
            # logger.info(f"Context messages: {self.context.messages}")
            def msg_to_text(msg):
                if 'content' in msg:
                    return f"{msg['role']}: {msg['content']}"
                elif 'tool_calls' in msg:
                    return f"{msg['role']}: [tool call: {msg['tool_calls']}]"
                else:
                    return f"{msg['role']}: [no content]"

            transcript = "\n".join(
                msg_to_text(msg)
                for msg in self.context.messages
                if msg['role'] in ('user', 'assistant')
            )
            if not transcript:
                transcript = "(transcript unavailable)"
            
            print("about to call the on call end jawn")
            
            # Send transcript before cancelling task
            try:
                await self.on_call_end_func(transcript, (self.session_id or ""), 
                                          self.firm_id, self.call_control_id)
            except Exception as e:
                logger.error(f"Error sending transcript: {e}")
            
            # Clean up check-in detector if it exists
            if self.check_in_detector:
                try:
                    await self.check_in_detector.cleanup()
                    logger.info("Check-in detector cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up check-in detector: {e}")
            
            # Cancel the pipeline task - this will automatically clean up the transport
            try:
                if not self.task_cancelled:
                    self.task_cancelled = True
                    await self.task.cancel()
                    logger.info("Pipeline task cancelled from participant left handler")
            except asyncio.CancelledError:
                logger.info("Task was already cancelled")
            except Exception as e:
                logger.error(f"Error cancelling task on participant left: {e}")
    
    def _setup_dialin_handlers(self):
        """Setup all dialin-related event handlers."""
        
        @self.transport.event_handler("on_dialin_connected")
        async def on_dialin_connected(transport, data):

            logger.info(f"Dial-in connected: {data}")
            # self.session_id = data.get("sessionId")  # Store the session ID
            # # Extract call_control_id from SIP headers if available
            # sip_headers = data.get("sipHeaders", {})
            # self.call_control_id = sip_headers.get("X-Telnyx-Call-Control-Id")
            # # self.call_control_id = call_control_id
            # logger.debug(f"Dial-in connected: {data}")
            # logger.info(f"Stored session ID: {self.session_id}")

        @self.transport.event_handler("on_dialin_stopped")
        async def on_dialin_stopped(transport, data):
            # if self.task_cancelled:
            #     logger.info("Task already cancelled, skipping dialin stopped handler")
            #     return

            # logger.debug(f"Dial-in stopped: {data}")
            logger.info("Dial-in stopped - waiting briefly then cancelling pipeline task")
            
            # Wait a moment to let the transfer complete cleanly
            await asyncio.sleep(0.5)
            
            try:
                if not self.task_cancelled:
                    self.task_cancelled = True
                    await self.task.cancel()
                    logger.info("Pipeline task cancelled from dialin stopped handler")
            except asyncio.CancelledError:
                logger.info("Task was already cancelled")
            except Exception as e:
                logger.error(f"Error cancelling task on dialin stopped: {e}")

        @self.transport.event_handler("on_dialin_error")
        async def on_dialin_error(transport, data):
            if self.task_cancelled:
                logger.info("Task already cancelled, skipping dialin error handler")
                return
                
            # logger.error(f"Dial-in error: {data}")
            # logger.info("Dial-in error - cancelling pipeline task")
            
            try:
                if not self.task_cancelled:
                    self.task_cancelled = True
                    await self.task.cancel()
                    logger.info("Pipeline task cancelled from dialin error handler")
            except asyncio.CancelledError:
                logger.info("Task was already cancelled")
            except Exception as e:
                logger.error(f"Error cancelling task on dialin error: {e}")

        @self.transport.event_handler("on_dialin_warning")
        async def on_dialin_warning(transport, data):
            logger.warning(f"Dial-in warning: {data}")
            # Don't cancel on warnings, just log them 