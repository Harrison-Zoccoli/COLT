"""
Check-in detector for voice bot.
Handles sending check-in messages when user doesn't respond after bot speaks.
"""

import asyncio
import time
from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserStartedSpeakingFrame


class CheckInDetector(FrameProcessor):

    """Detects when user hasn't responded and sends check-in messages with exponential falloff."""
    
    def __init__(self, check_in_timeout=None, initial_timeout=6.0, falloff_multiplier=2.0, max_timeout=60.0, check_in_message="Hey, are you still there?"):
        super().__init__()
        # Support backward compatibility with old parameter name
        if check_in_timeout is not None:
            self.initial_timeout = check_in_timeout
        else:
            self.initial_timeout = initial_timeout
        self.falloff_multiplier = falloff_multiplier
        self.max_timeout = max_timeout
        self.check_in_message = check_in_message
        
        # State tracking
        self.last_bot_speak_time = None
        self.last_bot_stop_time = None
        self.check_in_sent = False
        self.check_in_task = None
        self.user_speaking = False  # Track if user is currently speaking

        self.current_timeout = initial_timeout  # Current timeout duration
        self.check_in_count = 0  # Number of check-ins sent in current silence period
        
        # Callback for sending check-in message
        self.on_check_in_callback = None
        
    def set_check_in_callback(self, callback):
        """Set the callback function to send check-in messages."""
        self.on_check_in_callback = callback

    def reset_timeout(self):
        """Reset the timeout to initial value when user interaction occurs."""
        self.current_timeout = self.initial_timeout
        self.check_in_count = 0
        logger.debug(f"Reset check-in timeout to {self.current_timeout}s")
        
    def calculate_next_timeout(self):
        """Calculate the next timeout using exponential falloff."""
        # Calculate exponential falloff: timeout = initial * multiplier^count
        next_timeout = self.initial_timeout * (self.falloff_multiplier ** self.check_in_count)
        
        # Cap at maximum timeout
        next_timeout = min(next_timeout, self.max_timeout)
        
        self.current_timeout = next_timeout
        logger.debug(f"Next check-in timeout: {self.current_timeout}s (attempt #{self.check_in_count + 1})")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and detect when to send check-in messages."""
        await super().process_frame(frame, direction)
        
        # Bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            self.last_bot_speak_time = time.time()
            self.check_in_sent = False
            
            # Cancel any existing check-in timer
            if self.check_in_task and not self.check_in_task.done():
                self.check_in_task.cancel()

                logger.debug("Bot started speaking - cancelled check-in timer")
                
        # Bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.last_bot_stop_time = time.time()
            
            # Start check-in timer only if:
            # 1. Bot was speaking
            # 2. We haven't sent a check-in yet
            # 3. User isn't currently speaking (extra safety check)
            if (self.last_bot_speak_time and 
                not self.check_in_sent and 
                not hasattr(self, 'user_speaking') or not self.user_speaking):
                

                # Calculate timeout for this check-in attempt
                self.calculate_next_timeout()
                
                logger.debug(f"Bot stopped speaking - starting {self.current_timeout}s check-in timer")
                self.check_in_task = asyncio.create_task(self._schedule_check_in())
            else:
                logger.debug("Bot stopped speaking - skipping check-in timer (user speaking or check-in already sent)")
                
        # User starts speaking
        elif isinstance(frame, UserStartedSpeakingFrame):
            # Mark user as speaking
            self.user_speaking = True
            
            # Cancel check-in timer if user starts speaking
            if self.check_in_task and not self.check_in_task.done():
                self.check_in_task.cancel()

                logger.debug("User started speaking - cancelled check-in timer")
            
            # Reset timeout and check-in state for next interaction
            self.reset_timeout()
            self.check_in_sent = False
            
        # User stops speaking (optional - for better state tracking)
        elif hasattr(frame, '__class__') and 'UserStoppedSpeakingFrame' in str(frame.__class__):
            # Mark user as not speaking
            self.user_speaking = False

            logger.debug("User stopped speaking")
            
        # Always pass the frame along
        await self.push_frame(frame, direction)
        
    async def _schedule_check_in(self):

        """Schedule a check-in message after the current timeout period."""
        try:
            await asyncio.sleep(self.current_timeout)
            
            # Double-check that we haven't been cancelled and user hasn't spoken
            # Also verify the task is still the current one (not a stale timer)
            if (not self.check_in_sent and 
                self.on_check_in_callback and 
                self.check_in_task and 
                not self.check_in_task.done()):
                

                self.check_in_count += 1
                logger.info(f"Sending check-in message #{self.check_in_count}: '{self.check_in_message}' (after {self.current_timeout}s)")
                self.check_in_sent = False  # Allow for subsequent check-ins
                
                await self.on_check_in_callback(self.check_in_message)
                
                # Schedule the next check-in with exponential falloff
                if self.check_in_count < 5:  # Limit to prevent infinite check-ins
                    self.calculate_next_timeout()
                    logger.debug(f"Scheduling next check-in in {self.current_timeout}s")
                    self.check_in_task = asyncio.create_task(self._schedule_check_in())
                else:
                    logger.info("Maximum check-in attempts reached - stopping check-ins")
                    
            else:
                logger.debug("Check-in cancelled - user spoke or timer was cancelled")
                
        except asyncio.CancelledError:
            logger.debug("Check-in timer was cancelled")
        except Exception as e:
            logger.error(f"Error in check-in timer: {e}")
            
    async def cleanup(self):
        """Clean up any running tasks."""
        if self.check_in_task and not self.check_in_task.done():
            self.check_in_task.cancel()
            try:
                await self.check_in_task
            except asyncio.CancelledError:
                pass 