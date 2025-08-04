import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger
from pipecat.processors.aggregators.sentence import SentenceAggregator
from typing import List, Optional
from openai.types.chat import ChatCompletionMessageParam
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from src.LLMTools import LLMTools, tools
import aiohttp
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import Frame, BotStoppedSpeakingFrame
from src.check_in_detector import CheckInDetector


# Setup logging
load_dotenv(override=True)
logger.remove(0)
# logger.add(sys.stderr, level="INFO")
logger.add(sys.stderr, level="DEBUG")  # Change to DEBUG for maximum verbosity



class TTSCompletionHandler(FrameProcessor):
    """Frame processor that handles TTS completion events to execute queued hangups."""
    
    def __init__(self, llm_tools):
        super().__init__()
        self.llm_tools = llm_tools
        self.user_is_speaking = False
        self.sentence_aggregator = None  # Will be set after pipeline creation
        self.tts_service = None  # Will be set after pipeline creation
        
    def set_sentence_aggregator(self, sentence_aggregator):
        """Set reference to sentence aggregator for buffer clearing."""
        self.sentence_aggregator = sentence_aggregator
        
    def set_tts_service(self, tts_service):
        """Set reference to TTS service for buffer clearing."""
        self.tts_service = tts_service
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle TTS completion."""
        await super().process_frame(frame, direction)
        
        # Handle user interruptions for buffer clearing
        if frame.__class__.__name__ == 'UserStartedSpeakingFrame':
            self.user_is_speaking = True
            logger.debug("🧹 User interruption detected - clearing buffers")
            await self._clear_sentence_buffer()
            await self._clear_tts_buffer()
            
        elif frame.__class__.__name__ == 'UserStoppedSpeakingFrame':
            self.user_is_speaking = False
            logger.debug("🧹 User stopped speaking - clearing buffers")
            await self._clear_sentence_buffer()
            await self._clear_tts_buffer()
            
        elif frame.__class__.__name__ == 'BotStartedSpeakingFrame':
            logger.debug("🧹 Bot started speaking - clearing sentence buffer")
            await self._clear_sentence_buffer()
        
        # Check if this is a bot stopped speaking frame
        if isinstance(frame, BotStoppedSpeakingFrame):
            # logger.debug("✅ Bot stopped speaking - checking for queued operations")
            if self.llm_tools.hangup_queue:
                logger.info("Executing queued hangup after TTS completion")
                await self.llm_tools.execute_queued_hangup()
            if self.llm_tools.forward_queue:
                logger.info("Executing queued forward after TTS completion")
                await self.llm_tools.execute_queued_forward()
        
        # Always pass the frame along
        await self.push_frame(frame, direction)
        
    async def _clear_sentence_buffer(self):
        """Clear the sentence aggregator's buffer to prevent text concatenation."""
        try:
            if self.sentence_aggregator:
                # Try common buffer attribute names
                buffer_attrs = ['_buffer', '_current_sentence', '_accumulator', '_text', '_sentence_buffer']
                cleared = False
                cleared_content = []
                
                for attr in buffer_attrs:
                    if hasattr(self.sentence_aggregator, attr):
                        current_value = getattr(self.sentence_aggregator, attr)
                        if isinstance(current_value, str) and current_value:
                            setattr(self.sentence_aggregator, attr, "")
                            cleared_content.append(f"{attr}='{current_value}'")
                            cleared = True
                
                if cleared:
                    logger.info(f"✅ Cleared buffers: {', '.join(cleared_content)}")
                else:
                    logger.debug("⚠️ No buffer content found to clear")
            else:
                logger.debug("⚠️ No sentence aggregator reference available")
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not clear sentence buffer: {e}")
            
    async def _clear_tts_buffer(self):
        """Clear the TTS service's buffer to prevent audio concatenation."""
        try:
            if self.tts_service:
                # Try common TTS buffer attribute names
                buffer_attrs = ['_buffer', '_audio_buffer', '_text_buffer', '_queue', '_pending_text', '_current_text']
                cleared = False
                cleared_content = []
                
                for attr in buffer_attrs:
                    if hasattr(self.tts_service, attr):
                        current_value = getattr(self.tts_service, attr)
                        if isinstance(current_value, (str, list, dict)) and current_value:
                            # Clear based on type
                            if isinstance(current_value, str):
                                setattr(self.tts_service, attr, "")
                            elif isinstance(current_value, list):
                                setattr(self.tts_service, attr, [])
                            elif isinstance(current_value, dict):
                                setattr(self.tts_service, attr, {})
                            
                            cleared_content.append(f"{attr}='{str(current_value)[:50]}...'")
                            cleared = True
                
                if cleared:
                    logger.info(f"✅ Cleared TTS buffers: {', '.join(cleared_content)}")
                else:
                    logger.debug("⚠️ No TTS buffer content found to clear")
            else:
                logger.debug("⚠️ No TTS service reference available")
                        
        except Exception as e:
            logger.warning(f"⚠️ Could not clear TTS buffer: {e}")


# API keys
openai_api_key = os.getenv("OPENAI_API_KEY", "")
telnyx_api_key = os.getenv("TELNYX_API_KEY", "")
stallion_url = os.getenv("STALLION_URL", "")


async def on_call_end(transcript: str, session_id: str, firm_id: str, call_control_id: str):
    print("firm_id", firm_id)
    # Wait for 15 minutes
    payload = {
        "transcript": transcript,
        "call_control_id": call_control_id,
        "direction": "inbound",
        "firm_id": firm_id
    }
    print("TRYING TO PUT THE MF IN STALLION")
    print("stallion_url", stallion_url)
    print("payload were sending to stallion", payload)
    
    async with aiohttp.ClientSession() as session:
        await session.post(f"{stallion_url}/", json=payload)
    # Remove from queue after POST to stallion completes
    print("REMOVING FROM QUEUE")
    from main import remove_call_from_queue
    print("abt to remove call with call_control_id:", call_control_id)
    remove_call_from_queue(call_control_id)


async def run_bot(room_url: str, token: str, sip_uri: str, call_control_id: str, firm_id: str, caller_number: str, bot_ready_event: Optional[asyncio.Event] = None, prompt: Optional[str] = None, intro_message: str = "Introduce yourself and ask for their name.") -> None:
    """Run the voice bot with the given parameters.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        sip_uri: The Daily SIP URI for forwarding the call
        bot_ready_event: Optional event to signal when bot is ready
        prompt: custom system prompt for the bot
        intro_message: custom intro message when client connects
    """
    # logger.info(f"Starting bot with room: {room_url}")
    # logger.info(f"SIP endpoint: {sip_uri}")
    

    call_already_forwarded = False
    session_id = None
    task_cancelled = False  # Add this flag to track task cancellation
    # Initialize audio configuration
    from src.audio_config import AudioConfig
    audio_config = AudioConfig()
    
    # Check for required API keys
    if not audio_config.validate_api_keys():
        return
    
    # Create audio services
    vad_analyzer = audio_config.create_vad_analyzer()
    stt = audio_config.create_stt_service()
    tts = audio_config.create_tts_service()
    mixer = await audio_config.create_audio_mixer()
    
    # OpenAI LLM
    llm_params = OpenAILLMService.InputParams(frequency_penalty=0.0,
                                                  presence_penalty=0.0,
                                                  temperature=1.0,
                                                  extra={})
    llm = OpenAILLMService(api_key=openai_api_key,
                           model="gpt-4.1",
                           params=llm_params)

    try:
        transport = DailyTransport(
            room_url,
            token,
            "Phone Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=vad_analyzer,
                audio_in_sample_rate=audio_config.sample_rate,
                audio_out_sample_rate=audio_config.sample_rate,
                audio_in_channels=1,
                audio_out_channels=1,
                audio_out_mixer=mixer,
            ),
        )
    except Exception as e:
        print(f"Failed to initialize DailyTransport: {e}")
        return
    

    # telnyx_params = TelnyxFrameSerializer.InputParams(
    #     telnyx_sample_rate=sample_rate,
    #     sample_rate=sample_rate,
    #     inbound_encoding=inbound_encoding,
    #     outbound_encoding=outbound_encoding
    # )

    # # Telnyx frame serializer
    # serializer = TelnyxFrameSerializer(api_key=telnyx_api_key,
    #                                    call_control_id=call_control_id,
    #                                    inbound_encoding=inbound_encoding,
    #                                    outbound_encoding=outbound_encoding,
    #                                    params=telnyx_params)
    # Setup the Daily transport with echo cancellation and feedback prevention
    # Hook into the audio streaming start event
    if bot_ready_event:
        original_start_audio_in_streaming = transport.input().start_audio_in_streaming
        
        async def start_audio_in_streaming_wrapper():
            await original_start_audio_in_streaming()
            # logger.info("Bot is ready to receive audio - audio streaming started")
            bot_ready_event.set()
        
        transport.input().start_audio_in_streaming = start_audio_in_streaming_wrapper

   

    # Initialize LLM context with system prompt 
    # likely delete or sum cus this confusing
    default_prompt = (
        "You are a friendly phone assistant. Your responses will be read aloud, "
        "so keep them concise and conversational. Avoid special characters or "
        "formatting. Begin by greeting the caller and asking how you can help them today."
    )
    
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": prompt if prompt else default_prompt,
        },
    ]

    # Create LLMTools instance first (before pipeline)
    llm_tools = LLMTools(call_control_id, caller_number, pipeline_task=None)

    # Override LLM process_frame to conditionally skip processing
    # might wanna remove the hella logging buyt idc
    original_process_frame = llm.process_frame
    async def conditional_process_frame(frame, direction):
        if llm_tools.shouldLLM:
            return await original_process_frame(frame, direction)
        else:
            # logger.info("LLM processing skipped - call transferred")
            return None
    llm.process_frame = conditional_process_frame

    # LLM context with tools
    context = OpenAILLMContext(messages, tools=tools, tool_choice="auto")  # type: ignore
    context_aggregator = llm.create_context_aggregator(context)

    # Create TTS completion handler
    tts_completion_handler = TTSCompletionHandler(llm_tools)
    
    # Create sentence aggregator instance for buffer clearing
    sentence_aggregator = SentenceAggregator()
    
    # Connect sentence aggregator and TTS service to TTS handler for buffer clearing
    tts_completion_handler.set_sentence_aggregator(sentence_aggregator)
    tts_completion_handler.set_tts_service(tts)

    # Create check-in detector
    check_in_detector = CheckInDetector(check_in_timeout=6.0, check_in_message="Hey, are you still there?")
    
    # Set up check-in callback to send message through LLM
    async def send_check_in_message(message: str):
        """Send check-in message through the LLM pipeline."""
        logger.info(f"Sending check-in: {message}")
        # Add a system message to prompt the LLM for a check-in
        context.messages.append({
            "role": "system", 
            "content": "The user hasn't responded for a while. Send a brief, natural check-in message to see if they're still there. Keep it conversational and short."
        })
        # Create a context frame to trigger the LLM response
        context_frame = context_aggregator.user().get_context_frame()
        await task.queue_frames([context_frame])
    
    check_in_detector.set_check_in_callback(send_check_in_message)

    # Build the pipeline
    pipe_list = [
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        sentence_aggregator,  # Use the instance we created
        tts,
        tts_completion_handler,  # Add TTS completion handler after TTS
        check_in_detector,  # Add check-in detector after TTS completion
        transport.output(),
        context_aggregator.assistant()
    ]
    pipeline = Pipeline(pipe_list)
    pipeline_params = PipelineParams(audio_in_sample_rate=audio_config.sample_rate,
                                        audio_out_sample_rate=audio_config.sample_rate,
                                        allow_interruptions=True,
                                        interruption_strategies=[],
                                        enable_heartbeats=False,
                                        heartbeats_period_secs=1.0,
                                        enable_metrics=True,
                                        enable_usage_metrics=True,
                                        report_only_initial_ttfb=False,
                                        send_initial_empty_metrics=True)
    task = PipelineTask(pipeline=pipeline,
                        params=pipeline_params,
                        check_dangling_tasks=True,
                        enable_tracing=True,
                        enable_turn_tracking=True,
                        enable_watchdog_logging=False,
                        enable_watchdog_timers=False,
                        cancel_on_idle_timeout=True,
                        idle_timeout_secs=300.0)

    # Update LLMTools with the actual pipeline task
    llm_tools.pipeline_task = task
    
    # Register the functions for function calling
    llm.register_function("normalForwarding", llm_tools.normalForwarding)
    llm.register_function("hangupCall", llm_tools.hangupCall)

    # Setup event handlers
    from src.event_handlers import EventHandlers
    event_handlers = EventHandlers(
        transport=transport,
        task=task,
        context=context,
        context_aggregator=context_aggregator,
        llm_tools=llm_tools,
        firm_id=firm_id,
        session_id=session_id,
        call_control_id=call_control_id,
        task_cancelled=task_cancelled,
        on_call_end_func=on_call_end,
        check_in_detector=check_in_detector,  # Pass check-in detector for cleanup
        intro_message=intro_message or "Please introduce yourself to the user."
    )
    event_handlers.setup_handlers()

    # Run the pipeline
    runner = PipelineRunner()
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Pipeline task was cancelled")
        await task.cancel()
        raise
    except Exception as e:
        logger.error(f"Pipeline runner encountered an error: {e}")
        await task.cancel()
        raise
    finally:
        logger.info("Pipeline runner finished")


