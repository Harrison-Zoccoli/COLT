"""
Runs LiveKit Agent for intake conversations.
Adapted from originalbot_telnyx.py thing to use LiveKit's framework.
not a fan of this file tbh but if ur just testing its wtv
"""
import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, RoomInputOptions, AgentSession
from livekit.agents import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip
from livekit.agents import WorkerType, WorkerOptions, cli
from livekit.plugins import noise_cancellation, elevenlabs
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from src.prewarm_fnc import prewarm_fnc
from src.assistant import IntakeAssistant

# Load API keys n shih
_ = load_dotenv(override=True)

# Logging
logger = logging.getLogger("colt-livekit-agent")
logger.setLevel(logging.INFO)


# Async entrypoint function
async def entrypoint_fnc(ctx: JobContext) -> None:
    """
    Async function to define the entrypoint for agent.
    """
    # Connect the context
    await ctx.connect()

    # Get language from environment variable
    language_env = os.getenv("LANGUAGE", "en").lower()
    
    # LiveKit Cloud enhanced noise cancellation
    # - If self-hosting, omit this parameter
    # - For telephony applications, use `BVCTelephony` for best results
    noise_bvc = noise_cancellation.BVCTelephony()

    # Turn detection model - using multilingual model that supports both English and Turkish
    turn_detection = MultilingualModel(unlikely_threshold=None)

    # TTS - Using ElevenLabs with language-specific voice selection
    # Voice mapping for different languages
    voice_mapping = {
        "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel (English)
        "tr": "EXAVITQu4vr4xnSDxMaL"   # Bella (works well with Turkish)
    }
    
    voice_id = voice_mapping.get(language_env, "21m00Tcm4TlvDq8ikWAM")
    tts = elevenlabs.TTS(voice_id=voice_id, 
                        model="eleven_turbo_v2_5")

    # Room input options
    room_input_opts = RoomInputOptions(
        pre_connect_audio=True, 
        audio_enabled=True, 
        text_enabled=True,
        video_enabled=False, 
        audio_sample_rate=16000, 
        noise_cancellation=noise_bvc
    )

    # Agent session
    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"], 
        stt=ctx.proc.userdata["stt"], 
        turn_detection=turn_detection,
        llm=ctx.proc.userdata["llm"], 
        video_sampler=None, 
        allow_interruptions=True, 
        discard_audio_if_uninterruptible=True,
        min_interruption_duration=0.5, 
        min_interruption_words=0, 
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0, 
        user_away_timeout=15.0, 
        min_consecutive_speech_delay=0.0
    )

    # Language-specific instructions
    #will ,ake prettier for better prompt testing n shih mor emodular etc later on
    if language_env == "en":
        role_instructions = "answer all questions the user has."
        greet_instructions = "Greet user and help with whatever they ask."
    elif language_env == "tr":
        role_instructions = "Kullanıcının tüm sorularını yanıtla."
        greet_instructions = "Kullanıcıyı selamla ve ne isterse yardım et."
    else:
        print("defaultiung to eng prompt bc lang not supported check env var")
        role_instructions = "answer all questions the user has."
        greet_instructions = "Greet user and help with whatever they ask."

    # Agent
    agent = IntakeAssistant(
        tts=tts,
        instructions=role_instructions,
        firm_id="default"
    )

    # Start the session
    await session.start(room=ctx.room, room_input_options=room_input_opts, agent=agent)

    # Start the background noise player
    snd1 = AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8/8)
    snd2 = AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7/8)
    snd3 = AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8/3)
    background_audio = BackgroundAudioPlayer(
        ambient_sound=snd3,
        thinking_sound=[snd1, snd2]
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    # Start the conversation
    await session.generate_reply(instructions=greet_instructions)


if __name__ == "__main__":
    opts = WorkerOptions(
        prewarm_fnc=prewarm_fnc, 
        entrypoint_fnc=entrypoint_fnc,
        worker_type=WorkerType.ROOM, 
        agent_name="IntakeBot"
    )
    cli.run_app(opts=opts) 