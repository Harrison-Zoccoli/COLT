"""
This has the prewarm function for the LiveKit Assistant.
This pre-loads the various external plugins (STT, TTS, LLM, etc.).
important to note that i fw this heavy since we can initilize everything before ever picking up the phone
"""
import os
from livekit.agents import JobProcess
from livekit.plugins import silero, deepgram, openai

__all__ = ["prewarm_fnc"]


# Prewarm function
def prewarm_fnc(proc: JobProcess) -> None:
    """
    Non-async prep before entrypoint.
    This initializes the component models (STT, LLM, TTS, etc.).
    """
    # Get language from environment variable
    language_env = os.getenv("LANGUAGE", "en").lower()
    
    # Map language codes to Deepgram language codes
    language_mapping = {
        "en": "en-US",
        "tr": "tr-TR"
    }
    
    # Default to English if language not supported
    deepgram_language = language_mapping.get(language_env, "en-US")
    
    # Silero VAD
    #likley will reconfiger these settings to better allign with original silero pipecat settings
    proc.userdata["vad"] = silero.VAD.load(sample_rate=16000,
                                           min_speech_duration=0.05,
                                           min_silence_duration=0.55,
                                           prefix_padding_duration=0.5,
                                           max_buffered_speech=60.0,
                                           activation_threshold=0.5,
                                           force_cpu=True)

    # STT - Deepgram
    # Use different models based on language support
    if deepgram_language == "en-US":
        model = "nova-3"
    else:
        # For non-English languages, use a more compatible model
        model = "nova-2"
    
    stt_params = {
        "model": model,
        "language": deepgram_language,
        "detect_language": False,
        "sample_rate": 16000,
        "endpointing_ms": 25,
        "interim_results": True,
        "punctuate": True,
        "smart_format": True,
        "numerals": False,
        "no_delay": True,
        "filler_words": True,
        "profanity_filter": False,
        "mip_opt_out": True
    }

    # Only add keyterms for English
    if deepgram_language == "en-US":
        stt_params["keyterms"] = []

    proc.userdata["stt"] = deepgram.STT(**stt_params)

    # LLM - OpenAI
    # nice and easy to swap aroudn models here for testing yw cutie pie <3 <3 <3 <3 <3 <3 <3
    # lowkey thinking 4.1 or 5. 5 seems to be fast and good with context especialyt with ur large ahh prompt idk abt pricing tho
    proc.userdata["llm"] = openai.LLM(model="gpt-4o-mini",
                                      client=None) 