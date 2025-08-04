"""
Audio configuration helper for the voice bot.
Handles VAD, STT, TTS, and audio mixer setup.
"""

import os
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions, Language
from pipecat.services.rime.tts import RimeTTSService
from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer


class AudioConfig:
    """Handles all audio service configuration and initialization."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.inbound_encoding = "linear16"
        self.outbound_encoding = "linear16"
        
        # API keys
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self.rime_api_key = os.getenv("RIME_API_KEY", "")
        
    def create_vad_analyzer(self):
        """Create and configure the VAD analyzer."""
        #old params:
        # vad_params = VADParams(confidence=0.7,
    #                        start_secs=0.2,
    #                        stop_secs=0.8,
    #                        min_volume=0.6)
        vad_params = VADParams(confidence=0.8,
                               start_secs=0.4,
                               stop_secs=0.8,
                               min_volume=0.7)
        return SileroVADAnalyzer(sample_rate=self.sample_rate,
                                 params=vad_params)
    
    def create_stt_service(self):
        """Create and configure the Deepgram STT service."""
        live_options = LiveOptions(
            encoding='linear16',
            channels=1,
            sample_rate=self.sample_rate,
            language="en-US",
            model="nova-3",
            interim_results=True,
            smart_format=True,
            punctuate=True,
            profanity_filter=False,
            vad_events=False,
            endpointing=0,     # Wait longer before processing speech end
        )
        return DeepgramSTTService(api_key=self.deepgram_api_key,
                                  base_url="ws://localhost:8081" if os.getenv("ON_PREM", "False").lower() == "true" else "",
                                  live_options=live_options)
    
    def create_tts_service(self):
        """Create and configure the Rime TTS service."""
        rime_params = RimeTTSService.InputParams(language=Language.EN,
                                                 speed_alpha=1.0,
                                                 reduce_latency=False,
                                                 pause_between_brackets=True,
                                                 phonemize_between_brackets=False,
                                                 audioFormat="pcm",
                                                 sample_rate=self.sample_rate)
        try:
            return RimeTTSService(api_key=self.rime_api_key,
                                 url="ws://localhost:8001/" if os.getenv("ON_PREM", "False").lower() == "true" else "wss://users.rime.ai/ws2",
                                 model="mistv2",
                                 voice_id="marissa",
                                 params=rime_params)
        except Exception as e:
            logger.error(f"Failed to initialize RimeTTSService: {e}")
            raise
    
    async def create_audio_mixer(self):
        """Create and configure the background audio mixer."""
        Office_Noise_Path = "assets/audio/Office_Noise.wav"
        Static_Noise_Path = "assets/audio/Static_Noise.wav"

        mixer = SoundfileMixer(
            sound_files={
                "office": Office_Noise_Path,
                "static": Static_Noise_Path
            },
            default_sound="office",
            volume=0.001,
            mixing=True,
            loop=True
        )
        await mixer.start(self.sample_rate)
        return mixer
    
    def validate_api_keys(self):
        """Validate that required API keys are present."""
        if not self.deepgram_api_key:
            logger.error("Missing Deepgram API key!")
            return False
        if not self.rime_api_key:
            logger.error("Missing Rime API key!")
            return False
        return True 
