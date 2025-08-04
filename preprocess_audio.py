import soundfile as sf
import numpy as np
import os
"""
pls dont delete, move to a helper file if needed
needed to change audio level/volume of the backround noiuse

we likely wanna keep static and delete office noise
maybe down line add a large set of possible backround noises
"""
# List your files and desired volume multipliers here
AUDIO_CONFIG = [
    {
        "input": "assets/audio/Office_Noise.wav",
        "output": "assets/audio/Office_Noise_quiet.wav",
        "volume": 1.0, 
    },
    {
        "input": "assets/audio/Static_Noise.wav",
        "output": "assets/audio/Static_Noise_loud.wav",
        "volume": 0.75,
    },
    # Add more as needed
]

def adjust_volume(input_path, output_path, volume):
    data, samplerate = sf.read(input_path)
    # Apply volume (clip to avoid overflow)
    data = np.clip(data * volume, -1.0, 1.0)
    sf.write(output_path, data, samplerate)
    print(f"Saved {output_path} with volume {volume}")

if __name__ == "__main__":
    for cfg in AUDIO_CONFIG:
        if not os.path.exists(cfg["input"]):
            print(f"File not found: {cfg['input']}")
            continue
        adjust_volume(cfg["input"], cfg["output"], cfg["volume"]) 