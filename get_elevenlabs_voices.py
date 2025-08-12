#!/usr/bin/env python3
"""
Helper script to list available ElevenLabs voices.
Run this to get valid voice IDs for your account.
"""

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

def get_elevenlabs_voices():
    """Get list of available voices from ElevenLabs API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found in environment variables")
        print("Please set your ElevenLabs API key in your .env file:")
        print("ELEVENLABS_API_KEY=your_api_key_here")
        return
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
        response.raise_for_status()
        
        voices = response.json()["voices"]
        
        print(f"✅ Found {len(voices)} voices in your ElevenLabs account:")
        print("-" * 80)
        
        for voice in voices:
            print(f"Name: {voice['name']}")
            print(f"Voice ID: {voice['voice_id']}")
            print(f"Category: {voice.get('category', 'N/A')}")
            print(f"Description: {voice.get('description', 'N/A')}")
            print("-" * 40)
            
        # Show default voices that should work
        print("\n🔧 Default voices you can use without creating custom ones:")
        default_voices = [
            ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
            ("Domi", "AZnzlk1XvdvUeBnXmlld"),
            ("Bella", "EXAVITQu4vr4xnSDxMaL"),
            ("Antoni", "ErXwobaYiN019PkySvjV"),
            ("Elli", "MF3mGyEYCl7XYWbV9V6O"),
            ("Josh", "TxGEqnHWrfWFTfGW9XjX"),
            ("Arnold", "VR6AewLTigWG4xSOukaG"),
            ("Adam", "pNInz6obpgDQGcFmaJgB"),
            ("Sam", "yoZ06aMxZJJ28mfd3POQ"),
        ]
        
        for name, voice_id in default_voices:
            print(f"{name}: {voice_id}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to ElevenLabs API: {e}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    get_elevenlabs_voices() 