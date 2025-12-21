
import os
import sys
import argparse
import io
import time
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic._utils import save_audio

# Proven logic from debug_tts.py
def generate_audio(text, output_file, voice_id=None, lang_code=None, speed=1.0):
    print(f"DEBUG: Wrapper starting for text='{text}', output='{output_file}'", flush=True)
    
    # 1. API Key Hardcoded Fallback (Proven from Debug)
    api_key = os.environ.get('NEUPHONIC_API_KEY')
    if not api_key:
        api_key = "8999e98ca9370e30d6742fe614bbeb549bfb7132c8d808077ce81cdd6c7da4a1.3155bb0d-6e2f-46f2-af40-be217cd8b40d"
    
    # Verify Key
    print(f"DEBUG: Using API Key len={len(api_key)}", flush=True)
    client = Neuphonic(api_key=api_key)

    # 2. List Voices (Warmup - Proven)
    try:
        voices_resp = client.voices.list()
        print(f"DEBUG: Listed {len(voices_resp.data.get('voices', []))} voices", flush=True)
    except Exception as e:
        print(f"DEBUG: List voices warning: {e}", flush=True)

    # 3. Execution Logic
    try:
        sse = client.tts.SSEClient()
        
        # If speed is non-default, use config
        if abs(speed - 1.0) > 0.01:
             print(f"DEBUG: Using speed config {speed}", flush=True)
             config = TTSConfig(speed=float(speed))
             response = sse.send(text, tts_config=config)
        
        # If specific voice requested
        elif voice_id and voice_id != 'default':
             print(f"DEBUG: Using voice config {voice_id}", flush=True)
             # Ensure lang_code is present if voice is used
             l_code = lang_code if lang_code else 'en'
             config = TTSConfig(voice_id=voice_id, lang_code=l_code)
             response = sse.send(text, tts_config=config)
             
        else:
             print("DEBUG: Using DEFAULT (no config) - The Golden Path", flush=True)
             response = sse.send(text)

        # Save
        print("DEBUG: Saving audio...", flush=True)
        with open(output_file, 'wb') as f:
            save_audio(response, f)
        print("Success: Audio saved.", flush=True)

    except Exception as e:
        print(f"ERROR: Main attempt failed: {e}", flush=True)
        # Fallback to ABSOLUTE default
        try:
             print("DEBUG: Attempting absolute fallback...", flush=True)
             sse = client.tts.SSEClient()
             response = sse.send(text)
             with open(output_file, 'wb') as f:
                 save_audio(response, f)
             print("Success: Fallback audio saved.", flush=True)
        except Exception as fb_e:
             print(f"CRITICAL: Fallback failed: {fb_e}", flush=True)
             sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    generate_audio(args.text, args.output, args.voice, args.lang, args.speed)
