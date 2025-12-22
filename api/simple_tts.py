import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Using ElevenLabs TTS - Official SDK")

# Create temp directory for TTS files if it doesn't exist
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Define file paths in temp directory
INPUT_FILE = os.path.join(TEMP_DIR, 'tts_input.txt')
OUTPUT_FILE = os.path.join(TEMP_DIR, 'tts_output.wav')
ERROR_FILE = os.path.join(TEMP_DIR, 'tts_error.txt')

# Get API key from environment variable (SECURE - not hardcoded)
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

if not ELEVENLABS_API_KEY:
    print("WARNING: ELEVENLABS_API_KEY not found in environment variables")
    print("Will use gTTS fallback...")

# Voice mapping: Map frontend voice names to ElevenLabs voice IDs
VOICE_MAP = {
    'default': '21m00Tcm4TlvDq8ikWAM',      # Rachel - Female, American
    'rachel': '21m00Tcm4TlvDq8ikWAM',       # Rachel - Female, American
    'bella': 'EXAVITQu4vr4xnSDxMaL',        # Bella - Female, Soft
    'antoni': 'ErXwobaYiN019PkySvjV',       # Antoni - Male, Energetic
    'josh': 'TxGEqnHWrfWFTfGW9XjX',         # Josh - Male, American
    'charlotte': 'XB0fDUnXU5powFXDhCwa',    # Charlotte - Female, British
    'nicole': 'piTKgcLEGmPE4e6mEKli',       # Nicole - Female, Australian
    'adam': 'pNInz6obpgDQGcFmaJgB',         # Adam - Male, American
}

try:
    # 1. Read Input from File (JSON format for parameters)
    text_to_use = "Hello, this is a test"
    voice_id = VOICE_MAP['default']
    speed = 1.0
    
    if os.path.exists(INPUT_FILE):
        try:
            with open(INPUT_FILE, "r", encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    try:
                        config = json.loads(content)
                        text_to_use = config.get('text', text_to_use)
                        voice_param = config.get('voice', 'default').lower()
                        voice_id = VOICE_MAP.get(voice_param, VOICE_MAP['default'])
                        speed = config.get('speed', 1.0)
                    except json.JSONDecodeError as je:
                        print(f"Warning: JSON parse error, using text as-is: {je}")
                        text_to_use = content
        except FileNotFoundError:
            print(f"Warning: {INPUT_FILE} not found, using default text")
        except Exception as read_err:
            print(f"Warning: Failed to read {INPUT_FILE}: {read_err}")
    else:
        print(f"Info: {INPUT_FILE} not found, using default text")
            
    # Validate text input
    if not text_to_use or not text_to_use.strip():
        raise ValueError("Text input is empty or invalid")
    
    char_count = len(text_to_use)
    print(f"Synthesizing: {text_to_use[:50]}...")
    print(f"Voice ID: {voice_id}, Characters: {char_count}")

    # Helper function for gTTS fallback
    def fallback_to_gtts():
        try:
            from gtts import gTTS
            print("Using gTTS fallback...")
            slow = speed < 0.9
            tts = gTTS(text=text_to_use, lang='en', slow=slow)
            temp_mp3 = os.path.join(TEMP_DIR, 'temp_tts.mp3')
            tts.save(temp_mp3)
            
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(temp_mp3)
                audio.export(OUTPUT_FILE, format="wav")
                os.remove(temp_mp3)
            except ImportError:
                os.rename(temp_mp3, OUTPUT_FILE)
            
            file_size = os.path.getsize(OUTPUT_FILE)
            print(f"Fallback successful: gTTS ({file_size} bytes)")
            return True
        except Exception as fallback_err:
            print(f"gTTS fallback failed: {fallback_err}")
            return False

    # 2. Try ElevenLabs if API key is available
    if ELEVENLABS_API_KEY:
        print("Generating speech with ElevenLabs API...")
        
        # Retry logic for network errors
        max_retries = 3
        retry_delay = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                # ElevenLabs API endpoint
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY
                }
                
                # Adjust stability based on speed
                stability = 0.5 if speed == 1.0 else (0.7 if speed < 1.0 else 0.3)
                
                data = {
                    "text": text_to_use,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": 0.75
                    }
                }
                
                print(f"Request attempt {attempt + 1}/{max_retries}...")
                response = requests.post(url, json=data, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    # Success!
                    print(f"Saving to {OUTPUT_FILE}...")
                    with open(OUTPUT_FILE, 'wb') as f:
                        f.write(response.content)
                    
                    file_size = len(response.content)
                    print(f"SUCCESS! Generated {file_size} bytes ({char_count} characters used)")
                    success = True
                    break
                    
                elif response.status_code == 401:
                    print("Authentication failed: Invalid API key")
                    fallback_to_gtts()
                    success = True
                    break
                
                elif response.status_code == 403:
                    print("Access forbidden: API key lacks permissions")
                    fallback_to_gtts()
                    success = True
                    break
                
                elif response.status_code == 429:
                    print("Rate limit exceeded")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("Rate limit persists, using gTTS")
                        fallback_to_gtts()
                        success = True
                        break
                
                elif response.status_code == 422:
                    print("Invalid input parameters")
                    raise Exception("Invalid input. Please check text and voice settings.")
                
                elif response.status_code >= 500:
                    print(f"Server error: {response.status_code}")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("Server error persists, using gTTS")
                        fallback_to_gtts()
                        success = True
                        break
                
                else:
                    print(f"Unexpected error: {response.status_code}")
                    fallback_to_gtts()
                    success = True
                    break
                    
            except requests.exceptions.Timeout:
                print(f"Request timed out (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Timeout persists, using gTTS")
                    fallback_to_gtts()
                    success = True
                    break
                    
            except requests.exceptions.ConnectionError:
                print(f"Connection failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print("Connection failed, using gTTS")
                    fallback_to_gtts()
                    success = True
                    break
        
        if not success:
            raise Exception("Failed to generate speech after all retries")
    else:
        # No API key, use gTTS directly
        if not fallback_to_gtts():
            raise Exception("No ElevenLabs API key and gTTS fallback failed")
    
    # Verify file was created and has content
    if not os.path.exists(OUTPUT_FILE):
        raise Exception(f"Output file {OUTPUT_FILE} was not created")
    
    file_size = os.path.getsize(OUTPUT_FILE)
    if file_size == 0:
        raise Exception("Generated audio file is empty")

except ValueError as ve:
    print(f"Input Error: {ve}")
    try:
        with open(ERROR_FILE, "w") as f:
            f.write(f"Input validation failed: {ve}")
    except:
        pass
    sys.exit(1)

except Exception as main_e:
    print(f"Critical Error: {main_e}")
    import traceback
    traceback.print_exc()
    
    try:
        with open(ERROR_FILE, "w") as f:
            f.write(f"Error: {main_e}\n\nTraceback:\n")
            traceback.print_exc(file=f)
    except:
        pass
    
    sys.exit(1)


# Create temp directory for TTS files if it doesn't exist
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)
