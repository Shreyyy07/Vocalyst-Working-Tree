import os
import sys
import json
import time
from gtts import gTTS
import tempfile

print("Using gTTS (Google Text-to-Speech) - Free & Unlimited")

# Create temp directory for TTS files if it doesn't exist
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

# Define file paths in temp directory
INPUT_FILE = os.path.join(TEMP_DIR, 'tts_input.txt')
OUTPUT_FILE = os.path.join(TEMP_DIR, 'tts_output.wav')
ERROR_FILE = os.path.join(TEMP_DIR, 'tts_error.txt')
TEMP_MP3 = os.path.join(TEMP_DIR, 'temp_tts.mp3')

try:
    # 1. Read Input from File (JSON format for parameters)
    text_to_use = "Hello, this is a test"
    voice_name = "default"  # gTTS doesn't have multiple voices
    speed = 1.0  # gTTS has slow parameter
    
    if os.path.exists(INPUT_FILE):
        try:
            with open(INPUT_FILE, "r", encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Try to parse as JSON first
                    try:
                        config = json.loads(content)
                        text_to_use = config.get('text', text_to_use)
                        voice_name = config.get('voice', 'default')
                        speed = config.get('speed', 1.0)
                    except json.JSONDecodeError as je:
                        print(f"Warning: JSON parse error, using text as-is: {je}")
                        # Fallback to plain text
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
    
    print(f"Synthesizing: {text_to_use[:50]}...")
    print(f"Speed: {speed}")

    # 2. Generate speech using gTTS
    print("Generating speech with gTTS...")
    
    # gTTS parameters
    slow = speed < 0.9  # Use slow mode if speed is less than 0.9
    lang = 'en'  # English
    
    # Create gTTS object
    tts = gTTS(text=text_to_use, lang=lang, slow=slow)
    
    # 3. Save to WAV file in temp directory
    print(f"Saving to {OUTPUT_FILE}...")
    
    # gTTS saves as MP3, we need to convert to WAV
    # Save as MP3 first
    tts.save(TEMP_MP3)
    
    # Convert MP3 to WAV using pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(TEMP_MP3)
        audio.export(OUTPUT_FILE, format="wav")
        os.remove(TEMP_MP3)  # Clean up temp file
    except ImportError:
        # If pydub not available, just rename mp3 to wav (browser can handle it)
        print("Warning: pydub not available, saving as MP3 with .wav extension")
        os.rename(TEMP_MP3, OUTPUT_FILE)
    
    # Verify file was created and has content
    if not os.path.exists(OUTPUT_FILE):
        raise Exception(f"Output file {OUTPUT_FILE} was not created")
    
    file_size = os.path.getsize(OUTPUT_FILE)
    if file_size == 0:
        raise Exception("Generated audio file is empty")
    
    print(f"Success! Generated {file_size} bytes of audio")

except ValueError as ve:
    # Input validation errors
    print(f"Input Error: {ve}")
    try:
        with open(ERROR_FILE, "w") as f:
            f.write(f"Input validation failed: {ve}")
    except:
        pass
    sys.exit(1)

except Exception as main_e:
    # All other errors
    print(f"Critical Error: {main_e}")
    import traceback
    traceback.print_exc()
    
    # Write detailed error to file for debugging
    try:
        with open(ERROR_FILE, "w") as f:
            f.write(f"Error: {main_e}\n\nTraceback:\n")
            traceback.print_exc(file=f)
    except:
        pass
    
    sys.exit(1)
