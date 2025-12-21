import os
import sys
import json
import wave
import time
from google import genai
from google.genai import types

# Wave file helper function
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Save PCM audio data to WAV file"""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        return False

# Hardcoded Google API key (no .env needed)
GOOGLE_API_KEY = "AIzaSyBmFKpcMZfhkOuDUWCfAOPn5tR73Mx5t6s"

print(f"Using Google Gemini API Key: {GOOGLE_API_KEY[:10]}...")

try:
    # 1. Read Input from File (JSON format for parameters)
    text_to_use = "Hello, this is a test"
    voice_name = "Kore"  # Default voice
    speed = 1.0
    
    if os.path.exists("tts_input.txt"):
        try:
            with open("tts_input.txt", "r", encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Try to parse as JSON first
                    try:
                        config = json.loads(content)
                        text_to_use = config.get('text', text_to_use)
                        # Map voice selection to Gemini voices
                        voice_param = config.get('voice', 'default')
                        # Available Gemini voices: Puck, Charon, Kore, Fenrir, Aoede
                        voice_map = {
                            'default': 'Kore',
                            'calm': 'Aoede',
                            'energetic': 'Puck'
                        }
                        voice_name = voice_map.get(voice_param, 'Kore')
                        speed = config.get('speed', 1.0)
                    except json.JSONDecodeError as je:
                        print(f"Warning: JSON parse error, using text as-is: {je}")
                        # Fallback to plain text
                        text_to_use = content
        except FileNotFoundError:
            print("Warning: tts_input.txt not found, using default text")
        except Exception as read_err:
            print(f"Warning: Failed to read tts_input.txt: {read_err}")
    else:
        print("Info: tts_input.txt not found, using default text")
            
    # Validate text input
    if not text_to_use or not text_to_use.strip():
        raise ValueError("Text input is empty or invalid")
    
    print(f"Synthesizing: {text_to_use[:50]}...")
    print(f"Voice: {voice_name}, Speed: {speed}")

    # 2. Initialize Google Gemini client with error handling
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as client_err:
        raise Exception(f"Failed to initialize Gemini client: {client_err}")
    
    # 3. Generate speech using Gemini TTS with retry logic
    print("Generating speech with Gemini TTS...")
    
    # Build the prompt with speed instruction if needed
    prompt = text_to_use
    if speed != 1.0:
        speed_instruction = "slowly" if speed < 1.0 else "quickly"
        prompt = f"Say {speed_instruction}: {text_to_use}"
    
    # Retry logic for rate limits
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            # If successful, break out of retry loop
            break
            
        except Exception as api_err:
            error_str = str(api_err)
            
            # Handle rate limit errors (429)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded. Please try again in a few moments.")
            
            # Handle quota errors
            elif "quota" in error_str.lower() or "QUOTA_EXCEEDED" in error_str:
                raise Exception("API quota exceeded. Please check your Google Cloud billing.")
            
            # Handle invalid argument errors
            elif "400" in error_str or "INVALID_ARGUMENT" in error_str:
                raise Exception(f"Invalid request parameters: {error_str}")
            
            # Handle authentication errors
            elif "401" in error_str or "403" in error_str or "PERMISSION_DENIED" in error_str:
                raise Exception("Authentication failed. Please check your API key.")
            
            # Re-raise other errors
            else:
                raise Exception(f"API request failed: {error_str}")
    
    # 4. Extract audio data with validation
    print("Extracting audio data...")
    try:
        if not response or not response.candidates:
            raise ValueError("No response candidates received from API")
        
        if not response.candidates[0].content.parts:
            raise ValueError("No content parts in API response")
        
        inline_data = response.candidates[0].content.parts[0].inline_data
        if not inline_data or not inline_data.data:
            raise ValueError("No audio data in API response")
        
        data = inline_data.data
        
        if len(data) == 0:
            raise ValueError("Received empty audio data")
            
    except (AttributeError, IndexError, ValueError) as extract_err:
        raise Exception(f"Failed to extract audio data: {extract_err}")
    
    # 5. Save to WAV file with validation
    output_file = "tts_output.wav"
    print(f"Saving to {output_file}...")
    
    if not wave_file(output_file, data):
        raise Exception("Failed to save audio to WAV file")
    
    # Verify file was created and has content
    if not os.path.exists(output_file):
        raise Exception(f"Output file {output_file} was not created")
    
    file_size = os.path.getsize(output_file)
    if file_size == 0:
        raise Exception("Generated audio file is empty")
    
    print(f"Success! Generated {file_size} bytes of audio")

except ValueError as ve:
    # Input validation errors
    print(f"Input Error: {ve}")
    try:
        with open("tts_error.txt", "w") as f:
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
        with open("tts_error.txt", "w") as f:
            f.write(f"Error: {main_e}\n\nTraceback:\n")
            traceback.print_exc(file=f)
    except:
        pass
    
    sys.exit(1)
