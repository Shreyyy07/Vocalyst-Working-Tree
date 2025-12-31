from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import logging
import os

# Suppress TensorFlow/DeepFace logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("deepface").setLevel(logging.ERROR)

from pyneuphonic import Neuphonic, save_audio, TTSConfig
import io
from dotenv import load_dotenv
import traceback
import whisper  # Add import for Whisper
import numpy as np
import base64
import threading
from PIL import Image
from deepface import DeepFace
import subprocess  # Add at the top with other imports
import re
from itertools import tee
# import t
import pickle as pkl
import json
from flask import Flask, request, jsonify

# Debug information about Python environment
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("PYTHONPATH:", sys.path)
print("Current working directory:", os.getcwd())
print("Environment variables:", dict(os.environ))

# Load environment variables from api/.env
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000", "http://localhost:3001"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Accept"],
    "supports_credentials": True,
    "expose_headers": ["Content-Type", "Content-Disposition"]
}})

# Global Neuphonic Client (Load from environment)
NEUPHONIC_API_KEY = os.getenv('NEUPHONIC_API_KEY', '')
if NEUPHONIC_API_KEY:
    try:
        GLOBAL_TTS_CLIENT = Neuphonic(api_key=NEUPHONIC_API_KEY)
        print("Green Light: Global TTS Client Initialized.")
    except Exception as e:
        print(f"Red Alert: Global TTS Init Failed: {e}")
        GLOBAL_TTS_CLIENT = None
else:
    print("Warning: NEUPHONIC_API_KEY not set in environment")
    GLOBAL_TTS_CLIENT = None

# Define filler words set for analysis
FILLER_WORDS = {
    "um", "uh", "like", "you know", "well", "so", "actually", "basically", "i mean", 
    "right", "okay", "er", "hmm", "literally", "anyway", "of course", "i guess", 
    "in other words", "obviously", "to be honest", "just", "seriously", "you see", 
    "i suppose", "frankly", "well, i mean", "at the end of the day", "to tell the truth", 
    "as it were", "kind of", "sort of", "in a way", "that is", "as a matter of fact", 
    "in fact", "like i said", "more or less", "i don't know", "basically speaking", 
    "for sure", "you could say", "the thing is", "it s like", "put it another way", 
    "at least", "as such", "well you know", "i would say", "truth be told", "yeah", "and yeah",
    "um yeah", "um no", "um right", "like literally", "to", "erm", "let s see", "hm", "maybe",
    "maybe like", "really",
    # Add variations that Whisper commonly produces
    "umm", "ummm", "uhh", "uhhh", "hmm", "hmmm", "mm", "mmm", "mhm", "uh-huh", "mm-hmm",
    "ah", "ahh", "oh", "ohh", "eh", "ehh", "mm-mm", "uh huh", "mm hmm"
}


def ngrams(words, n):
    output = []
    for i in range(len(words) - n + 1):
        output.append(' '.join(words[i:i + n]))
    return output

# ============================================================================
# EXCEPTION HANDLING UTILITIES
# ============================================================================

def safe_write_json(filepath, data, create_backup=True):
    """
    Atomic JSON write with error handling and optional backup.
    Returns: (success: bool, error_message: str or None)
    """
    try:
        # Create backup if file exists
        if create_backup and os.path.exists(filepath):
            backup_path = f"{filepath}.backup"
            try:
                import shutil
                shutil.copy2(filepath, backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Write to temp file first (atomic operation)
        temp_path = f"{filepath}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        if os.name == 'nt':  # Windows
            if os.path.exists(filepath):
                os.replace(temp_path, filepath)
            else:
                os.rename(temp_path, filepath)
        else:  # Unix/Linux
            os.replace(temp_path, filepath)
        
        return True, None
    except IOError as e:
        error_msg = f"File I/O error: {str(e)}"
        logger.error(error_msg)
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error writing JSON: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def safe_read_json(filepath, default=None):
    """
    Safe JSON read with error handling.
    Returns: (data, error_message: str or None)
    """
    try:
        if not os.path.exists(filepath):
            return default if default is not None else [], None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error in {filepath}: {str(e)}"
        logger.error(error_msg)
        # Try to recover from backup
        backup_path = f"{filepath}.backup"
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Recovered data from backup: {backup_path}")
                return data, None
            except:
                pass
        return default if default is not None else [], error_msg
    except Exception as e:
        error_msg = f"Error reading JSON from {filepath}: {str(e)}"
        logger.error(error_msg)
        return default if default is not None else [], error_msg

def call_api_with_retry(api_func, max_retries=3, backoff_factor=2, *args, **kwargs):
    """
    Call an API function with exponential backoff retry logic.
    Returns: (result, error_message: str or None)
    """
    import time
    
    for attempt in range(max_retries):
        try:
            result = api_func(*args, **kwargs)
            return result, None
        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = f"API call failed after {max_retries} attempts: {str(e)}"
                logger.error(error_msg)
                return None, error_msg
            
            wait_time = backoff_factor ** attempt
            logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
    
    return None, "Max retries exceeded"

def get_fallback_insights():
    """
    Return generic insights when AI generation fails.
    """
    return [
        "Focus on maintaining a steady pace throughout your presentation.",
        "Try to minimize filler words like 'um' and 'uh' for clearer communication.",
        "Practice maintaining eye contact to engage your audience better.",
        "Work on varying your tone to keep the audience engaged.",
        "Consider recording yourself to identify areas for improvement."
    ]

def analyze_emotional_journey(emotion_timeline):
    """
    Analyzes a timeline of emotions captured during recording.
    Returns comprehensive emotion analysis including:
    - Dominant emotion
    - Emotional stability
    - Engagement score
    - Mood consistency
    """
    if not emotion_timeline or len(emotion_timeline) == 0:
        return {
            "dominant_emotion": "neutral",
            "emotional_stability": 1.0,
            "engagement_score": 50,
            "emotion_distribution": {},
            "mood_consistency": 1.0
        }
    
    # Calculate average of each emotion across all snapshots
    emotion_sums = {}
    for snapshot in emotion_timeline:
        for emotion, score in snapshot.items():
            if emotion not in emotion_sums:
                emotion_sums[emotion] = []
            emotion_sums[emotion].append(score)
    
    # Calculate averages
    emotion_averages = {}
    for emotion, scores in emotion_sums.items():
        emotion_averages[emotion] = sum(scores) / len(scores)
    
    # Find dominant emotion
    dominant_emotion = max(emotion_averages.items(), key=lambda x: x[1])[0]
    
    # Calculate emotional stability (low variance = high stability)
    variances = []
    for emotion, scores in emotion_sums.items():
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        variances.append(variance)
    avg_variance = sum(variances) / len(variances) if variances else 0
    emotional_stability = max(0, 1 - (avg_variance * 10))  # Normalize to 0-1
    
    # Calculate engagement score (based on expressiveness - opposite of neutral)
    neutral_avg = emotion_averages.get('neutral', 0)
    engagement_score = round((1 - neutral_avg) * 100)  # Higher when NOT neutral
    
    # Mood consistency (how much emotions change over time)
    mood_shifts = 0
    prev_dominant = None
    for snapshot in emotion_timeline:
        curr_dominant = max(snapshot.items(), key=lambda x: x[1])[0]
        if prev_dominant and curr_dominant != prev_dominant:
            mood_shifts += 1
        prev_dominant = curr_dominant
    
    mood_consistency = max(0, 1 - (mood_shifts / len(emotion_timeline))) if len(emotion_timeline) > 1 else 1.0
    
    return {
        "dominant_emotion": dominant_emotion,
        "emotional_stability": round(emotional_stability, 2),
        "engagement_score": engagement_score,
        "emotion_distribution": {k: round(v, 3) for k, v in emotion_averages.items()},
        "mood_consistency": round(mood_consistency, 2)
    }

def analyse_filler_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    # Debug: Print what we're analyzing
    print(f"Analyzing text for fillers: '{text[:200]}...'")  # First 200 chars
    print(f"Total words found: {total_words}")
    
    # Count filler words
    filler_count = 0
    found_fillers: list[str] = []
    
    for word in words:
        if word in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(word)
            print(f"Found single filler: '{word}'")
    
    bigrams = ngrams(words, 2)
    for bigram in bigrams:
        if bigram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(bigram)
            print(f"Found bigram filler: '{bigram}'")
    
    trigrams = ngrams(words, 3)
    for trigram in trigrams:
        if trigram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(trigram)
            print(f"Found trigram filler: '{trigram}'")
    
    quadgrams = ngrams(words, 4)
    for quadgram in quadgrams:
        if quadgram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(quadgram)
            print(f"Found quadgram filler: '{quadgram}'")
    
    # Calculate metrics
    filler_percentage = (filler_count / total_words * 100) if total_words > 0 else 0
    
    # Determine emoji based on percentage
    if filler_percentage <= 3:
        emoji = "🎯"  # Excellent
    elif filler_percentage <= 7:
        emoji = "👍"  # Good
    elif filler_percentage <= 12:
        emoji = "💭"  # Think about it
    elif filler_percentage <= 18:
        emoji = "⚠️"  # Warning
    else:
        emoji = "😞"  # Needs work
    
    # Calculate TTR
    ttr_analysis = calculate_ttr(text)
    

    
    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_percentage": round(filler_percentage, 2),
        "found_fillers": found_fillers,
        "filler_emoji": emoji,
        "ttr_analysis": ttr_analysis,

    }

def calculate_ttr(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words: int = len(words)
    unique_words: int = len(set(words))
    ttr: float = (unique_words / total_words) * 100 if total_words > 0 else 0
    
    # Determine diversity level and emoji
    if ttr >= 80:
        diversity = "very high"
        emoji = "🌟"
    elif ttr >= 60:
        diversity = "high"
        emoji = "✨"
    elif ttr >= 40:
        diversity = "average"
        emoji = "💫"
    elif ttr >= 20:
        diversity = "low"
        emoji = "📝"
    else:
        diversity = "very low"
        emoji = "📚"
        
    return {
        "ttr": round(ttr, 2),
        "unique_words": unique_words,
        "diversity_level": diversity,
        "emoji": emoji
    }



def detect_emotions(image: Image) -> dict:
    """
    Detects faces and emotions in an image using DeepFace.
    Returns normalized emotion scores and cropped face image.
    """
    # Convert PIL Image to numpy array for DeepFace
    image_arr = np.array(image)
    
    # Initialize variables to prevent UnboundLocalError
    face_base64 = None
    normalized_emotions = {}
    
    try:
        # Detect emotion using DeepFace
        try:
            # Wrap DeepFace to prevent crashing on OpenCV assertion errors
            # Use 'ssd' or 'mtcnn' if opencv fails, or 'skip' if we accept full image
            # We'll try 'ssd' as it's generally robust, or fall back to 'opencv' if needed
            result = DeepFace.analyze(image_arr, 
                                    actions=['emotion'], 
                                    detector_backend='ssd',
                                    enforce_detection=False)
        except Exception as e:
            logger.warning(f"DeepFace (ssd) analysis failed with error: {str(e)}")
            # Try once more with default opencv but catching errors
            try:
                 result = DeepFace.analyze(image_arr, 
                                    actions=['emotion'], 
                                    detector_backend='opencv',
                                    enforce_detection=False)
            except Exception as e2:
                logger.warning(f"DeepFace (opencv) analysis also failed: {str(e2)}")
                # Provide fallback/empty result so stream continues
                result = [{'dominant_emotion': 'neutral', 'emotion': {'neutral': 100}}]

        
        # Handle multiple faces by taking first one
        # Handle multiple faces by taking first one
        if isinstance(result, list):
            if len(result) > 0:
                result = result[0]
            else:
                 # Empty list returned
                 result = {'dominant_emotion': 'neutral', 'emotion': {'neutral': 100}, 'region': {}}
        
        # Extract face region coordinates
        face_region = result.get('region', {})
        x, y = face_region.get('x', 0), face_region.get('y', 0)
        w, h = face_region.get('w', 0), face_region.get('h', 0)
        
        # Crop the face if detected
        if w > 0 and h > 0:
            # crop and encode detected face
            face = image.crop((x, y, x + w, y + h))
            
            # Convert face to base64
            face_bytes = io.BytesIO()
            face.save(face_bytes, format='JPEG')
            face_base64 = base64.b64encode(face_bytes.getvalue()).decode('utf-8')
            
            # normalize emotion scores to 0-1 range
            emotions = result['emotion']
            total = sum(emotions.values())
            normalized_emotions = {
                emotion: round(score / total, 3)
                for emotion, score in emotions.items()
            }
            
            # Sort by intensity (highest first)
            normalized_emotions = dict(
                sorted(normalized_emotions.items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
            
            return {
                "success": True,
                "face_detected": True,
                "face_image": face_base64,
                "emotions": normalized_emotions
            }
    
    except Exception as e:
        print(f"Error detecting emotions: {str(e)}")
        traceback.print_exc()
    
    return {
        "success": True,
        "face_detected": False,
        "emotions": {}
    }

@app.route("/api/detect-emotion", methods=['POST'])
def detect_emotion():
    """Endpoint to detect emotions in uploaded images"""
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse base64 image data
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process the image
        result = detect_emotions(image)
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Lazy-loaded MediaPipe Face Landmarker (loads on first use)
_face_detector = None
_detector_lock = None

def get_face_detector():
    """Lazy load and cache the MediaPipe Face Landmarker."""
    global _face_detector, _detector_lock
    
    # Initialize lock on first call
    if _detector_lock is None:
        import threading
        _detector_lock = threading.Lock()
    
    if _face_detector is None:
        with _detector_lock:
            # Double-check after acquiring lock
            if _face_detector is None:
                try:
                    model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker_v2_with_blendshapes.task')
                    logger.info(f"Lazy loading MediaPipe model from: {model_path}")
                    if not os.path.exists(model_path):
                        logger.error(f"Model file not found at: {model_path}")
                        raise FileNotFoundError(f"Model file not found at: {model_path}")
                        
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                        num_faces=1,
                        running_mode=vision.RunningMode.IMAGE
                    )
                    _face_detector = vision.FaceLandmarker.create_from_options(options)
                    logger.info("Successfully loaded face landmarker model")
                except Exception as e:
                    logger.error(f"Error initialising face landmarker: {str(e)}")
                    traceback.print_exc()
                    raise
    
    return _face_detector

def determine_gaze_direction(face_landmarks):
    # Convert FACEMESH_LEFT_IRIS from tuple to list of integers
    left_iris_indices = list(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
    right_iris_indices = list(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)
    
    # Calculate mean position of left iris
    left_points = []
    for idx in left_iris_indices:
        if isinstance(idx, tuple):
            idx = idx[0]
        left_points.append([
            face_landmarks[idx].x,
            face_landmarks[idx].y,
            face_landmarks[idx].z
        ])
    left_iris = np.mean(left_points, axis=0)
    
    # Calculate mean position of right iris
    right_points = []
    for idx in right_iris_indices:
        if isinstance(idx, tuple):
            idx = idx[0]
        right_points.append([
            face_landmarks[idx].x,
            face_landmarks[idx].y,
            face_landmarks[idx].z
        ])
    right_iris = np.mean(right_points, axis=0)
    
    # Get eye corners for better vertical gaze detection
    left_eye_outer = face_landmarks[33]  # Outer corner of left eye
    left_eye_inner = face_landmarks[133]  # Inner corner of left eye
    right_eye_outer = face_landmarks[263]  # Outer corner of right eye
    right_eye_inner = face_landmarks[362]  # Inner corner of right eye
    
    # Calculate eye centers
    left_eye_center = np.mean([[left_eye_outer.x, left_eye_outer.y], 
                              [left_eye_inner.x, left_eye_inner.y]], axis=0)
    right_eye_center = np.mean([[right_eye_outer.x, right_eye_outer.y], 
                               [right_eye_inner.x, right_eye_inner.y]], axis=0)
    
    # Calculate relative positions
    x_diff = (left_iris[0] + right_iris[0]) / 2 - 0.5
    
    # Calculate vertical gaze using distance from iris to eye center
    left_y_diff = left_iris[1] - left_eye_center[1]
    right_y_diff = right_iris[1] - right_eye_center[1]
    y_diff = (left_y_diff + right_y_diff) / 2
    
    # Adjust thresholds
    x_threshold = 0.05
    y_threshold = 0.02  # More sensitive threshold for vertical movement
    
    if abs(x_diff) < x_threshold and abs(y_diff) < y_threshold:
        return "center"
    
    vertical = ""
    horizontal = ""
    
    if y_diff < -y_threshold:
        vertical = "up"
    elif y_diff > y_threshold:
        vertical = "down"
        
    if x_diff < -x_threshold:
        horizontal = "left"
    elif x_diff > x_threshold:
        horizontal = "right"
        
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "center"

def process_frame(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect face landmarks using lazy-loaded detector
    detector = get_face_detector()
    detection_result = detector.detect(mp_image)
    
    # If no faces detected, return original frame
    if not detection_result.face_landmarks:
        return frame
    
    # Get the first face detected
    face_landmarks = detection_result.face_landmarks[0]
    
    # Draw the face landmarks
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
        for landmark in face_landmarks
    ])

    # Draw face mesh tesselation
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )

    # Draw face mesh contours
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
    )

    # Draw irises
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
    )
    
    # Convert back to BGR for OpenCV
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

def gen_frames():
    logger.info("Starting video capture")
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            try:
                processed_frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                # Return original frame if processing fails
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

# Lazy-loaded Whisper model (loads on first use)
_whisper_model = None
_whisper_lock = None

def get_whisper_model():
    """Lazy load and cache the Whisper model."""
    global _whisper_model, _whisper_lock
    
    # Initialize lock on first call
    if _whisper_lock is None:
        import threading
        _whisper_lock = threading.Lock()
    
    if _whisper_model is None:
        with _whisper_lock:
            # Double-check after acquiring lock
            if _whisper_model is None:
                logger.info("Lazy loading Whisper model (base)...")
                model = whisper.load_model("base")
                if not hasattr(model, 'transcribe'):
                    logger.error(f"Loaded object is not a Whisper model! Type: {type(model)}")
                    # Try to force reload or fail hard
                    raise RuntimeError("Failed to load valid Whisper model")
                _whisper_model = model
                logger.info("Whisper model loaded successfully")
    
    return _whisper_model

@app.route("/api/python")
def hello_world():
    """Simple health check endpoint"""
    return "Hello, World!"

@app.route("/api/tts", methods=['POST'])
def text_to_speech():
    """Converts text to speech using Google Gemini API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        speed = max(0.7, min(2.0, float(data.get('speed', 1.0))))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Sanitize text
        text = text.strip()
        print(f"Processing TTS request - text: {repr(text)}, voice: {voice}, speed: {speed}")

        # Google Gemini TTS doesn't need client initialization
        # Voice parameter is passed directly to subprocess
             

        # SUBPROCESS ISOLATION (File-Based IPC)
        # We write input to file, run script, read output from file.
        # This bypasses all environment/threading options.
        import subprocess
        
        # Paths - all in temp directory to keep api folder clean
        current_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(current_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)  # Ensure temp directory exists
        
        script_path = os.path.join(current_dir, 'simple_tts.py')
        input_path = os.path.join(temp_dir, 'tts_input.txt')
        output_path = os.path.join(temp_dir, 'tts_output.wav')
        error_path = os.path.join(temp_dir, 'tts_error.txt')
        
        # Cleanup
        for p in [input_path, output_path, error_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
                
        # Write Input as JSON with all config
        tts_input_config = {
            'text': text,
            'voice': voice,  # Pass voice directly (Gemini handles mapping)
            'speed': speed
        }
        try:
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(tts_input_config, f)
        except Exception as write_err:
            print(f"Error writing input file: {write_err}")
            return jsonify({"error": "Failed to prepare TTS request"}), 500
            
        print(f"Executing TTS Subprocess: {script_path}")
        
        # Run
        try:
            # Run with same python executable
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout for Gemini API
                env=os.environ
            )
            
            print(f"Subprocess Stdout: {result.stdout}")
            if result.stderr:
                print(f"Subprocess Stderr: {result.stderr}")
            
            # Check for error file first
            if os.path.exists(error_path):
                with open(error_path, 'r') as f:
                    err_msg = f.read()
                # Parse error message for user-friendly response
                if "Rate limit" in err_msg or "429" in err_msg:
                    return jsonify({"error": "TTS service is busy. Please try again in a moment."}), 429
                elif "quota" in err_msg.lower() or "QUOTA" in err_msg:
                    return jsonify({"error": "TTS quota exceeded. Please contact support."}), 503
                elif "Authentication" in err_msg or "API key" in err_msg:
                    return jsonify({"error": "TTS service configuration error."}), 500
                else:
                    return jsonify({"error": f"TTS generation failed: {err_msg[:100]}"}), 500
            
            # Check return code
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                print(f"Subprocess failed with code {result.returncode}: {error_msg}")
                return jsonify({"error": "TTS generation failed. Please try again."}), 500
            
            # Validate output file exists
            if not os.path.exists(output_path):
                print("Error: Output file not created")
                return jsonify({"error": "TTS generation failed to produce audio."}), 500
            
            # Read and validate output
            try:
                with open(output_path, 'rb') as f:
                    audio_data = f.read()
            except Exception as read_err:
                print(f"Error reading output file: {read_err}")
                return jsonify({"error": "Failed to read generated audio."}), 500
            
            # Validate audio data
            if len(audio_data) == 0:
                print("Error: Generated audio file is empty")
                return jsonify({"error": "Generated audio is empty."}), 500
            
            if len(audio_data) < 100:  # WAV header is ~44 bytes, so <100 is suspicious
                print(f"Warning: Audio file is very small ({len(audio_data)} bytes)")
            
            # Load into BytesIO
            temp_file = io.BytesIO(audio_data)
            
            print(f"Subprocess Success. Audio size: {len(audio_data)} bytes")
            
        except subprocess.TimeoutExpired:
            print("TTS subprocess timed out")
            return jsonify({"error": "TTS generation timed out. Please try a shorter text."}), 504
        except Exception as proc_err:
            print(f"Subprocess Execution Error: {proc_err}")
            return jsonify({"error": "TTS service error. Please try again."}), 500

        # Debug audio generation
        temp_file.seek(0)
        audio_data = temp_file.read()
        print(f"Generated audio size: {len(audio_data)} bytes")
        
        temp_file.seek(0)
        
        # Return audio file with CORS headers
        response = send_file(
            temp_file,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='speech.wav'
        )
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response
    
    except Exception as e:
        print("TTS Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def extract_video_frames(video_path, fps=0.5):
    """
    Extract frames from video at specified FPS.
    Default: 0.5 FPS (1 frame every 2 seconds)
    
    Returns:
        List of (timestamp, frame_array) tuples
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return frames
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"Video FPS: {video_fps}, Duration: {duration}s, Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps * 2)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps if video_fps > 0 else extracted_count * 2
            frames.append((timestamp, frame))
            extracted_count += 1
            logger.info(f"Extracted frame at {timestamp:.2f}s")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames






@app.route("/api/upload-video", methods=['POST'])
def upload_video():
    """Handles video file uploads and saves them locally"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp}.mp4"
        audio_filename = f"recording_{timestamp}_audio.wav"
        video_path = os.path.join(upload_dir, video_filename)
        audio_path = os.path.join(upload_dir, audio_filename)
        
        # Save the video file
        video_file.save(video_path)
        
        # Check if video has audio stream using ffprobe
        probe_result = subprocess.run([
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=nw=1:nk=1',
            video_path
        ], capture_output=True, text=True)
        
        has_audio = probe_result.stdout.strip() == 'audio'
        
        if has_audio:
            try:
                # Extract audio with quality improvements
                subprocess.run([
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ac', '2',  # Stereo
                    audio_path
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print("FFmpeg error:", e.stderr)
                has_audio = False
            except FileNotFoundError:
                print("FFmpeg not found")
                has_audio = False
        
        
        # Extract frames and analyze emotions
        logger.info("Starting video frame extraction for emotion analysis...")
        emotion_timeline = []
        emotion_timestamps = []
        
        try:
            # Extract frames from video (1 frame every 2 seconds)
            frames = extract_video_frames(video_path, fps=0.5)
            logger.info(f"Extracted {len(frames)} frames for analysis")
            
            # Analyze emotion in each frame
            for timestamp, frame in frames:
                try:
                    # Convert cv2 frame (BGR) to PIL Image (RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Detect emotions using existing function
                    emotion_result = detect_emotions(pil_image)
                    
                    if emotion_result and 'emotions' in emotion_result:
                        emotion_timeline.append(emotion_result['emotions'])
                        emotion_timestamps.append(timestamp)
                        logger.info(f"Frame at {timestamp:.2f}s - Dominant: {emotion_result['emotions'].get('dominant', 'unknown')}")
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze frame at {timestamp:.2f}s: {e}")
                    continue
            
            logger.info(f"Successfully analyzed {len(emotion_timeline)} frames")
            
        except Exception as e:
            logger.error(f"Error during frame extraction/analysis: {e}")
            # Continue even if emotion analysis fails
        
        
        response_data = {
            "success": True,
            "message": "Video uploaded successfully",
            "filename": video_filename,
            "audio_filename": audio_filename if has_audio else None,
            "has_audio": has_audio
        }
        
        # Save emotion timeline and add to response
        if emotion_timeline:
            emotion_file = os.path.join(upload_dir, f"recording_{timestamp}_emotions.json")
            emotion_data = {
                "emotions": emotion_timeline,
                "timestamps": emotion_timestamps
            }
            with open(emotion_file, 'w') as f:
                json.dump(emotion_data, f)
            response_data['has_emotions'] = True
            response_data['emotion_count'] = len(emotion_timeline)
        
        return jsonify(response_data)
        
    except Exception as e:
        print("Video Upload Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Lock for transcription to avoid race conditions with global model state
_transcription_lock = threading.Lock()

def transcribe_long_audio(audio_path, max_duration=30):
    """
    Transcribe longer audio files by splitting into chunks if needed.
    Returns the full transcription text.
    """
    try:
        # Transcribe directly without duration checking
        # Whisper can handle short and long audio automatically
        print(f"Transcribing audio file directly with Whisper...")
        whisper_model = get_whisper_model()
        
        # Lock to ensure only one transcription happens at a time
        # This prevents 'KeyError: Linear' and other race conditions in model inference
        with _transcription_lock:
            result = whisper_model.transcribe(audio_path, language='en', fp16=False)
            
        transcribed_text = result.get("text", "").strip()
        print(f"Whisper result object keys: {result.keys()}")
        print(f"Raw transcription: '{transcribed_text}'")
        print(f"Transcription length: {len(transcribed_text)}")
        
        if not transcribed_text:
            print("WARNING: Whisper returned empty transcription!")
            print(f"Segments: {result.get('segments', [])}")
            return None
        
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Transcription error type: {type(e)}")
        logger.error(f"Transcription error: {str(e)}")
        # Check if the error is a PyTorch module (weird case observed)
        if 'Linear' in str(e) or 'Conv1d' in str(e):
             logger.critical("Model seems to be corrupted or misloaded. Wrapper exception?")
             
        traceback.print_exc()
        return None

@app.route("/api/speech2text", methods=['POST'])
def transcribe():
    """Converts speech audio to text using Whisper model"""
    try:
        print("Starting transcription request...")
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_audio = os.path.join(temp_dir, "temp_audio.mp3")

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Get extension from original filename
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = '.mp4' # Default fallback
            
        temp_input = os.path.join(temp_dir, f"temp_input{ext}")
        
        print(f"Received file: {file.filename}, mimetype: {file.content_type}")
        # Save and verify input file
        file.save(temp_input)
        input_size = os.path.getsize(temp_input)
        print(f"Saved input file to {temp_input} (size: {input_size} bytes)")
        
        if input_size == 0:
            raise Exception("Input file is empty")

        # Convert to standard WAV format (16kHz, mono) to prevent model crashes
        # This matches the robust logic in enhance_audio
        temp_wav = os.path.join(temp_dir, "temp_clean.wav")
        print(f"Normalizing audio format: {temp_input} -> {temp_wav}")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-i', temp_input,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            temp_wav
        ]
        
        try:
             subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)
             if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                 print("WARNING: FFmpeg conversion failed/empty, falling back to original.")
                 target_file = temp_input
             else:
                 target_file = temp_wav
        except Exception as e:
            print(f"FFmpeg conversion error: {e}. Using original file.")
            target_file = temp_input

        print(f"Transcribing file: {target_file}")
        text = transcribe_long_audio(target_file)
        print(f"Transcription result: '{text}' (length: {len(text) if text else 0})")
        
        if not text:
            print("WARNING: Transcription returned empty text")
            # Return a special response for no speech
            return jsonify({
                "text": "",
                "no_speech_detected": True,
                "analysis": {
                    "total_words": 0,
                    "filler_percentage": 0,
                    "filler_count": 0,
                    "found_fillers": {},
                    "logical_flow": {"score": 0, "emoji": "none"},
                    "ttr_analysis": {"diversity_level": "none", "emoji": "none"}
                },
                "speaking_metrics": {
                    "word_count": 0,
                    "confidence_level": "None",
                    "confidence_emoji": "😶"
                }
            })
        
        
        # Comprehensive speech analysis
        analysis = analyse_filler_words(text)
        
        # Add advanced speech metrics
        words = text.split()
        word_count = len(words)
        
        # Speaking rate (assuming average recording is around the actual duration)
        # We'll estimate based on word count - typical speech is 120-150 WPM
        unique_words = len(set([w.lower() for w in words if w.isalpha()]))
        
        # Vocabulary richness (unique word ratio)
        vocab_ratio = (unique_words / word_count * 100) if word_count > 0 else 0
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        
        # Pause detection (estimate based on punctuation and sentence structure)
        pause_indicators = text.count('.') + text.count('!') + text.count('?') + text.count(',')
        
        # Confidence score (based on filler percentage - inverse relationship)
        filler_pct = analysis.get('filler_percentage', 0)
        confidence_score = max(0, 100 - (filler_pct * 3))  # Lower fillers = higher confidence
        
        # Determine confidence level
        if confidence_score >= 80:
            confidence_level = "High"
            confidence_emoji = "💪"
        elif confidence_score >= 60:
            confidence_level = "Good" 
            confidence_emoji = "👍"
        elif confidence_score >= 40:
            confidence_level = "Fair"
            confidence_emoji = "🤔"
        else:
            confidence_level = "Needs Improvement"
            confidence_emoji = "📈"
        
        # Add all new metrics to analysis
        analysis['speaking_metrics'] = {
            'word_count': word_count,
            'unique_words': unique_words,
            'vocabulary_richness': round(vocab_ratio, 1),
            'avg_word_length': round(avg_word_length, 1),
            'estimated_pauses': pause_indicators,
            'confidence_score': round(confidence_score, 1),
            'confidence_level': confidence_level,
            'confidence_emoji': confidence_emoji
        }
        
        # Try to load and analyze emotions if available
        emotion_analysis = None
        try:
            # Extract base filename from uploaded file to find matching emotion file
            original_filename = file.filename
            # Look for emotion JSON file in uploads directory
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            
            # Try to find emotion file by looking for recent JSON files
            if os.path.exists(upload_dir):
                # Get all emotion JSON files sorted by modification time (newest first)
                emotion_files = [f for f in os.listdir(upload_dir) if f.endswith('_emotions.json')]
                if emotion_files:
                    # Use the most recent one
                    emotion_files.sort(key=lambda x: os.path.getmtime(os.path.join(upload_dir, x)), reverse=True)
                    emotion_file_path = os.path.join(upload_dir, emotion_files[0])
                    
                    print(f"Loading emotions from: {emotion_file_path}")
                    with open(emotion_file_path, 'r') as f:
                        data = json.load(f)
                        # Handle both list (legacy) and dict (new) formats
                        if isinstance(data, dict) and 'emotions' in data:
                            emotion_timeline = data['emotions']
                        else:
                            emotion_timeline = data
                    
                    if emotion_timeline and isinstance(emotion_timeline, list) and len(emotion_timeline) > 0:
                        emotion_analysis = analyze_emotional_journey(emotion_timeline)
                        print(f"Emotion analysis complete: {emotion_analysis.get('dominant_emotion')}")
        except Exception as e:
            print(f"Could not load emotion analysis: {e}")
            # Don't fail the request if emotions aren't available
            pass
            
        response_data = {
            "text": text,
            "analysis": analysis
        }
        
        # Include emotion analysis if available
        if emotion_analysis:
            response_data['emotion_analysis'] = emotion_analysis
            
        return jsonify(response_data)
            
    except Exception as e:
        print(f"General Error: {str(e)}")
        traceback.print_exc()
        traceback.print_exc()
        return jsonify({"error": "Failed to process request"}), 500
    finally:
        # Clean up temp files
        for temp_file in [temp_input, temp_audio]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Failed to clean up {temp_file}: {str(e)}")

@app.route("/api/test", methods=['GET'])
def test_endpoint():
    """Health check endpoint for emotion detection API"""
    return jsonify({"status": "ok", "message": "Emotion detection API is running"})

@app.route("/uploads/<path:filename>")
def serve_file(filename):
    """Serves files from the uploads directory"""
    try:
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
            
        # Determine mime type based on extension
        if filename.endswith('.mp4'):
            mime_type = 'video/mp4'
        elif filename.endswith('.wav'):
            mime_type = 'audio/wav'
        else:
            mime_type = 'audio/mpeg'
        
        print(f"Serving file: {file_path} with mime type: {mime_type}")
        return send_file(
            file_path,
            mimetype=mime_type,
            as_attachment=False
        )
    except Exception as e:
        print("Error serving file:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": "File not found"}), 404

@app.route("/api/detect-gaze", methods=['POST'])
def detect_gaze():
    """Endpoint to detect gaze direction in uploaded images"""
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse base64 image data
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert PIL Image to numpy array for MediaPipe
        image_arr = np.array(image)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        
        # Detect face landmarks using lazy-loaded detector
        detector = get_face_detector()
        detection_result = detector.detect(mp_image)
        
        # If no faces detected, return empty result
        if not detection_result.face_landmarks:
            return jsonify({
                "success": True,
                "face_detected": False
            })
        
        # Get the first face detected
        face_landmarks = detection_result.face_landmarks[0]
        
        # Get gaze direction
        gaze_direction = determine_gaze_direction(face_landmarks)
        
        # Convert landmarks to normalized coordinates
        landmarks = [[landmark.x, landmark.y] for landmark in face_landmarks]
        
        # Calculate face bounding box
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]
        face_box = {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }
        
        # Calculate gaze arrow
        # Use eye landmarks to create arrow
        left_eye_center = np.mean([[face_landmarks[33].x, face_landmarks[33].y],  # Outer corner
                                 [face_landmarks[133].x, face_landmarks[133].y]], axis=0)  # Inner corner
        right_eye_center = np.mean([[face_landmarks[263].x, face_landmarks[263].y],  # Outer corner
                                  [face_landmarks[362].x, face_landmarks[362].y]], axis=0)  # Inner corner
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
        
        # Calculate arrow end point based on gaze direction
        arrow_length = 0.1  # Adjust this value to change arrow length
        arrow_end = eye_center.copy()
        
        if "left" in gaze_direction:
            arrow_end[0] -= arrow_length
        elif "right" in gaze_direction:
            arrow_end[0] += arrow_length
        if "up" in gaze_direction:
            arrow_end[1] -= arrow_length
        elif "down" in gaze_direction:
            arrow_end[1] += arrow_length
            
        gaze_arrow = {
            "start": {"x": float(eye_center[0]), "y": float(eye_center[1])},
            "end": {"x": float(arrow_end[0]), "y": float(arrow_end[1])}
        }
        
        return jsonify({
            "success": True,
            "face_detected": True,
            "gaze_direction": gaze_direction,
            "landmarks": landmarks,
            "face_box": face_box,
            "gaze_arrow": gaze_arrow
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/detect-combined", methods=['POST'])
def detect_combined():
    """Endpoint to detect both emotions and gaze direction in a single call"""
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse base64 image data
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Process emotions
        emotion_result = detect_emotions(image)
        
        # Process gaze using lazy-loaded detector
        image_arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        detector = get_face_detector()
        detection_result = detector.detect(mp_image)
        
        # If no faces detected
        if not detection_result.face_landmarks:
            return jsonify({
                "success": True,
                "face_detected": False,
                "emotions": {},
                "gaze": None
            })
        
        # Get the first face detected
        face_landmarks = detection_result.face_landmarks[0]
        
        # Get gaze direction
        gaze_direction = determine_gaze_direction(face_landmarks)
        
        # Convert landmarks to normalized coordinates
        landmarks = [[landmark.x, landmark.y] for landmark in face_landmarks]
        
        # Calculate face bounding box
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]
        face_box = {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }
        
        # Calculate gaze arrow
        left_eye_center = np.mean([[face_landmarks[33].x, face_landmarks[33].y], 
                                 [face_landmarks[133].x, face_landmarks[133].y]], axis=0)
        right_eye_center = np.mean([[face_landmarks[263].x, face_landmarks[263].y], 
                                  [face_landmarks[362].x, face_landmarks[362].y]], axis=0)
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
        
        # Calculate arrow end point based on gaze direction
        arrow_length = 0.1
        arrow_end = eye_center.copy()
        
        if "left" in gaze_direction:
            arrow_end[0] -= arrow_length
        elif "right" in gaze_direction:
            arrow_end[0] += arrow_length
        if "up" in gaze_direction:
            arrow_end[1] -= arrow_length
        elif "down" in gaze_direction:
            arrow_end[1] += arrow_length
            
        gaze_arrow = {
            "start": {"x": float(eye_center[0]), "y": float(eye_center[1])},
            "end": {"x": float(arrow_end[0]), "y": float(arrow_end[1])}
        }
        
        return jsonify({
            "success": True,
            "face_detected": True,
            "emotions": emotion_result.get("emotions", {}),
            "gaze": {
                "direction": gaze_direction,
                "landmarks": landmarks,
                "face_box": face_box,
                "gaze_arrow": gaze_arrow
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/enhance-audio", methods=['POST', 'OPTIONS'])
def enhance_audio():
    """Enhances audio using Neuphonic API"""
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins for OPTIONS
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    try:
        logger.info("Starting audio enhancement request")
        
        # Default text in case transcription fails
        default_text = "Thank you for your speech. While we couldn't analyze the specific content, keep practicing your speaking skills regularly. Good communication is a valuable skill that improves with consistent practice."
        
        # Set speech speed
        speech_speed = 1.0
        
        # Parse category from request if available
        practice_category = None
        try:
            if request.form and 'category' in request.form:
                practice_category = request.form.get('category')
                logger.info(f"Received practice category: {practice_category}")
            elif request.args and 'category' in request.args:
                practice_category = request.args.get('category')
                logger.info(f"Received practice category from query params: {practice_category}")
        except Exception as e:
            logger.warning(f"Error parsing category: {str(e)}")
        # Load API key from environment
        api_key = os.getenv('NEUPHONIC_API_KEY', '')
        if not api_key:
            return jsonify({"error": "NEUPHONIC_API_KEY not configured"}), 500
        api_key = api_key.strip()
        
        # Initialize client early
        client = Neuphonic(api_key=api_key)
        
        # Attempt to process the audio file, but continue even if it fails
        analysis = None
        text = ""
        
        try:
            # Process audio file if present
            if 'file' in request.files and request.files['file'].filename:
                temp_dir = os.path.join(os.getcwd(), 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_audio = os.path.join(temp_dir, "temp_audio.wav")
                temp_input = os.path.join(temp_dir, "temp_input.mp4") # Default to mp4, will overwrite if file saved
                
                file = request.files['file']
                logger.info(f"Received file: {file.filename}, mimetype: {file.content_type}")
                
                # Save and verify input file
                file.save(temp_input)
                input_size = os.path.getsize(temp_input)
                logger.info(f"Saved input file to {temp_input} (size: {input_size} bytes)")
                
                if input_size > 0:
                    # Convert audio to MP3 format for transcription
                    logger.info("Converting audio for transcription")
                    ffmpeg_cmd = [
                        'ffmpeg',
 '-y', 
 '-i', temp_input, 
 '-vn', 
 '-acodec', 'pcm_s16le', 
 '-ar', '16000', 
 '-ac', '1', 
 temp_audio
                    ]
                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)
                        
                        if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                            # Transcribe the audio
                            logger.info("Transcribing audio")
                            text = transcribe_long_audio(temp_audio)
                            
                            if text:
                                # Analyze the text
                                logger.info(f"Analyzing transcribed text (length: {len(text)})")
                                analysis = analyse_filler_words(text)
                    except Exception as conversion_error:
                        logger.error(f"Audio processing error (continuing anyway): {str(conversion_error)}")
                    
                    finally:
                        # Clean up temp files
                        for temp_file in [temp_input, temp_audio]:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception as e:
                                logger.error(f"Failed to clean up {temp_file}: {str(e)}")
        
        except Exception as file_error:
            logger.error(f"File processing error (continuing anyway): {str(file_error)}")
        
        # Default category-specific feedback (used when no analysis available)
        category_default_feedback = {
            "persuasive": "This was a persuasive speech practice. Remember to use concrete evidence, address counterarguments, and maintain a confident tone.",
            "emotive": "This was an emotive speech practice. Keep using appropriate emotional language and matching your tone to the emotions you express.",
            "public-speaking": "This was a public speaking practice. Focus on clear projection, engaging body language, and structured points.",
            "rizzing": "This was a charismatic conversation practice. Stay confident and authentic while using appropriate humor and social cues.",
            "basic-conversations": "This was a casual conversation practice. Keep the conversation flowing naturally with open-ended questions.",
            "formal-conversations": "This was a formal conversation practice. Maintain professional language and be concise and clear.",
            "debating": "This was a debate practice. Present logical arguments, listen to counterpoints, and support claims with evidence.",
            "storytelling": "This was a storytelling practice. Focus on clear narrative structure, descriptive language, and audience engagement."
        }
        
        # If we have valid analysis, create customized summary
        enhanced_text = default_text
        metrics_data = {
            "total_words": 0,
            "unique_words": 0,
            "vocabulary_score": 0,
            "logical_flow_score": 0,
            "filler_percentage": 0,
            "practice_category": practice_category or "unknown"
        }
        
        if analysis:
            # Create a cohesive summary speech based on analysis
            filler_comment = ""
            if analysis['filler_percentage'] >= 18:
                filler_comment = "I noticed you used quite a few filler words in your speech. Try to be more mindful of these."
            elif analysis['filler_percentage'] >= 12:
                filler_comment = "You had some filler words in your speech. With practice, you can reduce these further."
            elif analysis['filler_percentage'] >= 7:
                filler_comment = "Your filler word usage was moderate. Keep practicing to improve."
            elif analysis['filler_percentage'] >= 3:
                filler_comment = "You did a good job minimizing filler words. Keep it up!"
            else:
                filler_comment = "Excellent work! You used almost no filler words in your speech."
                
            # Vocabulary diversity comment
            diversity_comment = ""
            ttr_level = analysis['ttr_analysis']['diversity_level']
            if ttr_level == "very high":
                diversity_comment = "Your vocabulary was exceptionally diverse and rich."
            elif ttr_level == "high":
                diversity_comment = "You used a strong variety of words throughout your speech."
            elif ttr_level == "average":
                diversity_comment = "Your vocabulary diversity was good, with room to incorporate more varied terms."
            elif ttr_level == "low":
                diversity_comment = "Consider expanding your vocabulary to make your speech more engaging."
            else:
                diversity_comment = "Try to use a wider range of words to enhance your speech's impact."
                
            # Logical flow comment removed
            flow_comment = ""
            flow_score = 0 # Dummy value to prevent errors
                
            # Category-specific advice based on mode
            category_specific_advice = ""
            if practice_category in category_default_feedback:
                if practice_category == "persuasive":
                    if flow_score < 60:
                        category_specific_advice = "For persuasive speaking, try to strengthen your logical flow with clearer transitions between arguments."
                    elif analysis['filler_percentage'] > 10:
                        category_specific_advice = "When persuading others, reducing filler words can make your arguments sound more authoritative."
                    else:
                        category_specific_advice = "Your persuasive speech is developing well. Continue using evidence and addressing counterarguments."
                
                elif practice_category == "emotive":
                    if ttr_level in ["low", "very low"]:
                        category_specific_advice = "For emotive speaking, try using more varied emotional vocabulary to express your feelings with greater nuance."
                    else:
                        category_specific_advice = "Your emotive speech conveys feelings well. Keep matching your tone to the emotions you're expressing."
                
                elif practice_category == "public-speaking":
                    if analysis['filler_percentage'] > 7:
                        category_specific_advice = "For public speaking, work on reducing filler words to sound more polished and confident on stage."
                    elif flow_score < 60:
                        category_specific_advice = "Public speaking benefits from clear structure. Try outlining your main points more clearly."
                    else:
                        category_specific_advice = "Your public speaking skills are developing well. Keep focusing on clear projection and structure."
                
                elif practice_category == "rizzing":
                    category_specific_advice = "In charismatic conversation, your natural flow is important. Stay authentic while working on smooth transitions."
                
                elif practice_category == "basic-conversations":
                    if analysis['filler_percentage'] > 15:
                        category_specific_advice = "Even in casual conversation, reducing filler words can help you sound more articulate."
                    else:
                        category_specific_advice = "Your casual conversation style is progressing well. Continue asking open-ended questions."
                
                elif practice_category == "formal-conversations":
                    if ttr_level in ["low", "very low"]:
                        category_specific_advice = "In formal settings, a more diverse vocabulary can enhance your professionalism."
                    else:
                        category_specific_advice = "Your formal communication style is developing appropriately. Maintain your concise and clear approach."
                
                elif practice_category == "debating":
                    if flow_score < 70:
                        category_specific_advice = "Debate requires strong logical flow. Focus on connecting your arguments more clearly."
                    else:
                        category_specific_advice = "Your debate skills show good logical reasoning. Continue supporting claims with evidence."
                
                elif practice_category == "storytelling":
                    if ttr_level in ["low", "very low"]:
                        category_specific_advice = "Storytelling benefits from rich, descriptive language. Try expanding your vocabulary."
                    elif flow_score < 60:
                        category_specific_advice = "Stories need a clear narrative arc. Work on transitions between parts of your story."
                    else:
                        category_specific_advice = "Your storytelling is developing well. Keep focusing on narrative structure and audience engagement."
            else:
                # Generic advice if category unknown
                category_specific_advice = "Continue practicing your speaking skills regularly to improve over time."
            
            # Create a cohesive summary with category-specific additions
            enhanced_text = f"""Here's a summary of your {practice_category or 'speech'} practice analysis.
            
            {filler_comment} {diversity_comment} {flow_comment}
            
            {category_specific_advice}
            
            You said {analysis['total_words']} total words, and about {analysis['filler_percentage']}% of them were filler words.
            
            Keep practicing to improve your {practice_category or 'speaking'} skills!"""
            
            # Update metrics
            metrics_data = {
                "total_words": analysis['total_words'],
                "unique_words": analysis['ttr_analysis']['unique_words'],
                "vocabulary_score": analysis['ttr_analysis']['ttr'],
                "filler_percentage": analysis['filler_percentage'],
                "practice_category": practice_category or "unknown"
            }
        elif practice_category in category_default_feedback:
            # If no analysis but we have a category, use the default feedback for that category
            enhanced_text = f"""Here's some feedback on your {practice_category} practice.
            
            {category_default_feedback[practice_category]}
            
            While I couldn't analyze the specific content of your speech, remember that regular practice is key to improvement.
            
            Keep practicing to enhance your {practice_category} skills!"""
        
        # Generate speech from enhanced text - THIS MUST NOT FAIL
        logger.info(f"Generating TTS with speed: {speech_speed}")
        
        # LOGGING PAYLOAD FOR DEBUGGING
        logger.info(f"Enhanced text length: {len(enhanced_text)}")
        logger.info(f"Enhanced text preview: {enhanced_text[:100]}...")
        
        # Robust check for empty or whitespace-only text
        if not enhanced_text or not str(enhanced_text).strip() or len(str(enhanced_text).strip()) < 2:
            logger.warning("Enhanced text is empty or too short! Using fallback.")
            enhanced_text = "Good job on your practice. Keep improving your skills."

            # Create a fresh client 
            sse = client.tts.SSEClient()
            
            # Create config
            tts_config = TTSConfig(speed=float(speech_speed))
            
            # Send request with timeout handling
            # Clean text of potentially problematic characters if needed, but UTF-8 should be fine.
            # Just ensure it's a string.
            if not isinstance(enhanced_text, str):
                enhanced_text = str(enhanced_text)

            # Save to temporary buffer
            temp_buffer = io.BytesIO()
            try:
                logger.info(f"Generating TTS for text: {enhanced_text[:50]}...")
                response = sse.send(enhanced_text, tts_config=tts_config)
                save_audio(response, temp_buffer)
            except Exception as tts_error:
                logger.error(f"TTS generation failed: {str(tts_error)}")
                # Fallback: If TTS fails, we still return the text but maybe no audio or a warning?
                # For now, let's just log it and NOT re-raise, effectively succeeding without audio or with empty audio.
                # But the client expects audio.
                # Let's try to return a silent audio or just ignore the audio part if possible? 
                # Or better, return a 200 but indicating failure in the response body if the client supports it.
                # Since the client likely checks for 200 OK, throwing 500 crashes the flow.
                # We will catch this specific 400 error from pyneuphonic.
                if "400" in str(tts_error):
                    logger.warning("TTS API rejected the text (400). Skipping audio generation.")
                # We return what we have (transcription + analysis), effectively bypassing the crash.
                pass
            temp_buffer.seek(0)
            
            # Verify audio was generated
            buffer_size = temp_buffer.getbuffer().nbytes
            logger.info(f"Generated audio size: {buffer_size} bytes")
            
            if buffer_size == 0:
                raise Exception("Generated empty audio file")
                
            # Return audio file with metrics in headers
            response = send_file(
                temp_buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='enhanced_speech.wav'
            )
            
            # Add analysis data to headers
            metrics_json = json.dumps(metrics_data)
            
            # Ensure CORS headers are set
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Content-Disposition, X-Speech-Metrics, X-Practice-Category'
            response.headers['X-Speech-Metrics'] = metrics_json
            response.headers['X-Practice-Category'] = practice_category or "unknown"
            
            logger.info("Successfully generated and returned enhanced audio")
            return response
            

        
    except Exception as e:
        logger.error(f"Speech Enhancement Error: {str(e)}")
        traceback.print_exc()
        error_response = jsonify({"error": f"Speech enhancement failed: {str(e)}"})
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response, 500

@app.route("/api/generate-suggestions", methods=['POST'])
def generate_suggestions():
    """Generate AI-powered personalized suggestions based on performance"""
    try:
        data = request.json
        
        # Validate input data
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        analysis = data.get('analysis', {})
        recording_analysis = data.get('recordingAnalysis', {})
        practice_mode = data.get('practiceMode', 'general')
        transcription = data.get('transcription', '')
        
        # Validate required fields
        if not analysis or not recording_analysis:
            return jsonify({"error": "Missing required analysis data"}), 400
        
        # Extract key metrics with safe defaults
        filler_pct = float(analysis.get('filler_percentage', 0))
        filler_count = int(analysis.get('filler_count', 0))
        total_words = int(analysis.get('total_words', 0))
        found_fillers = analysis.get('found_fillers', [])
        speaking_metrics = analysis.get('speaking_metrics', {})
        pauses = int(speaking_metrics.get('estimated_pauses', 0))
        confidence = int(speaking_metrics.get('confidence_score', 70))
        
        # Extract emotion and gaze data safely
        emotions = recording_analysis.get('emotions', [])
        gaze = recording_analysis.get('gaze', [])
        dominant_emotion = emotions[0].get('emotion', 'neutral') if emotions and len(emotions) > 0 else 'neutral'
        emotion_pct = emotions[0].get('percentage', '0') if emotions and len(emotions) > 0 else '0'
        primary_gaze = gaze[0].get('direction', 'center') if gaze and len(gaze) > 0 else 'center'
        gaze_pct = gaze[0].get('percentage', '0') if gaze and len(gaze) > 0 else '0'
        
        # Get top 3 fillers
        filler_breakdown = {}
        for filler in found_fillers:
            if isinstance(filler, str):  # Ensure filler is a string
                filler_breakdown[filler] = filler_breakdown.get(filler, 0) + 1
        top_fillers = sorted(filler_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
        top_fillers_str = ', '.join([f"'{f}' ({c}x)" for f, c in top_fillers]) if top_fillers else 'none'
        
        # Generate intelligent rule-based suggestions
        suggestions = generate_fallback_suggestions(
            filler_pct, filler_count, pauses, confidence,
            dominant_emotion, primary_gaze, practice_mode, top_fillers
        )
        
        return jsonify({
            "success": True,
            "suggestions": suggestions
        })
        
    except Exception as e:
        logger.error(f"Critical error in generate_suggestions: {e}")
        traceback.print_exc()
        # Return basic fallback even on critical errors
        return jsonify({
            "success": True,
            "suggestions": {
                "overall": {
                    "score": 70,
                    "summary": "Good effort! Keep practicing to improve.",
                    "nextSteps": ["Focus on reducing filler words", "Maintain eye contact"]
                }
            },
            "fallback": True,
            "error": "Unable to generate detailed suggestions"
        })


def generate_fallback_suggestions(filler_pct, filler_count, pauses, confidence, 
                                  dominant_emotion, primary_gaze, practice_mode, top_fillers):
    """Generate rule-based suggestions as fallback"""
    
    # Filler analysis
    if filler_pct < 3:
        filler_severity = "good"
        filler_tip = f"Excellent! Only {filler_pct:.1f}% filler words. You're speaking very cleanly."
        filler_action = "Maintain this level by staying mindful of your speech patterns."
    elif filler_pct < 7:
        filler_severity = "moderate"
        if top_fillers and len(top_fillers) > 0:
            top_filler = top_fillers[0][0]
            filler_count_top = top_fillers[0][1]
            filler_tip = f"You used '{top_filler}' {filler_count_top}x. Try to reduce to under 3%."
        else:
            filler_tip = f"You used filler words {filler_count} times. Try to reduce to under 3%."
        filler_action = "Pause for 1 second instead of saying filler words."
    else:
        filler_severity = "needs_work"
        filler_tip = f"{filler_pct:.1f}% fillers is high. Focus on eliminating them."
        filler_action = "Practice speaking slower and pausing between thoughts."
    
    # Pause analysis
    if pauses > 8:
        pause_severity = "good"
        pause_tip = f"Great use of {pauses} pauses! This helps emphasize key points."
        pause_action = "Keep using pauses strategically for impact."
    elif pauses >= 4:
        pause_severity = "moderate"
        pause_tip = f"You paused {pauses} times. More pauses can add emphasis."
        pause_action = "Try pausing after important statements."
    else:
        pause_severity = "needs_work"
        pause_tip = f"Only {pauses} pauses detected. You might be rushing."
        pause_action = "Practice taking deliberate 1-2 second pauses."
    
    # Emotion analysis
    emotion_map = {
        'persuasive': ('enthusiastic', 'happy'),
        'emotive': ('expressive', 'varied'),
        'public-speaking': ('confident', 'happy'),
        'formal': ('calm', 'neutral'),
        'storytelling': ('engaging', 'varied'),
        'debate': ('passionate', 'angry')
    }
    target_emotion = emotion_map.get(practice_mode, ('neutral', 'neutral'))[1]
    
    if dominant_emotion == target_emotion or dominant_emotion == 'happy':
        emotion_severity = "good"
        emotion_tip = f"Good emotional expression! You appeared {dominant_emotion}."
        emotion_action = "Keep this natural emotional delivery."
    elif dominant_emotion == 'neutral':
        emotion_severity = "needs_work"
        emotion_tip = f"You appeared mostly neutral. For {practice_mode}, show more emotion."
        emotion_action = "Try smiling and using hand gestures while speaking."
    else:
        emotion_severity = "moderate"
        emotion_tip = f"You showed {dominant_emotion} emotion. Consider if this matches your message."
        emotion_action = "Practice matching your facial expressions to your content."
    
    # Gaze analysis
    gaze_pct_num = float(gaze_pct) if isinstance(gaze_pct, str) else gaze_pct
    if primary_gaze == 'center' and gaze_pct_num > 60:
        gaze_severity = "good"
        gaze_tip = f"Excellent eye contact! You looked at the camera {gaze_pct}% of the time."
        gaze_action = "Maintain this strong eye contact in future sessions."
    elif gaze_pct_num >= 40:
        gaze_severity = "moderate"
        gaze_tip = f"You looked {primary_gaze} {gaze_pct}% of the time. Try centering your gaze more."
        gaze_action = "Imagine the camera is a friend and make eye contact."
    else:
        gaze_severity = "needs_work"
        gaze_tip = f"You looked {primary_gaze} often. This can seem uncertain."
        gaze_action = "Practice looking directly at the camera lens."
    
    # Overall
    if confidence >= 80:
        summary = "Excellent performance! You're speaking with strong confidence."
    elif confidence >= 60:
        summary = "Good job! A few improvements will boost your confidence."
    else:
        summary = "Keep practicing! Focus on the key areas below."
    
    next_steps = []
    if filler_severity != "good":
        next_steps.append("Reduce filler words by pausing instead")
    if gaze_severity != "good":
        next_steps.append("Improve eye contact with the camera")
    if not next_steps:
        next_steps = ["Keep practicing to maintain this level", "Try a harder practice mode"]
    
    return {
        "fillers": {
            "severity": filler_severity,
            "tip": filler_tip,
            "actionable": filler_action,
            "icon": "💭"
        },
        "pauses": {
            "severity": pause_severity,
            "tip": pause_tip,
            "actionable": pause_action,
            "icon": "⏸️"
        },
        "emotions": {
            "severity": emotion_severity,
            "tip": emotion_tip,
            "actionable": emotion_action,
            "icon": "😊"
        },
        "gaze": {
            "severity": gaze_severity,
            "tip": gaze_tip,
            "actionable": gaze_action,
            "icon": "👁️"
        },
        "overall": {
            "score": int(confidence),
            "summary": summary,
            "nextSteps": next_steps
        }
    }


# ============================================================================
# ANALYTICS ENDPOINTS - Session Tracking & Data Persistence
# ============================================================================

import uuid
from datetime import datetime

SESSIONS_FILE = os.path.join(script_dir, 'data', 'sessions.json')

def read_sessions():
    """Safely read sessions from JSON file with error recovery"""
    default_data = {"sessions": []}
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    
    data, error = safe_read_json(SESSIONS_FILE, default=default_data)
    
    if error:
        logger.warning(f"Using default sessions data due to error: {error}")
    
    # Validate structure
    if not isinstance(data, dict) or 'sessions' not in data:
        logger.warning("Invalid sessions data structure, using default")
        return default_data
    
    return data

def write_sessions(data):
    """Safely write sessions to JSON file with atomic operations"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    
    # Validate data structure
    if not isinstance(data, dict) or 'sessions' not in data:
        logger.error("Invalid data structure for sessions")
        return False
    
    success, error = safe_write_json(SESSIONS_FILE, data, create_backup=True)
    
    if error:
        logger.error(f"Failed to write sessions: {error}")
    
    return success

@app.route("/api/save-session", methods=['POST', 'OPTIONS'])
def save_session():
    """Save a practice session to the database"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ['analysis', 'recordingAnalysis']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create session object
        session = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "duration": data.get('duration', 0),
            "practiceMode": data.get('practiceMode', 'general'),
            "analysis": data.get('analysis', {}),
            "recordingAnalysis": data.get('recordingAnalysis', {}),
            "transcription": data.get('transcription', ''),
            "aiInsights": generate_ai_insights({
                "analysis": data.get('analysis', {}),
                "duration": data.get('duration', 0),
                "practiceMode": data.get('practiceMode', 'general')
            })
        }
        
        # Read existing sessions
        sessions_data = read_sessions()
        
        # Append new session
        sessions_data['sessions'].append(session)
        
        # Limit to last 100 sessions to prevent file bloat
        if len(sessions_data['sessions']) > 100:
            sessions_data['sessions'] = sessions_data['sessions'][-100:]
        
        # Write back to file
        if write_sessions(sessions_data):
            logger.info(f"Session saved successfully: {session['id']}")
            return jsonify({
                "success": True,
                "sessionId": session['id'],
                "message": "Session saved successfully"
            })
        else:
            return jsonify({"error": "Failed to save session"}), 500
            
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-analytics", methods=['GET', 'OPTIONS'])
def get_analytics():
    """Get aggregated analytics from all sessions"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        sessions_data = read_sessions()
        sessions = sessions_data.get('sessions', [])
        
        if not sessions:
            return jsonify({
                "success": True,
                "analytics": {
                    "totalSessions": 0,
                    "totalDuration": 0,
                    "averageEmotions": [],
                    "averageGazeDirections": [],
                    "averageFillerWords": 0,
                    "averageVocabularyScore": 0,
                    "averageLogicalFlow": 0,
                    "recentSessions": []
                }
            })
        
        # Aggregate data
        total_duration = 0
        emotion_counts = {}
        gaze_counts = {}
        filler_percentages = []
        vocab_scores = []
        flow_scores = []
        practice_mode_counts = {}
        
        for session in sessions:
            # Duration
            total_duration += session.get('duration', 0)
            
            # Practice modes
            practice_mode = session.get('practiceMode', 'general')
            if practice_mode in practice_mode_counts:
                practice_mode_counts[practice_mode] += 1
            else:
                practice_mode_counts[practice_mode] = 1
            
            # Emotions
            emotions = session.get('recordingAnalysis', {}).get('emotions', [])
            for emotion_entry in emotions:
                emotion = emotion_entry.get('emotion', 'neutral')
                percentage = float(emotion_entry.get('percentage', 0))
                if emotion in emotion_counts:
                    emotion_counts[emotion].append(percentage)
                else:
                    emotion_counts[emotion] = [percentage]
            
            # Gaze
            gaze = session.get('recordingAnalysis', {}).get('gaze', [])
            for gaze_entry in gaze:
                direction = gaze_entry.get('direction', 'center')
                percentage = float(gaze_entry.get('percentage', 0))
                if direction in gaze_counts:
                    gaze_counts[direction].append(percentage)
                else:
                    gaze_counts[direction] = [percentage]
            
            # Speech metrics
            analysis = session.get('analysis', {})
            if 'filler_percentage' in analysis:
                filler_percentages.append(float(analysis['filler_percentage']))
            
            speaking_metrics = analysis.get('speaking_metrics', {})
            if 'confidence_score' in speaking_metrics:
                vocab_scores.append(float(speaking_metrics['confidence_score']))
        
        # Calculate averages
        avg_emotions = [
            {
                "emotion": emotion,
                "percentage": sum(percentages) / len(percentages)
            }
            for emotion, percentages in emotion_counts.items()
        ]
        avg_emotions.sort(key=lambda x: x['percentage'], reverse=True)
        
        avg_gaze = [
            {
                "direction": direction,
                "percentage": sum(percentages) / len(percentages)
            }
            for direction, percentages in gaze_counts.items()
        ]
        avg_gaze.sort(key=lambda x: x['percentage'], reverse=True)
        
        avg_filler = sum(filler_percentages) / len(filler_percentages) if filler_percentages else 0
        avg_vocab = sum(vocab_scores) / len(vocab_scores) if vocab_scores else 0
        avg_flow = avg_vocab  # Using confidence as flow score
        
        # Get recent sessions (last 10)
        recent_sessions = [
            {
                "id": s['id'],
                "timestamp": s['timestamp'],
                "practiceMode": s.get('practiceMode', 'general'),
                "duration": s.get('duration', 0),
                "fillerPercentage": s.get('analysis', {}).get('filler_percentage', 0)
            }
            for s in sessions[-10:]
        ]
        recent_sessions.reverse()  # Most recent first
        
        # Practice modes breakdown
        practice_modes = [
            {
                "mode": mode,
                "count": count
            }
            for mode, count in practice_mode_counts.items()
        ]
        practice_modes.sort(key=lambda x: x['count'], reverse=True)
        
        analytics = {
            "totalSessions": len(sessions),
            "totalDuration": total_duration,
            "averageEmotions": avg_emotions[:5],  # Top 5 emotions
            "averageGazeDirections": avg_gaze,
            "averageFillerWords": round(avg_filler, 2),
            "averageVocabularyScore": round(avg_vocab, 2),
            "averageLogicalFlow": round(avg_flow, 2),
            "practiceModes": practice_modes,
            "recentSessions": recent_sessions
        }
        
        return jsonify({
            "success": True,
            "analytics": analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



def generate_ai_insights(session_data):
    """Generate dynamic AI-powered insights based on session performance"""
    try:
        import google.generativeai as genai
        
        # Get API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return ["Practice regularly to improve your speaking skills.", 
                    "Focus on reducing filler words for clearer communication.",
                    "Maintain steady pacing throughout your speech."]
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Extract metrics
        analysis = session_data.get('analysis', {})
        wpm = analysis.get('words_per_minute', 0)
        filler_pct = analysis.get('filler_percentage', 0)
        clarity = analysis.get('clarity_score', 0)
        duration = session_data.get('duration', 0)
        practice_mode = session_data.get('practiceMode', 'general')
        
        # Create prompt
        prompt = f"""You are a professional speech coach. Analyze this practice session and provide 3-5 specific, actionable insights.

Session Details:
- Practice Mode: {practice_mode}
- Duration: {duration} seconds
- Speaking Rate: {wpm} words per minute
- Filler Words: {filler_pct}%
- Clarity Score: {clarity}%

Provide insights in this format:
1. [Specific observation and actionable advice]
2. [Specific observation and actionable advice]
3. [Specific observation and actionable advice]

Keep each insight concise (1-2 sentences) and actionable. Focus on what they did well and what to improve."""

        response = model.generate_content(prompt)
        insights_text = response.text.strip()
        
        # Parse insights into list
        insights = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                clean_line = re.sub(r'^[\d\.\-\•\*]\s*', '', line)
                if clean_line:
                    insights.append(clean_line)
        
        return insights[:5] if insights else [
            "Great job completing this practice session!",
            "Keep practicing regularly to build confidence.",
            "Focus on maintaining a steady pace."
        ]
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        return [
            "Practice regularly to improve your speaking skills.",
            "Focus on reducing filler words for clearer communication.",
            "Maintain steady pacing throughout your speech."
        ]

@app.route("/api/reset-analytics", methods=['POST', 'OPTIONS'])
def reset_analytics():
    """Reset all analytics data"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Archive old data
        sessions_data = read_sessions()
        if sessions_data.get('sessions'):
            # Create archive directory
            archive_dir = os.path.join(os.path.dirname(SESSIONS_FILE), 'archive')
            os.makedirs(archive_dir, exist_ok=True)
            
            # Save archive with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_file = os.path.join(archive_dir, f'sessions_archive_{timestamp}.json')
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2)
        
        # Reset sessions
        write_sessions({"sessions": []})
        
        logger.info("Analytics data reset successfully")
        return jsonify({
            "success": True,
            "message": "Analytics data reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting analytics: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-insights", methods=['GET', 'OPTIONS'])
def get_insights():
    """Get enhanced insights with dynamic strengths and weaknesses"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        sessions_data = read_sessions()
        sessions = sessions_data.get('sessions', [])
        
        if not sessions:
            return jsonify({
                "success": True,
                "insights": {
                    "strengths": [],
                    "weaknesses": [],
                    "trends": {},
                    "recommendations": []
                }
            })
        
        # Calculate metrics
        total_sessions = len(sessions)
        wpms = []
        filler_pcts = []
        clarity_scores = []
        durations = []
        
        for session in sessions:
            analysis = session.get('analysis', {})
            wpms.append(analysis.get('words_per_minute', 0))
            filler_pcts.append(analysis.get('filler_percentage', 0))
            clarity_scores.append(analysis.get('clarity_score', 0))
            durations.append(session.get('duration', 0))
        
        avg_wpm = sum(wpms) / len(wpms) if wpms else 0
        avg_filler = sum(filler_pcts) / len(filler_pcts) if filler_pcts else 0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Determine strengths (metrics above threshold)
        strengths = []
        if avg_wpm >= 120 and avg_wpm <= 160:
            strengths.append("Excellent speaking pace - you maintain an ideal rhythm")
        elif avg_wpm > 90:
            strengths.append("Good speaking pace - clear and understandable")
        
        if avg_filler < 5:
            strengths.append("Minimal filler words - very articulate speech")
        elif avg_filler < 10:
            strengths.append("Low filler word usage - good fluency")
        
        if avg_clarity > 80:
            strengths.append("Exceptional clarity - very easy to understand")
        elif avg_clarity > 60:
            strengths.append("Good clarity - generally clear communication")
        
        if avg_duration > 60:
            strengths.append("Good session length - thorough practice")
        
        if total_sessions >= 10:
            strengths.append("Consistent practice - building strong habits")
        elif total_sessions >= 5:
            strengths.append("Regular practice - on the right track")
        
        # Determine weaknesses (areas to improve)
        weaknesses = []
        if avg_wpm < 90:
            weaknesses.append("Speaking pace is slow - try to speak more confidently")
        elif avg_wpm > 180:
            weaknesses.append("Speaking too fast - slow down for better clarity")
        
        if avg_filler > 15:
            weaknesses.append("High filler word usage - practice pausing instead")
        elif avg_filler > 10:
            weaknesses.append("Moderate filler words - work on reducing them")
        
        if avg_clarity < 60:
            weaknesses.append("Clarity needs improvement - focus on enunciation")
        elif avg_clarity < 80:
            weaknesses.append("Clarity could be better - practice clear pronunciation")
        
        if avg_duration < 30:
            weaknesses.append("Sessions are short - try longer practice sessions")
        
        # Calculate trends (last 5 vs previous sessions)
        trends = {}
        if len(sessions) >= 5:
            recent_sessions = sessions[-5:]
            older_sessions = sessions[:-5] if len(sessions) > 5 else sessions[:5]
            
            recent_wpm = sum(s.get('analysis', {}).get('words_per_minute', 0) for s in recent_sessions) / len(recent_sessions)
            older_wpm = sum(s.get('analysis', {}).get('words_per_minute', 0) for s in older_sessions) / len(older_sessions) if older_sessions else recent_wpm
            
            recent_filler = sum(s.get('analysis', {}).get('filler_percentage', 0) for s in recent_sessions) / len(recent_sessions)
            older_filler = sum(s.get('analysis', {}).get('filler_percentage', 0) for s in older_sessions) / len(older_sessions) if older_sessions else recent_filler
            
            trends['wpm_trend'] = 'improving' if recent_wpm > older_wpm else 'declining' if recent_wpm < older_wpm else 'stable'
            trends['filler_trend'] = 'improving' if recent_filler < older_filler else 'declining' if recent_filler > older_filler else 'stable'
        
        # Generate recommendations
        recommendations = []
        if avg_wpm < 120:
            recommendations.append("Try reading aloud to build speaking confidence and pace")
        if avg_filler > 10:
            recommendations.append("Practice pausing instead of using filler words")
        if avg_clarity < 70:
            recommendations.append("Record yourself and focus on clear enunciation")
        if total_sessions < 10:
            recommendations.append("Aim for daily practice to build consistency")
        
        return jsonify({
            "success": True,
            "insights": {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "trends": trends,
                "recommendations": recommendations,
                "metrics": {
                    "avgWpm": round(avg_wpm, 1),
                    "avgFiller": round(avg_filler, 1),
                    "avgClarity": round(avg_clarity, 1),
                    "avgDuration": round(avg_duration, 1),
                    "totalSessions": total_sessions
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        traceback.print_exc()
        return jsonify({" error": str(e)}), 500


if __name__ == "__main__":

    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, host='0.0.0.0', port=port)
