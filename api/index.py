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
from pyneuphonic import Neuphonic, save_audio, TTSConfig
import io
from dotenv import load_dotenv
import traceback
import whisper  # Add import for Whisper
import numpy as np
import base64
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
    "origins": ["http://localhost:3000"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Accept"],
    "supports_credentials": True,
    "expose_headers": ["Content-Type", "Content-Disposition"]
}})

# Add after the imports
FILLER_WORDS = [
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
]

def ngrams(words, n):
    output = []
    for i in range(len(words) - n + 1):
        output.append(' '.join(words[i:i + n]))
    return output

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
    
    # Get logical flow score
    try:
        logical_score = logical_flow(text)
        # Convert logical score to percentage and determine emoji
        logical_percentage = logical_score * 100
        if logical_percentage >= 80:
            logical_emoji = "🌠"  # Excellent flow
        elif logical_percentage >= 60:
            logical_emoji = "🌊"  # Good flow
        elif logical_percentage >= 40:
            logical_emoji = "🔄"  # Average flow
        elif logical_percentage >= 20:
            logical_emoji = "🌫️"  # Needs improvement
        else:
            logical_emoji = "🌪️"  # Poor flow
    except Exception as e:
        print(f"Error calculating logical flow: {str(e)}")
        logical_percentage = 0
        logical_emoji = "❓"
    
    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_percentage": round(filler_percentage, 2),
        "found_fillers": found_fillers,
        "filler_emoji": emoji,
        "ttr_analysis": ttr_analysis,
        "logical_flow": {
            "score": round(logical_percentage, 2),
            "emoji": logical_emoji
        }
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

def logical_flow(text):
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'logical_model.pk')
        print(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return 0.0
        
        # Check PyTorch version
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check transformers version
        import transformers
        print(f"Transformers version: {transformers.__version__}")
            
        with open(model_path, 'rb') as f:
            try:
                logical_model = pkl.load(f)
                print("Successfully loaded logical flow model")
            except RuntimeError as e:
                if "register_pytree_node()" in str(e):
                    print("Version mismatch detected between PyTorch and transformers")
                    print("Please ensure compatible versions are installed")
                    return 0.0
                raise
        
        print(f"Making prediction for text of length: {len(text)}")
        pred: list[dict] = logical_model.predict(text)
        score: float = pred[0]['score']
        print(f"Logical flow prediction result: {score}")
        
        return score
    except Exception as e:
        print(f"Error in logical flow prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
        return 0.0  # Return 0 as fallback to indicate failure

def detect_emotions(image: Image) -> dict:
    """
    Detects faces and emotions in an image using DeepFace.
    Returns normalized emotion scores and cropped face image.
    """
    # Convert PIL Image to numpy array for DeepFace
    image_arr = np.array(image)
    
    try:
        # Detect emotion using DeepFace
        try:
            # Wrap DeepFace to prevent crashing on OpenCV assertion errors
            result = DeepFace.analyze(image_arr, 
                                    actions=['emotion'], 
                                    enforce_detection=False)
        except Exception as e:
            logger.warning(f"DeepFace analysis failed for frame: {str(e)}")
            # Provide fallback/empty result so stream continues
            result = [{'dominant_emotion': 'neutral', 'emotion': {'neutral': 100}}]

        
        # Handle multiple faces by taking first one
        if isinstance(result, list):
            result = result[0]
        
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
                _whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
    
    return _whisper_model

@app.route("/api/python")
def hello_world():
    """Simple health check endpoint"""
    return "Hello, World!"

@app.route("/api/tts", methods=['POST'])
def text_to_speech():
    """Converts text to speech using Neuphonic API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        speed = max(0.7, min(2.0, float(data.get('speed', 1.0))))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"Processing TTS request - text: {text}, voice: {voice}, speed: {speed}")

        # Get SSE client and configure it
        # test
        api_key = os.environ.get('NEUPHONIC_API_KEY')
        if not api_key:
            raise ValueError("NEUPHONIC_API_KEY not found in environment variables")
        client = Neuphonic(api_key=api_key)
        #testend
        sse = client.tts.SSEClient()
        sse.speed = speed
        response = sse.send(text)
        
        # Save audio to temporary buffer
        temp_file = io.BytesIO()
        save_audio(response, temp_file)
        
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
        result = whisper_model.transcribe(audio_path, language='en', fp16=False)
        transcribed_text = result.get("text", "").strip()
        print(f"Whisper result object keys: {result.keys()}")
        print(f"Raw transcription: '{transcribed_text}'")
        print(f"Transcription length: {len(transcribed_text)}")
        
        if not transcribed_text:
            print("WARNING: Whisper returned empty transcription!")
            print(f"Segments: {result.get('segments', [])}")
        
        return transcribed_text
        
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

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

        # Skip FFmpeg conversion - transcribe the input file directly
        # Whisper can handle webm/mp4/wav files directly
        print(f"Transcribing input file directly: {temp_input}")
        text = transcribe_long_audio(temp_input)
        print(f"Transcription result: '{text}' (length: {len(text) if text else 0})")
        
        if not text:
            print("WARNING: Transcription returned empty text")
            text = " "  # Use a space to prevent downstream failures, or handle empty text in analysis
            # We don't raise Exception here to allow the UI to receive a response
        
        
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
        
        # Check for API key first thing
        api_key = os.environ.get('NEUPHONIC_API_KEY')
        if not api_key:
            logger.error("NEUPHONIC_API_KEY not found in environment variables")
            return jsonify({"error": "TTS API key not configured"}), 500
        
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
                
            # Logical flow comment
            flow_comment = ""
            flow_score = analysis['logical_flow']['score']
            if flow_score >= 80:
                flow_comment = "Your ideas flowed together excellently, creating a cohesive narrative."
            elif flow_score >= 60:
                flow_comment = "Your speech had good logical progression between points."
            elif flow_score >= 40:
                flow_comment = "The logical flow was adequate, but could use stronger transitions between ideas."
            elif flow_score >= 20:
                flow_comment = "Work on strengthening the connections between your points for better flow."
            else:
                flow_comment = "Focus on organizing your thoughts more logically when speaking."
                
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
                "logical_flow_score": analysis['logical_flow']['score'],
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

        try:
            # Create a fresh client 
            sse = client.tts.SSEClient()
            sse.speed = speech_speed
            
            # Send request with timeout handling
            # Clean text of potentially problematic characters if needed, but UTF-8 should be fine.
            # Just ensure it's a string.
            if not isinstance(enhanced_text, str):
                enhanced_text = str(enhanced_text)

            response = sse.send(enhanced_text)
            
            # Save to temporary buffer
            temp_buffer = io.BytesIO()
            save_audio(response, temp_buffer)
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
            
        except Exception as tts_error:
            logger.error(f"TTS generation error: {str(tts_error)}")
            traceback.print_exc()
            # If it's a 400 error, it might be the content.
            if "400" in str(tts_error):
                logger.error("Got 400 error. Text payload might be invalid.")
            return jsonify({"error": f"Speech generation failed: {str(tts_error)}"}), 500
        
    except Exception as e:
        logger.error(f"Speech Enhancement Error: {str(e)}")
        traceback.print_exc()
        error_response = jsonify({"error": f"Speech enhancement failed: {str(e)}"})
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response, 500

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, host='0.0.0.0', port=port)