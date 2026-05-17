<div align="center">
  <img src="https://via.placeholder.com/150?text=Vocalyst+Logo" alt="Vocalyst Logo" width="120" />
  <h1>🎙️ Vocalyst AI</h1>
  <p><strong>Your Intelligent AI Communication & Interview Coach</strong></p>

  <p>
    <a href="#live-demo"><img src="https://img.shields.io/badge/Deployed-Vercel-black?style=for-the-badge&logo=vercel" alt="Vercel" /></a>
    <a href="#live-demo"><img src="https://img.shields.io/badge/Backend-Hugging_Face-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face" /></a>
    <img src="https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js" alt="Next.js" />
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python" />
  </p>
</div>

---

## 🌟 About The Project

**Vocalyst AI** is an advanced, full-stack AI application designed to help users master their communication, presentation, and interview skills. By analyzing your real-time video feed and audio input, Vocalyst provides instant feedback on your facial expressions, speech tone, and eye contact.

Whether you're preparing for a high-stakes interview or just want to improve your public speaking confidence, Vocalyst acts as your personal, highly intelligent AI coach.

<div align="center">
  <!-- TODO: Replace with an actual GIF of the dashboard -->
  <img src="https://via.placeholder.com/800x400?text=Insert+Dashboard+GIF+Here" alt="Vocalyst Dashboard Demo" />
</div>

---

## 🚀 Live Demo

The application is deployed with a split-architecture for maximum performance:
* **Frontend:** Hosted on [Vercel](https://vocalyst.vercel.app/) *(Insert your actual link here)*
* **ML Backend:** Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Shreyyy07/vocalyst-backend)

👉 **[Try Vocalyst AI Live](#)** *(Add your Vercel URL here)*

*(Note: The AI backend goes to sleep when unused. The first request may take ~60 seconds to wake the server up.)*

---

## ✨ Key Features

### 👁️ Real-Time Emotion & Eye Tracking
Vocalyst utilizes **DeepFace** and **MediaPipe** to analyze your facial expressions and eye movements through your webcam. It calculates confidence levels and tracks exactly where you are looking during your speech.
<div align="center">
  <!-- TODO: Replace with an actual GIF of the eye tracking/emotion feature -->
  <img src="https://via.placeholder.com/800x300?text=Insert+Eye+Tracking+GIF+Here" alt="Eye Tracking Demo" />
</div>

### 🗣️ Whisper Speech-to-Text
Powered by **OpenAI Whisper**, Vocalyst perfectly transcribes your spoken words in real-time, allowing it to analyze your vocabulary, pacing, and filler word usage.

### 🤖 Instant AI Feedback & TTS
The system uses advanced LLMs to evaluate your performance and provides verbal feedback using ultra-realistic **Neuphonic** Text-to-Speech voices.
<div align="center">
  <!-- TODO: Replace with an actual GIF of the feedback feature -->
  <img src="https://via.placeholder.com/800x300?text=Insert+Feedback+GIF+Here" alt="Feedback Demo" />
</div>

### 📊 Comprehensive Analytics Dashboard
Track your progress over time with a dedicated dashboard. Review past sessions, analyze your core metrics (Confidence, Clarity, Eye Contact), and see visual graphs of your improvement.

---

## 🏗️ Architecture & Tech Stack

Vocalyst was built with a highly optimized, decoupled architecture separating the fast UI from the heavy Machine Learning processing.

### Frontend (Next.js & Vercel)
* **Framework:** Next.js 14, React
* **Styling:** TailwindCSS
* **Deployment:** Vercel

### Backend (Python & Hugging Face Spaces)
* **API Framework:** Flask, Gunicorn
* **Containerization:** Custom Docker Environment (Debian-based)
* **Machine Learning:**
  * `PyTorch` & `TensorFlow` (Core ML)
  * `DeepFace` (Emotion Recognition)
  * `OpenAI Whisper` (Transcription)
  * `MediaPipe` & `OpenCV` (Computer Vision & Face Mesh)
  * `PyNeuphonic` (Real-time TTS)
* **Deployment:** Hugging Face Spaces (Docker)

---

## 💻 Local Installation

If you want to run this project locally on your machine:

### Prerequisites
* Node.js (v18+)
* Python (3.11+)
* FFmpeg installed on your system

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/Vocalyst-Working-Tree.git
cd Vocalyst-Working-Tree
```

### 2. Setup the Backend (Python)
```bash
# Create a virtual environment
python -m venv venv311

# Activate it (Windows)
.\venv311\Scripts\activate
# Activate it (Mac/Linux)
source venv311/bin/activate

# Install heavy ML requirements
pip install -r api/requirements.txt
```

### 3. Setup the Frontend (Next.js)
```bash
# Open a new terminal tab
npm install
```

### 4. Environment Variables
Create a `.env` file in the root directory based on `.env.example`:
```env
# AI Keys
GEMINI_API_KEY=your_gemini_key
NEUPHONIC_API_KEY=your_neuphonic_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Routing (Leave as localhost for local development)
NEXT_PUBLIC_API_URL=http://localhost:5328
```

### 5. Run the Application
```bash
# This script concurrently starts both Next.js and Flask
npm run dev
```

---

## 👨‍💻 Developer

Built by **[Your Name]** 
* 🔗 [LinkedIn](#)
* 🌐 [Portfolio](#)
* ✉️ [Email](mailto:you@example.com)

<div align="center">
  <sub>Built with ❤️ and Artificial Intelligence.</sub>
</div>
