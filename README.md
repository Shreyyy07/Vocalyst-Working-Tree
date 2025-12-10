<div align="center">

# 🎙️ Voclayst

### AI-Powered Communication Coaching Platform

*Elevate your public speaking and presentation skills through intelligent multimodal analysis*

[![GitHub Stars](https://img.shields.io/github/stars/Shreyyy07/Vocalyst-Main? style=social)](https://github.com/Shreyyy07/Vocalyst-Main)
[![GitHub Forks](https://img.shields.io/github/forks/Shreyyy07/Vocalyst-Main?style=social)](https://github.com/Shreyyy07/Vocalyst-Main/fork)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Technologies](#-technologies-used) • [Roadmap](#-roadmap)

</div>

---

## 📖 About Voclayst

**Voclayst** is an advanced AI-driven communication coaching platform that revolutionizes how individuals improve their public speaking and presentation skills.  Unlike traditional tools that focus solely on text analysis, Voclayst provides comprehensive, real-time multimodal feedback by analyzing: 

- 🗣️ **Voice Analysis** - Speech fluency, pacing, tone, and delivery
- 😊 **Facial Expressions** - Emotional engagement and confidence levels
- 📝 **Content Structure** - Logical coherence, vocabulary, and engagement

Communication is more than just words—it's about how you sound, how you look, and how you structure your message. Voclayst bridges the gap left by conventional tools like ChatGPT or YouTube tutorials by offering a unified, intelligent solution for holistic communication improvement.

---

## ✨ Features

### 🎯 Core Capabilities

#### **1. Textual Intelligence**
- **Speech Transcription & Analysis** using NLP techniques
- **Coherence Detection** with RoBERTa (large) models
- **Vocabulary Sophistication** tracking and lexical density analysis
- **Filler Word Detection** (um, uh, like, etc.) with frequency metrics
- **Engagement Scoring** based on content structure and variety

#### **2. Visual Analysis**
- **Eye Contact Tracking** via MediaPipe and GazeTracking
- **Facial Microexpression Detection** using advanced CV techniques
- **Emotional Tone Recognition** to assess confidence and engagement
- **Engagement Estimation** through facial cues and body language
- **Real-time Visual Feedback** during presentations

#### **3. Audio Analysis**
- **Speech Fluency Classification** with XGBoost (93% F1 score)
- **Words Per Minute (WPM)** measurement
- **Zero Crossing Rate (ZCR)** analysis
- **MFCC (Mel-frequency cepstral coefficients)** feature extraction
- **Awkward Pause Detection** and tone shift identification
- **Enhanced Speech Processing** with Neuphonic integration

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VOCLAYST PLATFORM                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌───────▼────────┐
            │   Frontend      │       │    Backend     │
            │  (Next.js)      │◄─────►│   (Flask)      │
            │   React UI      │  REST │   Python API   │
            └─────────────────┘  API  └────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
            ┌───────▼────────┐       ┌───────▼────────┐       ┌───────▼────────┐
            │   TEXT MODULE   │       │  AUDIO MODULE  │       │  VISUAL MODULE │
            └─────────────────┘       └────────────────┘       └────────────────┘
                    │                         │                         │
         ┌──────────┴──────────┐   ┌─────────┴─────────┐    ┌─────────┴─────────┐
         │                     │   │                   │    │                   │
    ┌────▼────┐          ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
    │ RoBERTa │          │  NLTK   │            │ Librosa │            │MediaPipe│
    │ (Logic) │          │(Filler) │            │ (MFCC)  │            │  (Gaze) │
    └─────────┘          └─────────┘            └─────────┘            └─────────┘
         │                     │                      │                      │
    ┌────▼────┐          ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
    │Coherence│          │Lexical  │            │ XGBoost │            │ Facial  │
    │Analysis │          │Density  │            │(Fluency)│            │Expression│
    └─────────┘          └─────────┘            └─────────┘            └─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌───────▼────────┐
            │  FEEDBACK       │       │   ANALYTICS    │
            │  GENERATION     │       │   DASHBOARD    │
            └─────────────────┘       └────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  PERSONALIZED INSIGHTS  │
                    │   & RECOMMENDATIONS     │
                    └─────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Next.js, React, Tailwind CSS | User interface and real-time feedback display |
| **Backend API** | Flask, Python | Request handling and model orchestration |
| **Text Analysis** | RoBERTa, NLTK | Logical coherence and vocabulary analysis |
| **Audio Processing** | Librosa, XGBoost, Neuphonic | Speech fluency and acoustic feature extraction |
| **Visual Analysis** | MediaPipe, OpenCV, FER | Eye tracking and facial expression detection |
| **ML Pipeline** | PyTorch, TensorFlow, scikit-learn | Model training and inference |

---

## 🚀 Installation

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.9 or higher)
- **pip** (Python package manager)
- **npm** or **yarn** (Node package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shreyyy07/Vocalyst-Main.git
   cd Vocalyst-Main
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

4. **Download required ML models**
   ```bash
   npm run download-models
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

#### Development Mode

**Option 1: Run both servers concurrently**
```bash
npm run dev
```

**Option 2: Run servers separately**

Terminal 1 (Frontend):
```bash
npm run next-dev
```

Terminal 2 (Backend):
```bash
npm run flask-dev
```

#### Production Mode

```bash
npm run build
npm start
```

The application will be available at: 
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5328

---

## 💻 Usage

### Starting a Practice Session

1. **Navigate to the Practice Module**
   - Open the Voclayst web application
   - Click on "Start Practice Session"

2. **Record Your Presentation**
   - Allow camera and microphone permissions
   - Click "Start Recording" when ready
   - Speak naturally while the system captures multimodal data

3. **Receive Real-time Feedback**
   - Monitor live metrics during your session
   - Track eye contact, speech pace, and engagement

4. **Review Detailed Analytics**
   - Access comprehensive post-session analysis
   - View scores for fluency, coherence, and expression
   - Get personalized improvement recommendations

### Features Available

- **📊 Analytics Dashboard** - View historical performance trends
- **🎯 Interview Practice** - Simulate job interview scenarios
- **👁️ Eye Tracking** - Monitor eye contact patterns
- **🎬 Session Recordings** - Review past presentations
- **📈 Progress Tracking** - Track improvement over time

---

## 🛠️ Technologies Used

### Frontend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **React. js** | 18.3.1 | UI component library |
| **Next.js** | 14.2.25 | React framework with SSR |
| **TypeScript** | 5.3.3 | Type-safe JavaScript |
| **Tailwind CSS** | 3.4.1 | Utility-first CSS framework |
| **Framer Motion** | 12.5.0 | Animation library |
| **Recharts** | 2.15.3 | Data visualization |
| **Lucide React** | 0.483.0 | Icon library |

### Backend Stack

| Technology | Purpose |
|------------|---------|
| **Flask** | Python web framework |
| **Flask-CORS** | Cross-origin resource sharing |
| **Gunicorn** | WSGI HTTP server |

### AI/ML Models

| Model | F1 Score | Purpose |
|-------|----------|---------|
| **RoBERTa (large)** | - | Logical flow and coherence detection |
| **XGBoost** | 93% | Speech fluency classification |
| **SVM** | - | Multi-class classification tasks |
| **MLP + Embeddings** | - | Tonality and emotion prediction |

### Audio Processing

- **Librosa** - Audio feature extraction (WPM, ZCR, MFCC)
- **Neuphonic** - Enhanced speech signal analysis
- **OpenAI Whisper** - Speech-to-text transcription
- **SoundDevice** - Real-time audio capture

### Computer Vision

- **MediaPipe** - Face landmark detection and tracking
- **OpenCV** - Video processing and frame extraction
- **FER (Facial Expression Recognition)** - Emotion detection
- **GazeTracking** - Eye contact estimation
- **DeepFace** - Advanced facial analysis

### NLP & Text Processing

- **NLTK** - Natural language processing toolkit
- **Transformers (HuggingFace)** - Pre-trained language models
- **Tokenizers** - Text tokenization
- **SentencePiece** - Unsupervised text tokenizer

### Deep Learning Frameworks

- **PyTorch** - Neural network training and inference
- **TensorFlow** - ML model development
- **Keras** - High-level neural networks API
- **scikit-learn** - Machine learning utilities

---

## 📁 Project Structure

```
Vocalyst-Main/
├── api/                          # Backend Flask API
│   ├── index.py                  # Main API endpoints
│   ├── tonality.py              # Tonality analysis module
│   └── *. task/*. onnx            # Pre-trained model files
│
├── app/                         # Next.js frontend application
│   ├── analytics/               # Analytics dashboard pages
│   ├── camera/                  # Camera capture components
│   ├── eye-tracking/           # Eye tracking interface
│   ├── interview/              # Interview practice mode
│   ├── practice/               # General practice mode
│   ├── recordings/             # Session recordings management
│   ├── transcribe/             # Transcription services
│   ├── tts/                    # Text-to-speech integration
│   ├── page.tsx                # Landing page
│   └── globals.css             # Global styles
│
├── components/                  # Reusable React components
│   └── ui/                     # UI component library
│
├── experiments_models/          # ML model experiments
│   ├── texts/                  # Text analysis models
│   │   └── Engagement Level/   # Engagement detection
│   └── voice/                  # Voice analysis models
│
├── public/                     # Static assets
├── scripts/                    # Utility scripts
├── lib/                       # Helper functions and utilities
│
├── requirements.txt           # Python dependencies
├── package.json              # Node.js dependencies
├── next. config.js           # Next.js configuration
├── tailwind.config.ts       # Tailwind CSS configuration
└── README.md               # Project documentation
```

---

## 🎯 How It Works

### 1️⃣ Data Capture
The system captures multimodal data streams:
- **Video Feed**:  Real-time camera input for facial analysis
- **Audio Stream**:  Microphone input for speech processing
- **Transcription**: Speech-to-text conversion using Whisper

### 2️⃣ Feature Extraction

**Text Features:**
- Lexical density and vocabulary sophistication
- Filler word frequency and distribution
- Sentence structure and coherence

**Audio Features:**
- MFCC coefficients for acoustic analysis
- Zero-crossing rate for speech clarity
- Energy and pitch contours for tonality

**Visual Features:**
- Facial landmarks (68-point detection)
- Eye gaze direction and fixation duration
- Microexpressions and emotion classification

### 3️⃣ AI Analysis

**Text Processing:**
- RoBERTa analyzes logical flow and argument structure
- NLTK processes vocabulary complexity
- Custom algorithms detect engagement patterns

**Audio Processing:**
- XGBoost classifier predicts fluency levels
- Librosa extracts acoustic features
- Neuphonic enhances speech signal quality

**Visual Processing:**
- MediaPipe tracks facial landmarks in real-time
- DeepFace analyzes emotional expressions
- Custom algorithms calculate eye contact percentage

### 4️⃣ Feedback Generation
The system synthesizes insights from all modalities to provide:
- **Real-time Metrics**: Live display during presentation
- **Post-Session Analysis**: Comprehensive breakdown with scores
- **Personalized Recommendations**:  Targeted improvement suggestions
- **Historical Trends**: Progress tracking over time

---

## 📊 Performance Metrics

| Model | Task | Accuracy/F1 Score |
|-------|------|-------------------|
| XGBoost Classifier | Speech Fluency | 93% F1 Score |
| RoBERTa | Coherence Detection | High Performance |
| MediaPipe | Face Landmark Detection | Real-time (<30ms) |
| Whisper | Speech Transcription | State-of-the-art |

---

## 🗺️ Roadmap

### 🚀 Upcoming Features

- [ ] **Multilingual Support** - Expand to support 20+ languages for speech and text analysis
- [ ] **Enhanced Emotion Detection** - Refine sentiment analysis with contextual understanding
- [ ] **Real-Time Fluency Feedback** - Instant suggestions during live presentations
- [ ] **Context-Aware Coherence** - Improved logical flow detection with domain-specific models
- [ ] **Comprehensive Analytics** - Advanced performance tracking and insights dashboard
- [ ] **Mobile Application** - iOS and Android apps for on-the-go practice
- [ ] **AI Coach Assistant** - Conversational AI for personalized coaching sessions
- [ ] **Team Collaboration** - Multi-user sessions and peer feedback
- [ ] **Integration APIs** - Connect with video conferencing platforms (Zoom, Teams)
- [ ] **Custom Training Modules** - Industry-specific presentation templates

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git fork https://github.com/Shreyyy07/Vocalyst-Main. git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

5. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Describe your changes in detail
   - Reference any related issues
   - Wait for review and feedback

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/TypeScript
- Write meaningful commit messages
- Add comments for complex logic
- Update documentation as needed

---

## 🐛 Bug Reports & Feature Requests

Found a bug or have a feature suggestion? Please open an issue on GitHub:

- **Bug Reports**: Include steps to reproduce, expected vs actual behavior
- **Feature Requests**: Describe the feature and its use case
- **Questions**: Use GitHub Discussions for general questions

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Shreyyy07** - [GitHub Profile](https://github.com/Shreyyy07)

---

## 🙏 Acknowledgments

- **MediaPipe** team for facial landmark detection
- **HuggingFace** for transformer models
- **OpenAI** for Whisper speech recognition
- **Neuphonic** for enhanced audio processing
- **Edlitera** for microexpression detection techniques

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Shreyyy07/Vocalyst-Main/issues)
- **Email**: [Your Email]
- **Documentation**: [Full Documentation](#) *(Coming Soon)*

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Shreyyy07

</div>
