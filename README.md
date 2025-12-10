# **Voclayst - AI-powered Communication Coaching Platform**

**Voclayst** is an AI-driven communication coaching platform designed to enhance public speaking and presentation skills through multimodal analysis. It evaluates speech delivery, facial expressions, and content structure, offering personalized feedback to improve fluency, confidence, and clarity.

---

## 📑 **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Technologies Used](#technologies-used)
5. [Usage](#usage)
6. [What's Next for Voclayst?](#whats-next-for-voclayst)
7. [Contributing](#contributing)
8. [License](#license)

---

## 📝 **Introduction**

Communication isn't just about words; it's about tone, emotion, pacing, expressions, and structure. Current tools like ChatGPT or YouTube tutorials focus only on text, missing out on how you sound or look. **Voclayst** fills this gap by providing a unified solution that gives real-time, multimodal feedback—covering voice, face, and text. It helps users improve public speaking and presentation skills by capturing video inputs and analyzing speech delivery, facial expressions, and content structure.

---

## ⚙️ **Features**

- **Multimodal Feedback**: Combines voice, facial expressions, and text to offer real-time personalized feedback.
- **Textual Intelligence**: Analyzes speech logic, coherence, vocabulary variety, and engagement using NLP.
- **Visual Analysis**: Tracks eye contact, facial expressions, and emotional tone to evaluate confidence and engagement.
- **Audio Analysis**: Measures speech fluency, flags filler words, awkward pauses, and tone shifts.

---

## 🔍 **How It Works**

### **1. Textual Intelligence**  
Transcribes and analyzes speech logic, evaluating coherence, vocabulary variety, and engagement using **NLP**.

### **2. Visual Analysis**  
Tracks **eye contact**, facial expressions, and engagement. Detects emotional tone and confidence levels.

### **3. Audio Analysis**  
Measures **speech fluency** using metrics like **WPM** (Words Per Minute), **ZCR** (Zero Crossing Rate), and **MFCC** (Mel-frequency cepstral coefficients). Flags filler words, awkward pauses, and tone shifts.

---

## 🧑‍💻 **Technologies Used**

- **Frontend**: 
  - **React.js + Next.js**: For a seamless and dynamic UI.
  - **Tailwind CSS**: Stylish and responsive design.

- **Backend**: 
  - **Python + Flask**: Fast and flexible backend API.
  - **REST API**: For model and data integration.

- **AI/ML**:
  - **RoBERTa (large)**: For logical flow and coherence detection.
  - **XGBoost (XGB)**: For fluency classification (Randomized CV model, 93% F1 score).
  - **SVM**: For classification tasks.
  - **Torch**: For model development.
  - **MLP + Embeddings**: For tonality prediction.

- **Audio Signal Processing**:
  - **Librosa**: For feature extraction (WPM, ZCR, MFCC).
  - **Neuphonic**: For enhanced speech signal analysis.
  
- **Natural Language Processing**:
  - **NLTK**: For lexical density, filler word detection, and vocabulary sophistication.
  - **re (regex)**: For text cleaning and filler word frequency analysis.

- **Computer Vision & Expression Analysis**:
  - **GazeTracking** (via MediaPipe): Eye contact and engagement estimation.
  - **Facial Microexpression Detection**: Based on **Edlitera** techniques.
  - **OpenCV**: For frame extraction and video preprocessing.


🚀 **Usage**

Open the Voclayst web app in your browser.

Click the "Start Recording" button to begin a session.

Record your speech while the system captures and analyzes your emotions, expressions, and speech.

After the session, receive detailed feedback on your communication skills.

🔮 What's Next for Voclayst?
Multilingual Speech & Text Support: Expanding accessibility for diverse linguistic backgrounds.

Enhanced Emotion & Engagement Detection: Refining sentiment analysis and listener interest prediction for more accurate insights.

Real-Time Fluency Feedback: Providing instant analysis of speech fluency with actionable recommendations.

Context-Aware Coherence Evaluation: Improving logical flow detection with more robust reasoning models.

Comprehensive Communication Analytics: Introducing detailed performance tracking and insights for continuous improvement.

🤝 **Contributing**

We welcome contributions! If you'd like to contribute to Voclayst, follow these steps:

Fork the repository.

Create a new branch.

Make your changes.

Test your code.

Submit a pull request.

📝 **License**

This project is licensed under the MIT License. See the LICENSE file for details.


