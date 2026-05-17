---
title: Vocalyst Backend
emoji: 🎤
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 5328
pinned: false
---

# Vocalyst Backend API

Flask backend for the [Vocalyst](https://github.com/Shreyyy07/Vocalyst) speech analysis application.

## Endpoints

- `GET /api/python` — Health check
- `POST /api/tts` — Text-to-speech
- `POST /api/detect-emotion` — Emotion detection from image
- `POST /api/transcribe` — Audio transcription via Whisper
- `GET /api/video-feed` — Live face landmark stream

## Stack

- Flask + Gunicorn
- MediaPipe (face landmarks)
- DeepFace + TensorFlow (emotion detection)
- OpenAI Whisper (transcription)
- ElevenLabs / Neuphonic (TTS)
