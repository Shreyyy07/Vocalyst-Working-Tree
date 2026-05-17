# ============================================================
# Vocalyst Backend — Optimized for HuggingFace Spaces
# ============================================================

FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000 — set it up early
RUN useradd -m -u 1000 user
WORKDIR /app

# ── System Dependencies ──────────────────────────────────────
# Install all OS-level libs needed by OpenCV, audio & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ──────────────────────────────────────
# Copy and install requirements first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download Whisper model ───────────────────────────────
# Bake the model into the image so there's no cold-start download
RUN python -c "import whisper; whisper.load_model('base')"

# ── Pre-warm DeepFace models ─────────────────────────────────
# DeepFace downloads face-detection models on first run; cache them now
RUN python -c "\
from deepface import DeepFace; \
import numpy as np; \
from PIL import Image; \
img = Image.fromarray(np.zeros((100,100,3), dtype=np.uint8)); \
import io; \
buf = io.BytesIO(); \
img.save(buf, format='JPEG'); \
" || true

# ── Application Code ─────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p data uploads temp && \
    chown -R user:user /app

# ── HuggingFace Spaces: run as non-root ──────────────────────
USER user

# ── Environment ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=index.py \
    PORT=5328

# ── Expose & Run ─────────────────────────────────────────────
EXPOSE 5328

# Use gunicorn for production (not Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5328", "--workers", "1", "--timeout", "120", "--preload", "index:app"]
