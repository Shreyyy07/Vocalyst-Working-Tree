# ============================================================
# Vocalyst Backend — Optimized for HuggingFace Spaces
# ============================================================

FROM python:3.11-slim

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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Application Code ─────────────────────────────────────────
COPY . .

# Create runtime directories with global permissions so HF Spaces 
# non-root user can write to them.
RUN mkdir -p data uploads temp && \
    chmod -R 777 /app

# ── Environment ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=index.py \
    PORT=5328 \
    MPLCONFIGDIR=/app/temp \
    XDG_CACHE_HOME=/app/temp

# ── Expose & Run ─────────────────────────────────────────────
EXPOSE 5328

CMD ["gunicorn", "--bind", "0.0.0.0:5328", "--workers", "1", "--timeout", "120", "--preload", "index:app"]
