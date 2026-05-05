# ─── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps required by OpenCV and deep-sort-realtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir gunicorn \
 && pip install --no-cache-dir -r requirements.txt

# ─── App stage ────────────────────────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p temp_uploads output_views models

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
 && chown -R appuser:appgroup /app
USER appuser

# Gunicorn: 1 worker (avoids multiple RabbitMQ consumers), 4 threads for concurrency
# The YOLO model singleton is per-process, so 1 worker is intentional
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "gunicorn run:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300 --access-logfile - --error-logfile -"]
