# FROM python:3.9


# WORKDIR /code


# COPY ./requirements.txt /code/requirements.txt


# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# COPY ./app /code/app

# EXPOSE 8080

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies - More comprehensive list
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libopencv-dev \
    python3-opencv \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/

# Create directories
RUN mkdir -p /app/model /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV EVA_MODEL_PATH=/app/model/emotion_model.pth
ENV PORT=8080
ENV DEBIAN_FRONTEND=noninteractive

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/', timeout=10)" || exit 1

# Run application
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 7200 --timeout-graceful-shutdown 300 --limit-concurrency 5 --limit-max-requests 100"]

