# Use an official NVIDIA CUDA base image (CUDA 12.8.0, cuDNN 9 runtime) with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set environment variables to ensure Python uses UTF-8 and is unbuffered
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED 1

# Install Python 3.12, pip, and required system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY docker_requirements.txt .

# Install the CUDA-enabled version of PyTorch first
# This must match the CUDA version from the base image (12.8.0)
RUN python3.12 -m pip install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the dependencies from your requirements file
# Note: Ensure torch, torchvision, and torchaudio are REMOVED from requirements.txt
RUN python3.12 -m pip install --no-cache-dir --break-system-packages -r docker_requirements.txt

# Copy your entire application code into the current working directory (/app)
COPY ./app .

# Expose the port the container will run on
EXPOSE 8080

# Command to run the application using the installed python3.11
CMD ["python3.12", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]