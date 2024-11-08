# Use the official TensorFlow GPU base image matching your TensorFlow version
FROM tensorflow/tensorflow:2.13.0-gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /usr/src/app

# Remove NVIDIA CUDA repositories to prevent apt-get update errors
RUN rm -f /etc/apt/sources.list.d/cuda.list \
          /etc/apt/sources.list.d/nvidia-ml.list

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install numba==0.56.4 && \
    pip list  # This will list installed packages for debugging

# Copy project files
COPY . .

# Create directories for models and logs (if not already present)
RUN mkdir -p data/models data/logs

# Define default command
CMD ["python", "training.py"]