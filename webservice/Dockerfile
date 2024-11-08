# webservice/Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app/webservice

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from both webservice and training
COPY webservice/requirements.txt ./webservice-requirements.txt
COPY training/requirements.txt ./training-requirements.txt

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r webservice-requirements.txt \
    && pip install -r training-requirements.txt