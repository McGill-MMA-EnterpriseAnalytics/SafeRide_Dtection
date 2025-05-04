FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY scripts/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy scripts
COPY scripts/*.py /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# Create directories
RUN mkdir -p /app/data /app/models /app/mlruns /app/runs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Default command to run training
ENTRYPOINT ["python", "/app/scripts/run_training.py"]