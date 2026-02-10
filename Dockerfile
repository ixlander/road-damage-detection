# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY demo/ ./demo/
COPY src/ ./src/
COPY configs/ ./configs/
COPY datasets/ ./datasets/

# Create necessary directories
RUN mkdir -p data outputs runs

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api.main:APP", "--host", "0.0.0.0", "--port", "8000"]
