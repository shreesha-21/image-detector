# Base Image: Use a lightweight Python version
FROM python:3.10-slim

# System Dependencies: Required for YOLOv8 (OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \    
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# We use "--host 0.0.0.0" so the container is accessible from outside
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
