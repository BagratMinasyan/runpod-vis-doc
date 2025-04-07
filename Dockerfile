FROM python:3.10-slim

# Prevents prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install colpali-engine from GitHub
RUN pip install git+https://github.com/illuin-tech/colpali

# ✅ Copy all Python files (rp_handler.py, inference.py, model_loader.py)
COPY . ./

# Run the worker
CMD ["python3", "-u", "rp_handler.py"]
