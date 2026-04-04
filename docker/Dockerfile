FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    unzip \
    blender \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libgles2-mesa \
    libegl1-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure libGLESv2 symlink exists for MediaPipe
RUN ldconfig && \
    find / -name "libGLESv2*" 2>/dev/null || \
    apt-get update && apt-get install -y libgles-dev && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Rhubarb Lip Sync
RUN wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    -O /tmp/rhubarb.zip \
    && mkdir -p /opt/rhubarb \
    && cd /opt/rhubarb && unzip /tmp/rhubarb.zip \
    && rm /tmp/rhubarb.zip \
    && ln -s /opt/rhubarb/rhubarb /usr/local/bin/rhubarb

# Install Piper TTS
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz \
    && mkdir -p /opt/piper \
    && tar -xzf /tmp/piper.tar.gz -C /opt/piper \
    && rm /tmp/piper.tar.gz \
    && ln -s /opt/piper/piper/piper /usr/local/bin/piper

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/workspace /app/avatars

EXPOSE 7860

CMD ["python", "src/ui/app.py"]
