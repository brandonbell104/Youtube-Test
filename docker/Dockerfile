FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    git \
    wget \
    curl \
    unzip \
    xz-utils \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgles2-mesa \
    libegl1-mesa \
    libegl-mesa0 \
    libglvnd0 \
    libglx0 \
    libglx-mesa0 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install Blender 4.2 LTS (apt only has 3.0 which is too old)
RUN wget -q https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz \
    -O /tmp/blender.tar.xz \
    && tar -xf /tmp/blender.tar.xz -C /opt/ \
    && ln -sf /opt/blender-4.2.0-linux-x64/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

# Install Rhubarb Lip Sync
RUN wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    -O /tmp/rhubarb.zip \
    && mkdir -p /opt/rhubarb \
    && cd /opt/rhubarb && unzip /tmp/rhubarb.zip \
    && rm /tmp/rhubarb.zip \
    && find /opt/rhubarb -name "rhubarb" -type f -executable -exec ln -sf {} /usr/local/bin/rhubarb \; \
    && rhubarb --version || echo "Rhubarb installed at: $(which rhubarb 2>/dev/null || find /opt/rhubarb -name rhubarb)"

# Install Piper TTS
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz \
    && mkdir -p /opt/piper \
    && tar -xzf /tmp/piper.tar.gz -C /opt/piper \
    && rm /tmp/piper.tar.gz \
    && ln -s /opt/piper/piper/piper /usr/local/bin/piper

# Create app directory
WORKDIR /app

# Install Python dependencies (main pipeline env)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install GVHMR (3D human pose tracker, SMPL output)
# GVHMR produces world-grounded SMPL parameters from monocular video.
# Repo: https://github.com/zju3dv/GVHMR
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /opt/gvhmr \
    && cd /opt/gvhmr \
    && pip install --no-cache-dir -r requirements.txt || true \
    && pip install --no-cache-dir -e . || true

ENV GVHMR_ROOT=/opt/gvhmr
ENV PYTHONPATH=/opt/gvhmr:/app:$PYTHONPATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/workspace /app/avatars /app/models/smpl /app/models/gvhmr

EXPOSE 7860

CMD ["python", "src/ui/app.py"]
