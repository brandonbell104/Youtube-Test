FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda

# System dependencies (includes build-essential for cython_bbox compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
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

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3.11 -m pip install --upgrade pip

# Install Blender 4.2 LTS
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

WORKDIR /app

# ── Step 1: Install PyTorch first (large download, cached as its own layer) ──
RUN pip install --no-cache-dir \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 2: Install our pipeline's own requirements ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 3: Clone GVHMR ──
RUN git clone --depth 1 https://github.com/zju3dv/GVHMR.git /opt/gvhmr

# ── Step 4: Install GVHMR deps ──
# Exclude: pytorch3d (we provide a shim), chumpy (won't build on 3.11),
#          numpy pinned version (let pip resolve a compatible one)
RUN cd /opt/gvhmr \
    && grep -v -E 'pytorch3d|chumpy|numpy==' requirements.txt > /tmp/gvhmr_reqs.txt \
    && pip install --no-cache-dir -r /tmp/gvhmr_reqs.txt \
    && pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir yacs gdown iopath

# ── Step 5: Create pytorch3d transforms shim ──
# GVHMR uses pytorch3d.transforms for rotation conversions. The official
# pytorch3d wheel only supports Python 3.10, and building from source takes
# 30+ minutes. This shim reimplements the ~8 functions GVHMR actually calls.
# Install to both dist-packages (for our code) and /opt/gvhmr (on PYTHONPATH,
# so GVHMR's subprocess can find it too).
COPY docker/pytorch3d_shim/pytorch3d /usr/local/lib/python3.11/dist-packages/pytorch3d
COPY docker/pytorch3d_shim/pytorch3d /opt/gvhmr/pytorch3d

# ── Step 6: Verify critical imports ──
# Lightning uses pkg_resources which lives in setuptools (not installed by default on 3.11)
RUN pip install --no-cache-dir 'setuptools<71' \
    && python -c "import torch; print(f'torch {torch.__version__}')" \
    && python -c "from pytorch3d.transforms import quaternion_to_matrix; print('pytorch3d shim OK')" \
    && python -c "import smplx; print('smplx OK')" \
    && python -c "import lightning; print(f'lightning {lightning.__version__}')"

ENV GVHMR_ROOT=/opt/gvhmr
ENV PYTHONPATH=/opt/gvhmr:/app:$PYTHONPATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/workspace /app/avatars /app/models/smpl /app/models/gvhmr

EXPOSE 7860

CMD ["python", "src/ui/app.py"]
