# ============================================================================
# Multi-stage build for ComfyUI 50GB Docker Image
# Optimized for Runpod serverless deployment
# ============================================================================

# Stage 1: Base image with CUDA and dependencies
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Install PyTorch and ML dependencies
# ============================================================================
FROM base AS ml-base

# Install PyTorch with CUDA support (large, takes up ~2-3GB)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
COPY requirements-ml.txt /tmp/
RUN pip install -r /tmp/requirements-ml.txt && rm /tmp/requirements-ml.txt

# ============================================================================
# Stage 3: Clone ComfyUI and install dependencies
# ============================================================================
FROM ml-base AS comfyui-base

WORKDIR /app

# Clone ComfyUI repository
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/comfyui

WORKDIR /app/comfyui

# Install ComfyUI specific requirements
RUN pip install -r requirements.txt

# Install additional ComfyUI dependencies for better compatibility
RUN pip install \
    kornia==0.7.1 \
    insightface==0.7.3 \
    onnxruntime==1.16.3 \
    imageio==2.33.1 \
    imageio-ffmpeg==0.4.9 \
    opencv-contrib-python==4.8.1.78 \
    psd-tools==1.9.33

# Create necessary directories
RUN mkdir -p models/checkpoints \
    && mkdir -p models/loras \
    && mkdir -p models/embeddings \
    && mkdir -p models/upscale_models \
    && mkdir -p models/controlnet \
    && mkdir -p models/vae \
    && mkdir -p models/gligen \
    && mkdir -p custom_nodes \
    && mkdir -p input \
    && mkdir -p output \
    && mkdir -p /models_cache

# ============================================================================
# Stage 4: Download Large AI Models (30-35GB)
# ============================================================================
FROM comfyui-base AS model-downloader

WORKDIR /app/comfyui

# Create a script to download models efficiently
RUN cat > /download_models.sh << 'MODELSCRIPT'
#!/bin/bash
set -e

echo "=========================================="
echo "Downloading ComfyUI Models (30-35GB)"
echo "=========================================="

CHECKPOINTS_DIR="/app/comfyui/models/checkpoints"
LORAS_DIR="/app/comfyui/models/loras"
UPSCALE_DIR="/app/comfyui/models/upscale_models"
CONTROLNET_DIR="/app/comfyui/models/controlnet"
VAE_DIR="/app/comfyui/models/vae"

# Function to download with retry
download_model() {
    local url=$1
    local filename=$2
    local dir=$3

    if [ -f "$dir/$filename" ]; then
        echo "✓ $filename already exists, skipping..."
        return
    fi
    
    echo "⏳ Downloading $filename..."
    wget -q --show-progress "$url" -O "$dir/$filename" || {
        echo "✗ Failed to download $filename"
        return 1
    }
    echo "✓ $filename downloaded ($(du -h "$dir/$filename" | cut -f1))"
}

# ============== CHECKPOINT MODELS (Main models for image generation) ==============
echo ""
echo "--- Downloading Checkpoint Models (20-25GB) ---"

# Stable Diffusion v1.5 (4.2GB)
download_model \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
    "sd-v1-5.safetensors" \
    "$CHECKPOINTS_DIR"

# Stable Diffusion v2.1 (5.2GB)
download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors" \
    "sd-v2-1.safetensors" \
    "$CHECKPOINTS_DIR"

# SDXL Base (6.9GB)
download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "sdxl-v1.0-base.safetensors" \
    "$CHECKPOINTS_DIR"

# SDXL Refiner (2.7GB)
download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" \
    "sdxl-v1.0-refiner.safetensors" \
    "$CHECKPOINTS_DIR"

# ControlNet Models (used for better control) - ~2GB
echo ""
echo "--- Downloading ControlNet Models (2-3GB) ---"

download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.safetensors" \
    "control_v11p_sd15_canny.safetensors" \
    "$CONTROLNET_DIR"

download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_depth.safetensors" \
    "control_v11p_sd15_depth.safetensors" \
    "$CONTROLNET_DIR"

# VAE Models (alternative VAE encoders) - ~1GB
echo ""
echo "--- Downloading VAE Models (1GB) ---"

download_model \
    "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-original.safetensors" \
    "vae-ft-ema.safetensors" \
    "$VAE_DIR"

# ============== LORA MODELS (Fine-tuning models for specific styles) ==============
echo ""
echo "--- Downloading LoRA Models (3-4GB) ---"

# Popular style LoRAs
download_model \
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors" \
    "lcm-lora-sdxl.safetensors" \
    "$LORAS_DIR"

# ============== UPSCALE MODELS (For image upscaling) ==============
echo ""
echo "--- Downloading Upscale Models (500MB) ---"

download_model \
    "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth" \
    "RealESRGAN_x4plus.pth" \
    "$UPSCALE_DIR"

download_model \
    "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth" \
    "RealESRGAN_x2plus.pth" \
    "$UPSCALE_DIR"

echo ""
echo "=========================================="
echo "✓ All models downloaded successfully!"
echo "=========================================="

# Print total size
echo ""
echo "Total model size:"
du -sh /app/comfyui/models/

MODELSCRIPT

RUN chmod +x /download_models.sh

# Download all models
RUN bash /download_models.sh

# Verify downloads
RUN du -sh /app/comfyui/models/checkpoints/ || true
RUN du -sh /app/comfyui/models/ || true


# ============================================================================
# Stage 5: Install Custom Nodes (Additional functionality)
# ============================================================================
FROM model-downloader AS custom-nodes

WORKDIR /app/comfyui/custom_nodes

# Install popular custom nodes for ComfyUI
RUN echo "Installing custom nodes..." && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git || true && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git || true && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git || true && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git || true

# ============================================================================
# Stage 6: Install Runpod Handler and Final Setup
# ============================================================================
FROM custom-nodes AS final

WORKDIR /app

# Install Runpod SDK
RUN pip install runpod==0.10.0

# Copy handler script from build context
COPY handler.py /app/handler.py

# Copy entrypoint script
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "ComfyUI 50GB Docker Image"
echo "=========================================="
echo "CUDA Version: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader || echo 'N/A')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'N/A')"
echo "=========================================="

# Check if running on Runpod
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Running on Runpod Serverless"
    echo "Pod ID: $RUNPOD_POD_ID"
    
    # Start Runpod handler
    cd /app
    python handler.py
else
    echo "Running in local/standard environment"
    
    # Start ComfyUI web interface
    cd /app/comfyui
    python main.py --listen 0.0.0.0 --port 8188
fi
EOF

RUN chmod +x /entrypoint.sh

# Set working directory
WORKDIR /app/comfyui

# Expose port
EXPOSE 8188 8000

# Labels for documentation
LABEL maintainer="Your Name"
LABEL description="ComfyUI 50GB Docker Image for Runpod Serverless"
LABEL version="1.0.0"
LABEL size="50GB"

# Cleanup to save space (only remove build artifacts, not models)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Entry point
ENTRYPOINT ["/entrypoint.sh"]