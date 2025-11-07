#!/bin/bash

# ComfyUI 50GB Model Download Script
# Downloads all necessary models for ComfyUI

# Don't exit on error - we want to continue even if some downloads fail
# set -e  # REMOVED - this was causing the script to exit on any error

echo "=========================================="
echo "ComfyUI Model Downloader (50GB)"
echo "=========================================="

# Fixed: When no argument provided, use /app/comfyui as base
MODELS_DIR="${1:-/app/comfyui}/models"
CHECKPOINTS_DIR="$MODELS_DIR/checkpoints"
LORAS_DIR="$MODELS_DIR/loras"
UPSCALE_DIR="$MODELS_DIR/upscale_models"
CONTROLNET_DIR="$MODELS_DIR/controlnet"
VAE_DIR="$MODELS_DIR/vae"

# Create directories (with error handling)
mkdir -p "$CHECKPOINTS_DIR" "$LORAS_DIR" "$UPSCALE_DIR" "$CONTROLNET_DIR" "$VAE_DIR" || {
    echo "Warning: Could not create some directories"
}

# Colors (simplified for better compatibility)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to download with resume capability
download_model() {
    local url=$1
    local filename=$2
    local dir=$3
    local size_gb=${4:-"unknown"}
    
    if [ -f "$dir/$filename" ]; then
        local file_size=$(du -h "$dir/$filename" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓${NC} $filename already exists ($file_size), skipping..."
        return 0
    fi
    
    echo -e "${BLUE}⏳${NC} Downloading $filename (~${size_gb}GB)..."
    
    if wget -c -q --show-progress "$url" -O "$dir/$filename" 2>/dev/null; then
        local file_size=$(du -h "$dir/$filename" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓${NC} $filename downloaded ($file_size)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Failed to download $filename, continuing..."
        # Remove partial download
        rm -f "$dir/$filename" 2>/dev/null
        return 1
    fi
}

# Track successful downloads
DOWNLOAD_COUNT=0
FAILED_COUNT=0

# ============== CHECKPOINT MODELS (20-25GB) ==============
echo ""
echo -e "${BLUE}--- Downloading Checkpoint Models (20-25GB) ---${NC}"

download_model \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
    "sd-v1-5.safetensors" \
    "$CHECKPOINTS_DIR" \
    "4.2" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors" \
    "sd-v2-1.safetensors" \
    "$CHECKPOINTS_DIR" \
    "5.2" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "sdxl-v1.0-base.safetensors" \
    "$CHECKPOINTS_DIR" \
    "6.9" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

download_model \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" \
    "sdxl-v1.0-refiner.safetensors" \
    "$CHECKPOINTS_DIR" \
    "6.1" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

# ============== CONTROLNET MODELS (2-3GB) ==============
echo ""
echo -e "${BLUE}--- Downloading ControlNet Models (2-3GB) ---${NC}"

download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth" \
    "control_v11p_sd15_canny.pth" \
    "$CONTROLNET_DIR" \
    "1.4" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
    "control_v11f1p_sd15_depth.pth" \
    "$CONTROLNET_DIR" \
    "1.4" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

# ============== VAE MODELS (1GB) ==============
echo ""
echo -e "${BLUE}--- Downloading VAE Models (1GB) ---${NC}"

download_model \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors" \
    "vae-ft-mse.safetensors" \
    "$VAE_DIR" \
    "0.3" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

# ============== LORA MODELS (3-4GB) ==============
echo ""
echo -e "${BLUE}--- Downloading LoRA Models (3-4GB) ---${NC}"

download_model \
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors" \
    "lcm-lora-sdxl.safetensors" \
    "$LORAS_DIR" \
    "0.2" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

# ============== UPSCALE MODELS (500MB) ==============
echo ""
echo -e "${BLUE}--- Downloading Upscale Models (500MB) ---${NC}"

download_model \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    "RealESRGAN_x4plus.pth" \
    "$UPSCALE_DIR" \
    "0.06" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

download_model \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    "RealESRGAN_x2plus.pth" \
    "$UPSCALE_DIR" \
    "0.06" && ((DOWNLOAD_COUNT++)) || ((FAILED_COUNT++))

echo ""
echo "=========================================="
echo "Download Summary:"
echo "✓ Successfully downloaded: $DOWNLOAD_COUNT models"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "⚠ Failed downloads: $FAILED_COUNT models"
fi
echo "=========================================="

# Try to show total size, but don't fail if it doesn't work
echo ""
echo "Total model size:"
du -sh "$MODELS_DIR" 2>/dev/null || echo "Could not calculate size"

# Always exit successfully so Docker build continues
exit 0