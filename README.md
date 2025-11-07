# ComfyUI Docker

A containerized version of ComfyUI with pre-downloaded AI models for stable diffusion image generation.

## Quick Start

### Build the Docker Image
```bash
docker build -t comfyui:latest .
```

### Run the container
```
 docker run -d \
  --name comfyui \
  --gpus all \
  -p 8188:8188 \
  -v $(pwd)/output:/app/comfyui/output \
  -v $(pwd)/input:/app/comfyui/input \
  comfyui:latest 
   ```

## Access ComfyUI

Open your browser and navigate to: http://localhost:8188
Requirements

* Docker with NVIDIA GPU support
* NVIDIA drivers
* ~50GB disk space for models
* 16GB+ RAM recommended

### Included Models

* Checkpoints: SD 1.5, SD 2.1, SDXL Base & Refiner
* ControlNet: Canny, Depth
* VAE: ft-ema
* LoRA: LCM-LoRA-SDXL
* Upscale: RealESRGAN x2/x4

### Volume Mounts

* /app/comfyui/output - Generated images
* /app/comfyui/input - Input images for img2img
* /app/comfyui/models - Model files (optional, to persist/share models)

### Environment Variables

* CUDA_VISIBLE_DEVICES - Select GPU (default: 0)
* COMFYUI_PORT - Change port (default: 8188)

### Troubleshooting
Out of Memory
Reduce batch size or image resolution in ComfyUI settings.
Container won't start
Check GPU drivers: nvidia-smi
Models not downloading
Check disk space and internet connection. Models are large (30-50GB total).
License
This project uses ComfyUI and various AI models. Please check their respective licenses.