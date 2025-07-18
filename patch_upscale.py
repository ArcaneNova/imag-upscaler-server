#!/usr/bin/env python3
"""
Real-ESRGAN Import Direct Patch

This is a direct replacement patch for the upscale.py file
that fixes the RealESRGAN import issue. This script can be run
in the production environment to patch the file in-place.

Usage:
  python3 patch_upscale.py

The script will create a backup of the original file and then
apply the patch.
"""

import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realesrgan-patch")

# Path to the upscale.py file
UPSCALE_FILE = "app/upscale.py"

# The fixed content for upscale.py
FIXED_CONTENT = """from PIL import Image, ImageFilter, ImageEnhance
import torch
import cv2
import numpy as np
import os
import logging
import gc
import psutil
import tempfile
from typing import Optional, Tuple, Union
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torch.hub import download_url_to_file

logger = logging.getLogger(__name__)

# Global model instances for reuse
_models = {}
_device = None

def get_optimal_device():
    """Determine the best available device"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            _device = "cpu"
            logger.info("Using CPU for inference")
    return _device

def get_model(scale: int = 2):
    """Get or initialize the Real-ESRGAN model with caching"""
    global _models
    
    model_key = f"realesrgan_{scale}x"
    
    if model_key not in _models:
        try:
            device = get_optimal_device()
            
            # Create model architecture based on scale
            if scale == 2:
                model_name = "RealESRGAN_x2plus"
                netscale = 2
            elif scale == 4:
                model_name = "RealESRGAN_x4plus"
                netscale = 4
            else:
                raise ValueError(f"Unsupported scale: {scale}")
                
            # Define models directory in the current working directory
            models_dir = os.path.join(os.getcwd(), 'weights')
            os.makedirs(models_dir, exist_ok=True)
            
            # Initialize RealESRGANer with the correct parameters
            model = RealESRGANer(
                scale=netscale,
                model_path=os.path.join(models_dir, f'{model_name}.pth'),
                dni_weight=None,
                model=None,  # Let the library create the appropriate model
                half=device == 'cuda',  # Use half precision for CUDA
                tile=512,    # Tile size for processing large images
                tile_pad=10, # Padding for tiles to avoid seams
                pre_pad=0,   # No pre-padding needed
                device=device
            )
            
            # Ensure the model file exists
            if not os.path.exists(model.model_path):
                logger.info(f"Downloading model weights for {model_name}")
                
                # Model URLs based on the official repo
                model_urls = {
                    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                }
                
                # Download weights if needed
                if model_name in model_urls:
                    load_file_from_url(
                        url=model_urls[model_name],
                        model_dir=models_dir,
                        progress=True,
                        file_name=f'{model_name}.pth'
                    )
                
            _models[model_key] = model
            logger.info(f"Real-ESRGAN {scale}x model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise
            
    return _models[model_key]

def preprocess_image(image: Image.Image, max_dimension: int = 2048) -> Image.Image:
    """Preprocess image for optimal upscaling"""
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
        logger.debug(f"Converted image from {image.mode} to RGB")
    
    # Check and resize if too large
    original_size = image.size
    if max(image.size) > max_dimension:
        # Calculate new size maintaining aspect ratio
        ratio = max_dimension / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {original_size} to {image.size}")
    
    return image

def enhance_face_regions(image: Image.Image) -> Image.Image:
    """Apply additional face enhancement if requested"""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            logger.info(f"Detected {len(faces)} face(s) for enhancement")
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = cv_image[y:y+h, x:x+w]
                
                # Apply additional sharpening to face
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(face_region, -1, kernel)
                
                # Blend original and sharpened (70% sharpened, 30% original)
                blended = cv2.addWeighted(face_region, 0.3, sharpened, 0.7, 0)
                
                # Replace face region
                cv_image[y:y+h, x:x+w] = blended
            
            # Convert back to PIL
            enhanced_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            return enhanced_image
        else:
            logger.info("No faces detected for enhancement")
            return image
            
    except Exception as e:
        logger.warning(f"Face enhancement failed, using original: {e}")
        return image

def postprocess_image(image: Image.Image, face_enhance: bool = False) -> Image.Image:
    """Apply post-processing enhancements"""
    
    try:
        if face_enhance:
            image = enhance_face_regions(image)
        
        # Apply subtle sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        
        # Slight color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)  # 5% color boost
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.02)  # 2% contrast boost
        
        return image
        
    except Exception as e:
        logger.warning(f"Post-processing failed, using original: {e}")
        return image

def run_upscale(
    input_path: str, 
    output_path: str, 
    scale: int = 2, 
    face_enhance: bool = False,
    max_dimension: int = 2048
):
    """
    Enhanced upscale function with optimizations
    
    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscaling factor (2 or 4)
        face_enhance: Apply additional face enhancement
        max_dimension: Maximum input dimension before resizing
    """
    
    start_memory = psutil.virtual_memory().percent
    logger.info(f"Starting upscale: scale={scale}x, face_enhance={face_enhance}, memory={start_memory}%")
    
    try:
        # Load image using cv2 for RealESRGANer compatibility
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {input_path}")
            
        original_height, original_width = img.shape[:2]
        logger.info(f"Input image: {original_width}x{original_height}")
        
        # Check and resize if too large
        if max(original_width, original_height) > max_dimension:
            ratio = max_dimension / max(original_width, original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        
        # Get model and perform upscaling
        model = get_model(scale)
        
        # Monitor memory usage
        before_upscale_memory = psutil.virtual_memory().percent
        
        # Perform upscaling
        logger.info(f"Starting Real-ESRGAN inference on {get_optimal_device()}")
        
        # Use RealESRGANer's enhance method which returns a NumPy array
        output, _ = model.enhance(img, outscale=scale)
        
        after_upscale_memory = psutil.virtual_memory().percent
        logger.info(f"Upscaling completed. Memory: {before_upscale_memory}% -> {after_upscale_memory}%")
        
        # Convert to PIL for post-processing
        if output.shape[2] == 3:
            # BGR to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            upscaled_img = Image.fromarray(output_rgb)
        else:
            # BGRA to RGBA
            output_rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
            upscaled_img = Image.fromarray(output_rgba)
        
        # Apply post-processing
        if face_enhance or scale == 4:  # Always post-process 4x images
            upscaled_img = postprocess_image(upscaled_img, face_enhance)
        
        # Save result with optimization
        save_kwargs = {
            "format": "PNG",
            "optimize": True,
            "compress_level": 6  # Good compression without quality loss
        }
        
        # For very large images, use JPEG with high quality
        if upscaled_img.size[0] * upscaled_img.size[1] > 16_000_000:  # > 16MP
            output_path = output_path.replace('.png', '.jpg')
            save_kwargs = {
                "format": "JPEG",
                "quality": 95,
                "optimize": True
            }
            logger.info("Using JPEG format for large output image")
        
        upscaled_img.save(output_path, **save_kwargs)
        
        final_size = upscaled_img.size
        final_memory = psutil.virtual_memory().percent
        
        logger.info(f"Image saved: {original_width}x{original_height} -> {final_size[0]}x{final_size[1]}")
        logger.info(f"Memory usage: {start_memory}% -> {final_memory}%")
        
        # Force garbage collection
        del upscaled_img, img, output
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        
        # Cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        raise

def clear_model_cache():
    """Clear all cached models to free memory"""
    global _models
    
    logger.info("Clearing model cache")
    
    for model_key in list(_models.keys()):
        del _models[model_key]
    
    _models.clear()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model cache cleared")

def get_model_info():
    """Get information about loaded models"""
    device = get_optimal_device()
    loaded_models = list(_models.keys())
    
    info = {
        "device": device,
        "loaded_models": loaded_models,
        "cuda_available": torch.cuda.is_available(),
        "memory_usage": f"{psutil.virtual_memory().percent:.1f}%"
    }
    
    if torch.cuda.is_available():
        info["gpu_memory"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return info
"""

def patch_requirements():
    """Update requirements.txt file"""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        logger.warning(f"{req_file} not found, skipping requirements update")
        return False
    
    with open(req_file, "r") as f:
        content = f.read()
    
    # Check if the required dependencies are already present
    required_deps = ["basicsr>=1.4.2", "facexlib>=0.2.5", "gfpgan>=1.3.8"]
    missing_deps = [dep for dep in required_deps if dep not in content]
    
    if not missing_deps:
        logger.info("Requirements are already up to date")
        return True
    
    # Update the requirements
    with open(req_file, "a") as f:
        f.write("\n# Added by Real-ESRGAN patch\n")
        for dep in missing_deps:
            f.write(f"{dep}\n")
    
    logger.info(f"Updated {req_file} with missing dependencies")
    return True

def patch_upscale():
    """Patch the upscale.py file"""
    if not os.path.exists(UPSCALE_FILE):
        logger.error(f"{UPSCALE_FILE} not found")
        return False
    
    # Create a backup of the original file
    backup_file = f"{UPSCALE_FILE}.bak.{int(time.time())}"
    shutil.copy2(UPSCALE_FILE, backup_file)
    logger.info(f"Created backup: {backup_file}")
    
    # Write the fixed content
    with open(UPSCALE_FILE, "w") as f:
        f.write(FIXED_CONTENT)
    
    logger.info(f"Patched {UPSCALE_FILE} with fixed implementation")
    return True

def main():
    import time
    logger.info("Starting Real-ESRGAN direct patch")
    
    try:
        # Update requirements
        patch_requirements()
        
        # Patch upscale.py
        if patch_upscale():
            logger.info("Patch applied successfully!")
            logger.info("Please restart the API server to apply the changes.")
        else:
            logger.error("Failed to apply patch")
    
    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
