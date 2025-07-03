#!/usr/bin/env python3
"""
Real-ESRGAN Import Hotfix

This script provides a direct hot-fix for the RealESRGAN import error
by creating a custom implementation that's compatible with the existing code.
It can be run directly in the deployed environment without redeploying.

Usage:
  python3 hotfix.py

The script will automatically create the necessary files and fix the imports.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realesrgan-hotfix")

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        logger.info(f"Created directory: {path}")

def create_realesrgan_adapter():
    """Create a RealESRGAN adapter class that uses RealESRGANer internally"""
    
    # Create the adapter file in the app directory
    adapter_file = os.path.join("app", "realesrgan_adapter.py")
    ensure_dir("app")
    
    adapter_code = """
# RealESRGAN adapter - provides compatibility layer
from realesrgan import RealESRGANer
import torch
import os
import cv2
import numpy as np
from PIL import Image
import logging
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

logger = logging.getLogger(__name__)

class RealESRGAN:
    \"\"\"Compatibility adapter for RealESRGANer\"\"\"
    
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = None
        self.weights_loaded = False
        
    def load_weights(self, weights_path, download=True):
        \"\"\"Load model weights\"\"\"
        # Ensure weights directory exists
        weights_dir = os.path.dirname(weights_path)
        os.makedirs(weights_dir, exist_ok=True)
        
        # Determine model parameters based on scale
        if self.scale == 2:
            model_name = "RealESRGAN_x2plus"
        elif self.scale == 4:
            model_name = "RealESRGAN_x4plus"
        else:
            raise ValueError(f"Unsupported scale: {self.scale}")
        
        # Download weights if needed
        if download and not os.path.exists(weights_path):
            logger.info(f"Downloading weights for {model_name}")
            # Official model URLs
            model_urls = {
                'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            }
            
            if model_name in model_urls:
                load_file_from_url(
                    url=model_urls[model_name],
                    model_dir=weights_dir,
                    progress=True,
                    file_name=os.path.basename(weights_path)
                )
        
        # Initialize RealESRGANer
        self.model = RealESRGANer(
            scale=self.scale,
            model_path=weights_path,
            dni_weight=None,
            model=None,  # Let the library create the appropriate model
            half=self.device == 'cuda',  # Use half precision for CUDA
            tile=512,    # Tile size for processing large images
            tile_pad=10, # Padding for tiles to avoid seams
            pre_pad=0,   # No pre-padding needed
            device=self.device
        )
        
        self.weights_loaded = True
        logger.info(f"RealESRGAN model loaded for {self.scale}x upscaling on {self.device}")
    
    def predict(self, img):
        \"\"\"Upscale an image\"\"\"
        if not self.weights_loaded or self.model is None:
            raise RuntimeError("Model weights not loaded. Call load_weights first.")
        
        # Convert PIL Image to OpenCV format
        if isinstance(img, Image.Image):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            img_cv = img
            
        # Process with RealESRGANer
        output, _ = self.model.enhance(img_cv, outscale=self.scale)
        
        # Convert back to PIL format
        if isinstance(img, Image.Image):
            if output.shape[2] == 3:
                # BGR to RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(output_rgb)
            else:
                # BGRA to RGBA
                output_rgba = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
                result = Image.fromarray(output_rgba)
            return result
        else:
            return output
"""

    with open(adapter_file, "w") as f:
        f.write(adapter_code.strip())
    
    logger.info(f"Created RealESRGAN adapter: {adapter_file}")
    return adapter_file

def fix_upscale_imports():
    """Fix imports in upscale.py"""
    upscale_file = os.path.join("app", "upscale.py")
    
    if not os.path.exists(upscale_file):
        logger.error(f"Could not find {upscale_file}")
        return False
    
    # Backup the original file
    backup_file = upscale_file + ".bak"
    shutil.copy2(upscale_file, backup_file)
    logger.info(f"Created backup: {backup_file}")
    
    with open(upscale_file, "r") as f:
        content = f.read()
    
    # Replace the import
    new_content = content.replace(
        "from realesrgan import RealESRGAN", 
        "from app.realesrgan_adapter import RealESRGAN"
    )
    
    if new_content == content:
        logger.info("No changes needed in upscale.py")
        return True
        
    with open(upscale_file, "w") as f:
        f.write(new_content)
    
    logger.info(f"Updated imports in {upscale_file}")
    return True

def main():
    logger.info("Starting Real-ESRGAN import hotfix")
    
    try:
        # Create adapter
        adapter_file = create_realesrgan_adapter()
        
        # Fix imports
        if fix_upscale_imports():
            logger.info("Hotfix applied successfully!")
            logger.info("Please restart the API server to apply the changes.")
        else:
            logger.error("Failed to apply hotfix")
    
    except Exception as e:
        logger.error(f"Error applying hotfix: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
