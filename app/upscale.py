from PIL import Image, ImageFilter, ImageEnhance
import torch
import cv2
import numpy as np
import os
import logging
import gc
import psutil
import tempfile
import threading
import time as time_module
from typing import Optional, Tuple, Union
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torch.hub import download_url_to_file

logger = logging.getLogger(__name__)

# Global model instances for reuse
_models = {}
_device = None

# Track upscale operations for memory management
_upscale_counter = 0
_last_cache_clear_time = time_module.time()

def get_optimal_device():
    """Determine the best available device with memory management"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            try:
                # Check GPU memory availability before choosing it
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_memory_mb = free_memory / (1024 * 1024)
                
                if free_memory_mb > 1000:  # At least 1GB free
                    _device = "cuda"
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} with {free_memory_mb:.0f}MB free memory")
                else:
                    logger.warning(f"CUDA available but low memory ({free_memory_mb:.0f}MB free), falling back to CPU")
                    _device = "cpu"
            except Exception as e:
                logger.warning(f"Error checking GPU memory, falling back to CPU: {e}")
                _device = "cpu"
        else:
            _device = "cpu"
            logger.info("Using CPU for inference (CUDA not available)")
    return _device

def get_model(scale: int = 2):
    """Get or initialize the Real-ESRGAN model with caching and memory management"""
    global _models
    
    # Check current memory usage before loading model
    current_memory = psutil.virtual_memory().percent
    if current_memory > 90:
        # If memory usage is very high, clear cache first
        logger.warning(f"Memory usage critical ({current_memory}%), clearing model cache")
        clear_model_cache()
    
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
            
            # Define model path
            model_path = os.path.join(models_dir, f'{model_name}.pth')
            
            # Download model weights if they don't exist
            if not os.path.exists(model_path):
                logger.info(f"Model weights not found, downloading {model_name}")
                
                # Model URLs based on the official repo
                model_urls = {
                    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                }
                
                # Download weights
                if model_name in model_urls:
                    try:
                        load_file_from_url(
                            url=model_urls[model_name],
                            model_dir=models_dir,
                            progress=True,
                            file_name=f'{model_name}.pth'
                        )
                        logger.info(f"Successfully downloaded {model_name}.pth")
                    except Exception as download_err:
                        logger.error(f"Failed to download model weights: {download_err}")
                        raise RuntimeError(f"Could not download model weights for {model_name}")
                else:
                    raise ValueError(f"No download URL available for model {model_name}")
            
            # Verify model file exists after download
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found after download: {model_path}")
            
            # Optimize tile size based on available memory
            # Smaller tiles use less memory but may be slower
            tile_size = 512
            if device == "cpu" or current_memory > 75:
                tile_size = 256  # Use smaller tiles when memory constrained
                logger.info(f"Using smaller tile size ({tile_size}) due to memory constraints")
            
            # Initialize RealESRGANer with memory-optimized parameters
            model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=None,
                model=None,  # Let the library create the appropriate model
                half=device == 'cuda',  # Use half precision for CUDA
                tile=tile_size,    # Tile size for processing large images
                tile_pad=10, # Padding for tiles to avoid seams
                pre_pad=0,   # No pre-padding needed
                device=device
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
    Enhanced upscale function with optimizations and improved memory management
    
    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscaling factor (2 or 4)
        face_enhance: Apply additional face enhancement
        max_dimension: Maximum input dimension before resizing
    """
    # Check system load and adjust parameters
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
    
    # Dynamic max dimension based on system load
    if memory_percent > 85 or load_avg > 10:
        # Reduce resolution when system is heavily loaded
        adjusted_max_dim = min(max_dimension, 1536)  # Max 1536px under heavy load
        if adjusted_max_dim != max_dimension:
            logger.warning(f"System under heavy load (CPU: {cpu_percent}%, Mem: {memory_percent}%, Load: {load_avg:.1f})")
            logger.warning(f"Reducing max dimension from {max_dimension} to {adjusted_max_dim}")
            max_dimension = adjusted_max_dim
    
    start_memory = psutil.virtual_memory().percent
    logger.info(f"Starting upscale: scale={scale}x, face_enhance={face_enhance}, memory={start_memory}%")
    
    try:
        # Track upscale operations for cache management
        _increment_upscale_counter()
        
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
        # Add memory optimization - do cleanup before the heavy operation if memory is tight
        current_memory = psutil.virtual_memory().percent
        if current_memory > 80:
            logger.warning(f"High memory usage before inference: {current_memory}%")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Perform upscaling with proper error handling
        try:
            output, _ = model.enhance(img, outscale=scale)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                # Handle CUDA OOM errors by falling back to CPU
                logger.warning("CUDA out of memory, falling back to CPU")
                # Force CPU mode and retry
                backup_model = RealESRGANer(
                    scale=model.scale,
                    model_path=model.model_path,
                    dni_weight=None,
                    model=None,
                    tile=128,  # Smaller tiles for CPU mode
                    tile_pad=10,
                    pre_pad=0,
                    device="cpu"
                )
                output, _ = backup_model.enhance(img, outscale=scale)
                del backup_model  # Clean up immediately
            else:
                raise  # Re-raise other errors
        
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
        
        _increment_upscale_counter()  # Track this upscale operation
    
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        
        # Cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        raise

def clear_model_cache():
    """Clear all cached models to free memory with additional cleanup"""
    global _models
    
    logger.info("Clearing model cache")
    
    for model_key in list(_models.keys()):
        try:
            # More thorough cleanup of model resources
            if hasattr(_models[model_key], 'model') and _models[model_key].model is not None:
                del _models[model_key].model
            del _models[model_key]
        except Exception as e:
            logger.warning(f"Error while deleting model {model_key}: {e}")
    
    _models.clear()
    
    # Aggressive garbage collection
    for _ in range(3):  # Multiple GC passes
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Log GPU memory status
        try:
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            logger.info(f"GPU memory after cleanup: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        except Exception as e:
            logger.warning(f"Failed to log GPU memory stats: {e}")
    
    # Log system memory
    mem = psutil.virtual_memory()
    logger.info(f"System memory after cleanup: {mem.percent}% used, {mem.available/(1024**3):.1f}GB available")
    
    logger.info("Model cache cleared")

def schedule_periodic_cache_clearing():
    """Setup periodic cache clearing to prevent memory buildup"""
    def clear_cache_periodically():
        global _models, _upscale_counter, _last_cache_clear_time
        
        while True:
            try:
                time_module.sleep(300)  # Check every 5 minutes
                
                current_time = time_module.time()
                time_since_last_clear = current_time - _last_cache_clear_time
                
                # Clear cache if:
                # 1. More than 50 upscales since last clear OR
                # 2. More than 30 minutes since last clear AND memory usage is high
                memory_usage = psutil.virtual_memory().percent
                
                if (_upscale_counter >= 50 or 
                    (time_since_last_clear > 1800 and memory_usage > 70)):
                    logger.info(f"Scheduled cache clearing: {_upscale_counter} upscales, "
                                f"{time_since_last_clear:.0f}s since last clear, "
                                f"memory at {memory_usage}%")
                    clear_model_cache()
                    _upscale_counter = 0
                    _last_cache_clear_time = current_time
                
            except Exception as e:
                logger.error(f"Error in cache clearing thread: {e}")
    
    # Start background thread for cache clearing
    cache_thread = threading.Thread(target=clear_cache_periodically, daemon=True)
    cache_thread.start()
    logger.info("Periodic cache clearing scheduler started")

# Initialize the periodic cache clearing
try:
    schedule_periodic_cache_clearing()
except Exception as e:
    logger.error(f"Failed to start cache clearing scheduler: {e}")

def _increment_upscale_counter():
    """Increment the upscale counter for cache management"""
    global _upscale_counter
    _upscale_counter += 1

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
