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
                # Create the model architecture for 2x upscaling
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            elif scale == 4:
                model_name = "RealESRGAN_x4plus"
                netscale = 4
                # Create the model architecture for 4x upscaling
                model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                raise ValueError(f"Unsupported scale: {scale}")
            
            logger.info(f"Creating model architecture: {model_name} (scale={netscale})")
            logger.info(f"Model architecture parameters: num_feat=64, num_block=23, num_grow_ch=32")
            
            # Verify the model architecture was created successfully
            if model_arch is None:
                raise RuntimeError(f"Failed to create model architecture for {model_name}")
            
            logger.info(f"Model architecture created successfully: {type(model_arch)}")
            
            # Move model to appropriate device
            model_arch = model_arch.to(device)
            logger.info(f"Model architecture moved to device: {device}")
                
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
            
            logger.info(f"Model weights file exists: {model_path} ({os.path.getsize(model_path)} bytes)")
            
            # Optimize tile size based on available memory and device
            # Larger tiles are faster but use more memory
            tile_size = 512
            tile_pad = 10
            
            if device == "cuda":
                # GPU can handle larger tiles for better performance
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    if gpu_memory >= 8:
                        tile_size = 1024  # Large tiles for high-end GPUs
                        tile_pad = 16
                    elif gpu_memory >= 4:
                        tile_size = 768   # Medium tiles for mid-range GPUs
                        tile_pad = 12
                    else:
                        tile_size = 512   # Standard tiles for entry-level GPUs
                        tile_pad = 10
                except:
                    tile_size = 512
                    tile_pad = 10
            else:
                # CPU optimization: balance between speed and memory
                cpu_count = os.cpu_count() or 4
                if cpu_count >= 16 and current_memory < 70:
                    tile_size = 768   # Larger tiles for powerful CPUs with plenty of RAM
                    tile_pad = 12
                elif cpu_count >= 8 and current_memory < 75:
                    tile_size = 640   # Medium tiles for decent CPUs
                    tile_pad = 10
                elif current_memory > 80:
                    tile_size = 256   # Small tiles when memory constrained
                    tile_pad = 8
                else:
                    tile_size = 512   # Default balanced setting
                    tile_pad = 10
                    
            logger.info(f"Using optimized tile size: {tile_size} (pad: {tile_pad}) for {device}")
            logger.info(f"System: CPU cores={os.cpu_count()}, Memory={current_memory}%")
            
            logger.info(f"Creating model architecture: {model_arch}")
            logger.info(f"Model device: {device}, netscale: {netscale}")
            
            # Initialize RealESRGANer with explicit model architecture and optimized settings
            model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=None,
                model=model_arch,  # Pass the explicitly created model architecture
                half=device == 'cuda',  # Use half precision for CUDA to increase speed
                tile=tile_size,    # Optimized tile size for better performance
                tile_pad=tile_pad, # Optimized padding to minimize seams
                pre_pad=0,   # No pre-padding needed for most images
                device=device
            )
            
            logger.info(f"RealESRGANer initialized successfully")
            _models[model_key] = model
            logger.info(f"Real-ESRGAN {scale}x model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            logger.error(f"Model architecture: {model_arch if 'model_arch' in locals() else 'None'}")
            logger.error(f"Model path: {model_path if 'model_path' in locals() else 'None'}")
            logger.error(f"Device: {device if 'device' in locals() else 'None'}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    """Apply fast post-processing enhancements"""
    
    try:
        # Only do face enhancement if explicitly requested
        if face_enhance:
            image = enhance_face_regions(image)
        
        # Skip heavy processing for faster results - only apply very light enhancements
        # Apply subtle sharpening (reduced intensity for speed)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=105, threshold=3))
        
        # Very light color enhancement (reduced for speed)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.02)  # 2% color boost instead of 5%
        
        # Light contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.01)  # 1% contrast boost instead of 2%
        
        return image
        
    except Exception as e:
        logger.warning(f"Post-processing failed, using original: {e}")
        return image

def run_upscale(
    input_path: str, 
    output_path: str, 
    scale: int = 2, 
    face_enhance: bool = False,
    max_dimension: int = 1536  # Reduced from 2048 for better speed
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
    
    # Dynamic max dimension based on system load - optimize for speed
    if memory_percent > 85 or load_avg > 10:
        # Reduce resolution when system is heavily loaded
        adjusted_max_dim = min(max_dimension, 1024)  # More aggressive reduction for speed
        if adjusted_max_dim != max_dimension:
            logger.warning(f"System under heavy load (CPU: {cpu_percent}%, Mem: {memory_percent}%, Load: {load_avg:.1f})")
            logger.warning(f"Reducing max dimension from {max_dimension} to {adjusted_max_dim}")
            max_dimension = adjusted_max_dim
    elif memory_percent > 75:
        # Moderate reduction for performance
        adjusted_max_dim = min(max_dimension, 1536)
        if adjusted_max_dim != max_dimension:
            logger.info(f"Moderate system load, reducing max dimension to {adjusted_max_dim} for better performance")
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
        
        # Check and resize if too large - optimize for processing speed
        original_pixels = original_width * original_height
        if max(original_width, original_height) > max_dimension:
            ratio = max_dimension / max(original_width, original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            # Use faster interpolation for speed
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        
        # For very large images, use even more aggressive optimization
        current_pixels = img.shape[1] * img.shape[0]
        if current_pixels > 2073600:  # > 1920x1080
            logger.info("Large image detected - using speed-optimized processing")
            # We'll pass this info to the model for optimization
        
        # Get model and perform upscaling
        model = get_model(scale)
        
        # Monitor memory usage
        before_upscale_memory = psutil.virtual_memory().percent
        
        # Perform upscaling
        logger.info(f"Starting Real-ESRGAN inference on {get_optimal_device()}")
        
        # Use RealESRGANer's enhance method which returns a NumPy array
        # Pre-optimize memory before heavy operation
        current_memory = psutil.virtual_memory().percent
        if current_memory > 80:
            logger.warning(f"High memory usage before inference: {current_memory}%")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Perform upscaling with optimized error handling
        try:
            with torch.no_grad():  # Disable gradient computation for inference speed
                output, _ = model.enhance(img, outscale=scale)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Memory error during inference, trying with reduced tile size")
                # Recreate model with smaller tiles
                device = get_optimal_device()
                smaller_model = get_model(scale)  # This will use cached model
                smaller_model.tile = max(128, smaller_model.tile // 2)  # Reduce tile size
                smaller_model.tile_pad = max(4, smaller_model.tile_pad // 2)
                
                # Force garbage collection and retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    output, _ = smaller_model.enhance(img, outscale=scale)
            else:
                raise  # Re-raise other errors
        
        # Clean up input image from memory immediately
        del img
        gc.collect()
        
        after_upscale_memory = psutil.virtual_memory().percent
        logger.info(f"Upscaling completed. Memory: {before_upscale_memory}% -> {after_upscale_memory}%")
        
        # Fast conversion to PIL (avoid unnecessary color space conversions when possible)
        if len(output.shape) == 3 and output.shape[2] == 3:
            # RGB format
            upscaled_img = Image.fromarray(output.astype('uint8'), 'RGB')
        elif len(output.shape) == 3 and output.shape[2] == 4:
            # RGBA format
            upscaled_img = Image.fromarray(output.astype('uint8'), 'RGBA')
        else:
            # Fallback: convert BGR to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            upscaled_img = Image.fromarray(output_rgb)
        
        # Clean up output array immediately
        del output
        
        # Determine if we should use fast mode based on system load
        fast_mode = False
        if _upscale_counter > 5:  # If we've processed multiple images recently
            current_load = psutil.cpu_percent(interval=0.1)
            if current_load > 70 or after_upscale_memory > 80:
                fast_mode = True
                logger.info("Using fast mode due to high system load")
        
        # Apply post-processing based on mode
        if fast_mode:
            # Skip post-processing for maximum speed
            logger.info("Skipping post-processing for maximum throughput")
        elif face_enhance:
            # Only do face enhancement if explicitly requested
            upscaled_img = postprocess_image(upscaled_img, face_enhance)
        elif scale == 4:
            # Light post-process 4x images for quality
            upscaled_img = postprocess_image(upscaled_img, False)
        
        # Save result with speed-optimized settings
        final_size = upscaled_img.size
        output_pixels = final_size[0] * final_size[1]
        
        # Choose format and settings based on size and performance requirements
        if fast_mode or output_pixels > 8_000_000:  # > 8MP or fast mode
            # Use JPEG for faster saving of large images
            output_path = output_path.replace('.png', '.jpg')
            save_kwargs = {
                "format": "JPEG",
                "quality": 92,  # Good quality but faster than 95
                "optimize": False  # Skip optimization for speed
            }
            logger.info("Using JPEG format for faster processing")
        elif output_pixels > 16_000_000:  # > 16MP
            # Use JPEG with optimization for very large images
            output_path = output_path.replace('.png', '.jpg')
            save_kwargs = {
                "format": "JPEG",
                "quality": 95,
                "optimize": True
            }
            logger.info("Using optimized JPEG format for very large image")
        else:
            # Use PNG with light compression for smaller images
            save_kwargs = {
                "format": "PNG",
                "optimize": False,  # Skip optimization for speed
                "compress_level": 3  # Light compression for speed
            }
        
        upscaled_img.save(output_path, **save_kwargs)
        
        final_memory = psutil.virtual_memory().percent
        
        logger.info(f"Image saved: {original_width}x{original_height} -> {final_size[0]}x{final_size[1]}")
        logger.info(f"Memory usage: {start_memory}% -> {final_memory}%")
        
        # Immediate cleanup for better performance
        del upscaled_img
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
                
                # Clear cache based on throughput and memory usage:
                # More aggressive clearing for high-throughput scenarios
                memory_usage = psutil.virtual_memory().percent
                
                if (_upscale_counter >= 30 or  # More frequent clearing for high throughput
                    (time_since_last_clear > 900 and memory_usage > 60) or  # 15 min with moderate memory usage
                    memory_usage > 85):  # Immediate clearing if memory is very high
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

def preload_models():
    """Preload commonly used models to reduce cold start times"""
    try:
        logger.info("Preloading Real-ESRGAN models for faster response times...")
        
        # Preload 2x model (most common)
        get_model(2)
        logger.info("✅ Real-ESRGAN 2x model preloaded")
        
        # Only preload 4x if we have sufficient memory
        memory_usage = psutil.virtual_memory().percent
        if memory_usage < 70:
            get_model(4)
            logger.info("✅ Real-ESRGAN 4x model preloaded")
        else:
            logger.info("⚠️ Skipped 4x model preload due to memory constraints")
            
    except Exception as e:
        logger.warning(f"Model preloading failed (will load on demand): {e}")

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
