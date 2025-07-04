from celery import Celery
from PIL import Image
import cloudinary.uploader
import cloudinary
import os
import logging
import time
import psutil
from dotenv import load_dotenv
from app.upscale import run_upscale
from contextlib import contextmanager
import gc

load_dotenv()

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Celery with optimized configuration
celery = Celery("real_esrgan_tasks")
celery.config_from_object("celeryconfig")

# Configure Cloudinary with optimized settings
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
    timeout=60  # Increase timeout for large files
)

@contextmanager
def redis_connection():
    """Context manager for local Redis connections only"""
    from redis import Redis
    redis_client = None
    
    try:
        # Always use local Redis at 127.0.0.1:6379
        logger.info("Connecting to local Redis at 127.0.0.1:6379")
        
        redis_client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test the connection with a ping
        redis_client.ping()
        logger.info("Successfully connected to local Redis")
        
        yield redis_client
        
    except Exception as e:
        logger.error(f"Failed to connect to local Redis: {e}")
        logger.warning("Operating without Redis - job status tracking disabled")
        yield None
        
    finally:
        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass

@celery.task(name="upscale_image", bind=True)
def upscale_image(self, job_id: str, input_path: str, scale: int = 2, face_enhance: bool = False):
    """
    Enhanced Celery task to upscale image using Real-ESRGAN
    """
    start_time = time.time()
    
    with redis_connection() as redis_client:
        try:
            # Update status to processing with timestamp if Redis is available
            if redis_client:
                redis_client.hset(f"job:{job_id}", mapping={
                    "status": "processing",
                    "processing_started": str(start_time),
                    "worker_pid": str(os.getpid())
                })
                redis_client.decr("stats:queued_jobs")
            
            logger.info(f"Starting upscale for job {job_id} (scale={scale}, face_enhance={face_enhance})")
            
            # Validate input file exists and is readable
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if not os.access(input_path, os.R_OK):
                raise PermissionError(f"Cannot read input file: {input_path}")
            
            # Check file size and type
            file_size = os.path.getsize(input_path)
            logger.info(f"Processing file: {input_path} ({file_size} bytes)")
            
            # Create output directory with job-specific naming
            os.makedirs("output", exist_ok=True)
            output_filename = f"{job_id}_upscaled_{scale}x{'_face' if face_enhance else ''}.png"
            output_path = f"output/{output_filename}"
            
            # Run upscaling with enhanced parameters
            upscale_start = time.time()
            run_upscale(input_path, output_path, scale=scale, face_enhance=face_enhance)
            upscale_time = time.time() - upscale_start
            
            logger.info(f"Upscaling completed for job {job_id} in {upscale_time:.2f}s")
            
            # Validate output file was created
            if not os.path.exists(output_path):
                raise RuntimeError("Upscaling completed but output file not found")
            
            output_size = os.path.getsize(output_path)
            logger.info(f"Output file created: {output_path} ({output_size} bytes)")
            
            # Upload to Cloudinary with optimized settings
            upload_start = time.time()
            upload_result = cloudinary.uploader.upload(
                output_path,
                public_id=f"upscaled_{job_id}_{int(start_time)}",
                folder="ai-upscaler/results",
                resource_type="image",
                quality="auto:best",  # Best quality
                fetch_format="auto",
                transformation=[
                    {"quality": "auto:best"},
                    {"fetch_format": "auto"}
                ],
                tags=[f"scale_{scale}", "real_esrgan", f"job_{job_id}"],
                context=f"job_id={job_id}|scale={scale}|face_enhance={face_enhance}"
            )
            upload_time = time.time() - upload_start
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Clean up local files
            cleanup_files = [input_path, output_path]
            for file_path in cleanup_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
            
            # Prepare result data
            result_data = {
                "status": "completed",
                "result_url": upload_result["secure_url"],
                "cloudinary_id": upload_result["public_id"],
                "cloudinary_url": upload_result["url"],
                "processing_time": str(round(total_time, 2)),
                "upscale_time": str(round(upscale_time, 2)),
                "upload_time": str(round(upload_time, 2)),
                "completed_at": str(time.time()),
                "output_size": str(output_size),
                "scale_used": str(scale),
                "face_enhance_used": str(face_enhance),  # Convert boolean to string
                "compression_ratio": str(round(output_size / file_size, 2) if file_size > 0 else 0)
            }
            
            # Update Redis if available
            if redis_client:
                redis_client.hset(f"job:{job_id}", mapping=result_data)
                
                # Update statistics
                redis_client.incr("stats:completed_jobs")
                redis_client.lpush("stats:processing_times", str(total_time))
                redis_client.ltrim("stats:processing_times", 0, 99)  # Keep last 100 times
            else:
                # Log result when Redis isn't available
                logger.info(f"Job {job_id} completed (Redis unavailable): {result_data['result_url']}")
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info(f"Job {job_id} completed successfully in {total_time:.2f}s")
            return upload_result["secure_url"]
            
        except Exception as exc:
            error_time = time.time() - start_time
            logger.error(f"Job {job_id} failed after {error_time:.2f}s: {exc}")
            
            # Clean up files on error
            cleanup_files = [input_path, f"output/{job_id}_upscaled_{scale}x{'_face' if face_enhance else ''}.png"]
            for file_path in cleanup_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path} on error: {e}")
            
            # Prepare error data
            error_data = {
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": str(time.time()),
                "processing_time": str(round(error_time, 2)),
                "worker_pid": str(os.getpid())
            }
            
            # Update Redis if available
            if redis_client:
                redis_client.hset(f"job:{job_id}", mapping=error_data)
                redis_client.incr("stats:failed_jobs")
            else:
                logger.warning(f"Job {job_id} failed (Redis unavailable): {str(exc)}")
            
            # Force garbage collection
            gc.collect()
            
            # Retry with exponential backoff for transient errors
            if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 2:
                countdown = 2 ** self.request.retries * 60  # 60s, 120s
                logger.info(f"Retrying job {job_id} in {countdown}s (attempt {self.request.retries + 1})")
                raise self.retry(exc=exc, countdown=countdown, max_retries=2)
            
            raise exc

@celery.task(name="cleanup_old_jobs")
def cleanup_old_jobs():
    """Cleanup old job data from Redis"""
    with redis_connection() as redis_client:
        # Skip if Redis is not available
        if not redis_client:
            logger.warning("Skipping cleanup_old_jobs: Redis not available")
            return {"status": "skipped", "reason": "redis_unavailable"}
            
        try:
            # Find jobs older than 7 days
            cutoff_time = time.time() - (7 * 24 * 60 * 60)
            
            job_keys = redis_client.keys("job:*")
            cleaned_count = 0
            
            for key in job_keys:
                try:
                    created_at = redis_client.hget(key, "created_at")
                    if created_at and float(created_at) < cutoff_time:
                        redis_client.delete(key)
                        cleaned_count += 1
                except:
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
            raise

@celery.task(name="health_check")
def worker_health_check():
    """Worker health check task"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_data = {
            "timestamp": time.time(),
            "worker_pid": os.getpid(),
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available_mb": round(memory.available / (1024**2), 2),
            "status": "healthy"
        }
        
        logger.info(f"Worker health check: CPU {cpu_percent}%, Memory {memory.percent}%")
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
