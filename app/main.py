from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from uuid import uuid4
from redis import Redis
import shutil
import os
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import aiofiles
from typing import Optional
from contextlib import asynccontextmanager
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
redis_client = None
executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client, executor
    
    # Startup
    logger.info("Starting up Real-ESRGAN API server")
    logger.info("Step 1: Setting up local Redis configuration")
    
    # Force local Redis setup - always use local Redis server only
    logger.info("Forcing local Redis configuration - removing all external Redis settings")
    
    # Clear ALL possible external Redis environment variables
    redis_vars_to_clear = [
        "REDIS_URL", "REDISURL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD",
        "REDIS_TLS_URL", "REDIS_PRIVATE_URL", "REDIS_USERNAME", "REDIS_DB",
        "REDISCLOUD_URL", "REDISTOGO_URL", "OPENREDIS_URL", "REDISGREEN_URL",
        "REDIS_MASTER_SERVICE_HOST", "REDIS_MASTER_SERVICE_PORT"
    ]
    
    for var in redis_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            logger.info(f"Removed {var} environment variable")
    
    # Force local Redis configuration and prevent any external connections
    os.environ["REDIS_HOST"] = "127.0.0.1"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["REDIS_DISABLE"] = "false"
    
    logger.info("✅ All external Redis environment variables cleared")
    logger.info("✅ Local Redis configuration enforced")
    
    # Debug: Log all remaining Redis-related environment variables
    logger.info("=== Current Redis Environment Variables ===")
    for key, value in os.environ.items():
        if 'redis' in key.lower() or 'REDIS' in key:
            logger.info(f"{key}={value}")
    logger.info("=== End Redis Environment Variables ===")
    
    # Also check for any service discovery variables that might affect Redis
    k8s_vars = [k for k in os.environ.keys() if 'SERVICE' in k and 'REDIS' in k.upper()]
    if k8s_vars:
        logger.warning(f"Found Kubernetes service discovery variables: {k8s_vars}")
        for var in k8s_vars:
            logger.warning(f"Removing {var}={os.environ[var]}")
            del os.environ[var]
    
    # Check if Redis is already running locally
    try:
        result = subprocess.run(
            ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "ping"], 
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            logger.info("Redis is already running locally at 127.0.0.1:6379")
        else:
            raise Exception("Redis not running")
    except:
        # Redis not running, start it locally
        logger.info("Starting Redis server locally at 127.0.0.1:6379...")
        
        # Create Redis directory
        os.makedirs("/tmp/redis", exist_ok=True)
        
        # Start Redis server with minimal configuration for local use only
        redis_cmd = [
            "redis-server",
            "--daemonize", "yes",
            "--port", "6379",
            "--bind", "127.0.0.1",
            "--dir", "/tmp/redis",
            "--save", "",
            "--appendonly", "no",
            "--protected-mode", "no",
            "--maxmemory", "256mb",
            "--maxmemory-policy", "allkeys-lru"
        ]
        
        try:
            result = subprocess.run(redis_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("Redis server started successfully")
                
                # Wait for Redis to be ready
                for i in range(15):
                    try:
                        test_result = subprocess.run(
                            ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "ping"], 
                            capture_output=True, text=True, timeout=2
                        )
                        if test_result.returncode == 0:
                            logger.info("Local Redis is ready and responding")
                            break
                    except:
                        pass
                    time.sleep(1)
                else:
                    raise Exception("Redis started but not responding to ping after 15 seconds")
            else:
                raise Exception(f"Failed to start Redis: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to start local Redis: {e}")
            logger.warning("Continuing without Redis - some features will be limited")
    
    # Initialize Redis client - local Redis only, but don't fail if unavailable
    logger.info("Step 2: Attempting to connect to local Redis")
    try:
        logger.info("Connecting to local Redis at 127.0.0.1:6379")
        
        redis_client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test the connection
        redis_client.ping()
        logger.info("Successfully connected to local Redis!")
        
    except Exception as e:
        logger.warning(f"Failed to connect to local Redis: {e}")
        logger.warning("Operating without Redis - job status tracking will be limited")
        redis_client = None
    
    # Initialize thread pool with optimal workers
    logger.info("Step 3: Initializing thread pool")
    cpu_count = psutil.cpu_count()
    max_workers = min(cpu_count * 2, 8)  # 2x CPU cores, max 8
    executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"Initialized thread pool with {max_workers} workers")
    
    # Create necessary directories
    logger.info("Step 4: Creating necessary directories")
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    logger.info("✅ Application startup completed successfully!")
    
    # Preload models in background for faster response times
    try:
        import threading
        def preload_models_background():
            try:
                from app.upscale import preload_models
                preload_models()
            except Exception as e:
                logger.warning(f"Background model preloading failed: {e}")
        
        preload_thread = threading.Thread(target=preload_models_background, daemon=True)
        preload_thread.start()
        logger.info("🚀 Model preloading started in background")
    except Exception as e:
        logger.warning(f"Failed to start model preloading: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Real-ESRGAN API server")
    if executor:
        executor.shutdown(wait=True)
    if redis_client:
        redis_client.close()

# Rate limiter with IP-based limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Real-ESRGAN Image Upscaler API",
    description="High-performance image upscaling service with Real-ESRGAN",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Root endpoint that returns simple health status
@app.get("/")
async def root():
    """Simple root endpoint for health checks"""
    return {
        "service": "realesrgan-api", 
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/ping")
async def ping():
    """Ultra-simple ping endpoint for basic connectivity test"""
    return {"ping": "pong"}

@app.get("/status")
async def simple_status():
    """Simple status endpoint for basic health checks"""
    return {
        "status": "ok",
        "service": "realesrgan-api",
        "timestamp": time.time()
    }

# CORS configuration - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400  # Cache preflight requests for 24 hours
)

async def get_redis_client():
    """
    Dependency to get Redis client, returns None if Redis is unavailable.
    Includes automatic reconnection attempts if Redis was disconnected.
    """
    global redis_client
    
    # If Redis was previously connected but now appears disconnected, try to reconnect
    if redis_client is not None:
        try:
            # Test if the connection is still alive
            redis_client.ping()
        except Exception as e:
            # Connection seems to be lost, try to reconnect
            logger.warning(f"Redis connection lost, attempting to reconnect: {e}")
            try:
                # Close the existing connection if possible
                try:
                    redis_client.close()
                except:
                    pass
                
                # Only reconnect to local Redis - no external fallbacks
                logger.info("Attempting to reconnect to local Redis at 127.0.0.1:6379")
                redis_client = Redis(
                    host="127.0.0.1",
                    port=6379,
                    decode_responses=True,
                    socket_timeout=3,
                    socket_connect_timeout=3
                )
                redis_client.ping()
                logger.info("Successfully reconnected to local Redis")
            except Exception as reconnect_err:
                logger.error(f"Redis reconnection failed: {reconnect_err}")
                redis_client = None
    
    # Return None instead of raising an exception to allow routes to handle missing Redis
    return redis_client

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with debugging information"""
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "redis": "disconnected",
        "system": {},
        "debug": {
            "docker_container": os.getenv("DOCKER_CONTAINER"),
            "redis_host": "127.0.0.1",
            "redis_port": "6379",
            "redis_disable": "false",
            "working_directory": os.getcwd(),
            "python_path": os.getenv("PYTHONPATH")
        }
    }
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            health_data["redis"] = "connected"
            # Get Redis info
            redis_info = redis_client.info()
            health_data["redis_info"] = {
                "used_memory": redis_info.get("used_memory_human", "N/A"),
                "connected_clients": redis_info.get("connected_clients", "N/A"),
                "total_connections_received": redis_info.get("total_connections_received", "N/A")
            }
        except Exception as e:
            health_data["redis"] = f"error: {str(e)}"
    
    # System metrics
    try:
        health_data["system"] = {
            "cpu_usage": f"{psutil.cpu_percent(interval=0.1):.1f}%",
            "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
            "disk_usage": f"{psutil.disk_usage('/').percent:.1f}%",
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else "N/A"
        }
    except Exception as e:
        health_data["system"] = f"error: {str(e)}"
    
    return health_data

@app.post("/upscale")
@limiter.limit("20/minute")  # Increased rate limit
async def submit_upscale(
    request: Request,  # Properly type-annotate as Request for rate limiter
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = 2,
    face_enhance: bool = False,
    direct_process: bool = False,  # Add support for direct processing option
    redis_client: Redis = Depends(get_redis_client)
):
    """Submit image for upscaling with enhanced validation and processing"""
    
    # Enhanced file validation
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Only images are supported."
        )
    
    # Validate scale parameter
    if scale not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
    
    # Check file size with more granular limits
    max_size = 15 * 1024 * 1024  # 15MB limit
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large: {file.size / (1024*1024):.1f}MB. Max size: {max_size / (1024*1024)}MB"
        )
    
    # Check minimum file size
    if file.size and file.size < 1024:  # 1KB minimum
        raise HTTPException(status_code=400, detail="File too small (minimum 1KB)")
    
    try:
        job_id = str(uuid4())
        timestamp = time.time()
        temp_path = f"temp/{job_id}_{int(timestamp)}.jpg"
        
        # Use async file operations for better performance
        async with aiofiles.open(temp_path, 'wb') as f:
            while True:
                chunk = await file.read(8192)  # 8KB chunks
                if not chunk:
                    break
                await f.write(chunk)
        
        # Prepare job data - convert all values to strings for Redis compatibility
        job_data = {
            "status": "queued",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": str(file.size),  # Convert to string
            "scale": str(scale),  # Convert to string
            "face_enhance": str(face_enhance),  # Convert boolean to string
            "created_at": str(timestamp),  # Convert to string
            "temp_path": temp_path
        }
        
        # Store job info in Redis if available
        if redis_client:
            # Set job data with 24-hour expiration
            redis_client.hset(f"job:{job_id}", mapping=job_data)
            redis_client.expire(f"job:{job_id}", 86400)  # 24 hours
            
            # Add to processing queue with priority - convert boolean to string
            queue_data = {
                "job_id": job_id,
                "priority": 1 if face_enhance else 0,  # Higher priority for face enhancement
                "timestamp": timestamp
            }
            redis_client.lpush("processing_queue", f"{job_id}|{scale}|{str(face_enhance)}")
            
            # Update queue stats
            redis_client.incr("stats:total_jobs")
            redis_client.incr("stats:queued_jobs")
        else:
            logger.warning(f"Redis unavailable: job {job_id} not tracked in Redis")
        
        # Process based on direct_process parameter
        if direct_process:
            # For direct processing, handle the upscaling right away in this request
            try:
                logger.info(f"Direct processing requested for job {job_id}")
                
                # Import and use upscaler directly
                from app.upscale import run_upscale
                
                # Define output path
                output_path = f"output/{job_id}_{int(timestamp)}.png"
                
                # Run upscaling directly
                run_upscale(temp_path, output_path, scale, face_enhance)
                
                # Check if the output file exists
                if not os.path.exists(output_path):
                    raise ValueError("Upscaling failed to produce output file")
                
                # Handle file upload to Cloudinary
                cloudinary_url = None
                try:
                    import cloudinary
                    import cloudinary.uploader
                    
                    if all([
                        os.getenv("CLOUDINARY_CLOUD_NAME"),
                        os.getenv("CLOUDINARY_API_KEY"),
                        os.getenv("CLOUDINARY_API_SECRET")
                    ]):
                        # Configure Cloudinary
                        cloudinary.config(
                            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
                            api_key=os.getenv("CLOUDINARY_API_KEY"),
                            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
                            secure=True
                        )
                        
                        # Upload to Cloudinary
                        upload_result = cloudinary.uploader.upload(
                            output_path,
                            folder="realesrgan_upscales",
                            resource_type="image"
                        )
                        
                        cloudinary_url = upload_result.get("secure_url")
                        logger.info(f"Image uploaded to Cloudinary: {cloudinary_url}")
                except Exception as cloud_err:
                    logger.error(f"Cloudinary upload failed: {cloud_err}")
                
                # Update job data
                completion_time = time.time()
                processing_time = completion_time - timestamp
                
                # Store result in Redis if available
                if redis_client:
                    result_data = {
                        "status": "completed",
                        "completed_at": str(completion_time),  # Convert to string
                        "processing_time": str(processing_time),  # Convert to string
                        "result_url": cloudinary_url or "",
                        "output_path": output_path
                    }
                    redis_client.hset(f"job:{job_id}", mapping=result_data)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Return the result directly
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "processing_time": f"{processing_time:.2f} seconds",
                    "result_url": cloudinary_url,
                    "message": "Image upscaled successfully",
                    "parameters": {
                        "scale": scale,
                        "face_enhance": face_enhance
                    }
                }
                
            except Exception as direct_err:
                logger.error(f"Direct processing failed: {direct_err}")
                # Clean up temp files
                for path in [temp_path, f"output/{job_id}_{int(timestamp)}.png"]:
                    if os.path.exists(path):
                        os.remove(path)
                raise HTTPException(
                    status_code=500, 
                    detail=f"Direct processing failed: {str(direct_err)}"
                )
        else:
            # Try to submit to Celery worker asynchronously for background processing
            try:
                from app.worker import upscale_image
                
                # Ensure parameters are correct types for Celery
                scale_int = int(scale)
                face_enhance_bool = bool(face_enhance)
                
                # Test Celery connection first
                celery_task = upscale_image.delay(job_id, temp_path, scale_int, face_enhance_bool)
                
                logger.info(f"Job {job_id} queued via Celery: {file.filename} ({file.size} bytes, scale={scale_int})")
                
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "message": "Image submitted for upscaling via background worker",
                    "estimated_time": "10-60 seconds",
                    "parameters": {
                        "scale": scale_int,
                        "face_enhance": face_enhance_bool
                    }
                }
                
            except Exception as celery_err:
                logger.warning(f"Celery submission failed: {celery_err}")
                logger.info(f"Falling back to direct processing for job {job_id}")
                
                # Fall back to direct processing
                try:
                    # Import and use upscaler directly
                    from app.upscale import run_upscale
                    
                    # Define output path
                    output_path = f"output/{job_id}_{int(timestamp)}.png"
                    
                    # Run upscaling directly
                    run_upscale(temp_path, output_path, scale, face_enhance)
                    
                    # Check if the output file exists
                    if not os.path.exists(output_path):
                        raise ValueError("Upscaling failed to produce output file")
                    
                    # Handle file upload to Cloudinary
                    cloudinary_url = None
                    try:
                        import cloudinary
                        import cloudinary.uploader
                        
                        if all([
                            os.getenv("CLOUDINARY_CLOUD_NAME"),
                            os.getenv("CLOUDINARY_API_KEY"),
                            os.getenv("CLOUDINARY_API_SECRET")
                        ]):
                            # Configure Cloudinary
                            cloudinary.config(
                                cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
                                api_key=os.getenv("CLOUDINARY_API_KEY"),
                                api_secret=os.getenv("CLOUDINARY_API_SECRET"),
                                secure=True
                            )
                            
                            # Upload to Cloudinary
                            upload_result = cloudinary.uploader.upload(
                                output_path,
                                folder="realesrgan_upscales",
                                resource_type="image"
                            )
                            
                            cloudinary_url = upload_result.get("secure_url")
                            logger.info(f"Image uploaded to Cloudinary: {cloudinary_url}")
                    except Exception as cloud_err:
                        logger.error(f"Cloudinary upload failed: {cloud_err}")
                    
                    # Update job data
                    completion_time = time.time()
                    processing_time = completion_time - timestamp
                    
                    # Store result in Redis if available
                    if redis_client:
                        result_data = {
                            "status": "completed",
                            "completed_at": str(completion_time),  # Convert to string
                            "processing_time": str(processing_time),  # Convert to string
                            "result_url": cloudinary_url or "",
                            "output_path": output_path
                        }
                        redis_client.hset(f"job:{job_id}", mapping=result_data)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Return the result directly
                    return {
                        "job_id": job_id,
                        "status": "completed",
                        "processing_time": f"{processing_time:.2f} seconds",
                        "result_url": cloudinary_url,
                        "message": "Image upscaled successfully (fallback processing)",
                        "parameters": {
                            "scale": scale,
                            "face_enhance": face_enhance
                        }
                    }
                    
                except Exception as fallback_err:
                    logger.error(f"Fallback processing also failed: {fallback_err}")
                    # Clean up temp files
                    for path in [temp_path, f"output/{job_id}_{int(timestamp)}.png"]:
                        if os.path.exists(path):
                            os.remove(path)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Both Celery and fallback processing failed: {str(fallback_err)}"
                    )
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.get("/status/{job_id}")
@limiter.limit("60/minute")  # Higher limit for status checks
async def get_status(
    request: Request, 
    job_id: str,
    redis_client: Redis = Depends(get_redis_client)
):
    """Get job status with enhanced information"""
    
    try:
        # Handle case where Redis is unavailable
        if redis_client is None:
            # If Redis is unavailable, we can't retrieve status
            # Return a reasonable fallback response with limited information
            return {
                "status": "unknown",
                "message": "Job status tracking unavailable (Redis not connected)",
                "job_id": job_id,
                "fallback": True
            }
            
        job_data = redis_client.hgetall(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        
        # Add runtime information
        current_time = time.time()
        created_at = float(job_data.get("created_at", current_time))
        
        job_data["runtime_seconds"] = round(current_time - created_at, 2)
        
        # Add queue position if still queued
        if job_data.get("status") == "queued":
            try:
                queue_items = redis_client.lrange("processing_queue", 0, -1)
                position = next(
                    (i for i, item in enumerate(queue_items) if item.startswith(job_id)), 
                    -1
                )
                job_data["queue_position"] = position + 1 if position >= 0 else "unknown"
            except:
                job_data["queue_position"] = "unknown"
        
        return job_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")

@app.get("/queue")
@limiter.limit("30/minute")
async def get_queue_stats(
    request: Request,
    redis_client: Redis = Depends(get_redis_client)
):
    """Get queue statistics"""
    
    # Handle case where Redis is unavailable
    if redis_client is None:
        return {
            "queue_length": 0,
            "total_jobs": 0,
            "queued_jobs": 0,
            "status_counts": {"queued": 0, "processing": 0, "completed": 0, "failed": 0},
            "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else "N/A",
            "redis_available": False,
            "message": "Queue tracking unavailable (Redis not connected)"
        }
    
    try:
        queue_length = redis_client.llen("processing_queue")
        total_jobs = redis_client.get("stats:total_jobs") or 0
        queued_jobs = redis_client.get("stats:queued_jobs") or 0
        
        # Get jobs by status
        keys = redis_client.keys("job:*")
        status_counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
        
        for key in keys[:100]:  # Limit to first 100 for performance
            try:
                status = redis_client.hget(key, "status")
                if status in status_counts:
                    status_counts[status] += 1
            except:
                continue
        
        return {
            "queue_length": queue_length,
            "total_jobs": int(total_jobs),
            "queued_jobs": int(queued_jobs),
            "status_counts": status_counts,
            "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else "N/A",
            "redis_available": True
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue stats")

@app.delete("/job/{job_id}")
@limiter.limit("10/minute")
async def cancel_job(
    request: Request,
    job_id: str,
    redis_client: Redis = Depends(get_redis_client)
):
    """Cancel a queued job"""
    
    try:
        job_data = redis_client.hgetall(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        status = job_data.get("status")
        
        if status == "processing":
            raise HTTPException(status_code=400, detail="Cannot cancel job in progress")
        
        if status in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail=f"Job already {status}")
        
        # Remove from queue
        redis_client.lrem("processing_queue", 0, f"{job_id}|*")
        
        # Update job status
        redis_client.hset(f"job:{job_id}", "status", "cancelled")
        
        # Clean up temp file
        temp_path = job_data.get("temp_path")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {"message": "Job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")

@app.get("/metrics")
async def get_metrics(
    request: Request,
    redis_client: Redis = Depends(get_redis_client)
):
    """Get detailed API metrics"""
    
    try:
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Redis metrics
        redis_info = redis_client.info()
        
        # Job metrics
        total_jobs = int(redis_client.get("stats:total_jobs") or 0)
        queue_length = redis_client.llen("processing_queue")
        
        return {
            "timestamp": time.time(),
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_mb": round(redis_info.get("used_memory", 0) / (1024**2), 2),
                "operations_per_second": redis_info.get("instantaneous_ops_per_sec", 0)
            },
            "jobs": {
                "total_processed": total_jobs,
                "current_queue_length": queue_length,
                "average_processing_time": "N/A"  # Would need historical data
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker with lifespan
        loop="uvloop",
        log_level="info",
        access_log=True,
        reload=False
    )
