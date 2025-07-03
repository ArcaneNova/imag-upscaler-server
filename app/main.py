from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
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
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import aiofiles
from typing import Optional
from app.worker import upscale_image
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
    
    # Initialize Redis with connection pooling and fallback
    try:
        # Get Redis host with multiple fallback options
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        # Check if Redis is disabled by environment variable
        if os.getenv("REDIS_DISABLE", "").lower() in ("true", "1", "yes"):
            logger.warning("Redis disabled by environment variable - operating in local mode")
            redis_client = None
        else:
            # Check if we're running in a local development environment
            if redis_host == "redis" and not os.getenv("DOCKER_CONTAINER"):
                # Try localhost as fallback for local development
                try:
                    test_client = Redis(
                        host="localhost",
                        port=redis_port,
                        socket_timeout=2,
                        decode_responses=True
                    )
                    test_client.ping()
                    logger.info("Using localhost Redis connection for local development")
                    redis_host = "localhost"
                    test_client.close()
                except Exception:
                    # Localhost also failed, will continue with original host
                    pass
                
            # Try to connect to Redis with better error handling
            redis_client = Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=50  # Connection pooling
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port} with connection pooling")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        logger.warning("Operating without Redis - some features like job status tracking will be limited")
        redis_client = None
    
    # Initialize thread pool with optimal workers
    cpu_count = psutil.cpu_count()
    max_workers = min(cpu_count * 2, 8)  # 2x CPU cores, max 8
    executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"Initialized thread pool with {max_workers} workers")
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
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

# Root endpoint that redirects to docs
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

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
    """Dependency to get Redis client, returns None if Redis is unavailable"""
    # Return None instead of raising an exception to allow routes to handle missing Redis
    return redis_client

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "redis": "disconnected",
        "system": {}
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
            "cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%",
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
    request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = 2,
    face_enhance: bool = False,
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
        
        # Prepare job data
        job_data = {
            "status": "queued",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size,
            "scale": scale,
            "face_enhance": face_enhance,
            "created_at": timestamp,
            "temp_path": temp_path
        }
        
        # Store job info in Redis if available
        if redis_client:
            # Set job data with 24-hour expiration
            redis_client.hset(f"job:{job_id}", mapping=job_data)
            redis_client.expire(f"job:{job_id}", 86400)  # 24 hours
            
            # Add to processing queue with priority
            queue_data = {
                "job_id": job_id,
                "priority": 1 if face_enhance else 0,  # Higher priority for face enhancement
                "timestamp": timestamp
            }
            redis_client.lpush("processing_queue", f"{job_id}|{scale}|{face_enhance}")
            
            # Update queue stats
            redis_client.incr("stats:total_jobs")
            redis_client.incr("stats:queued_jobs")
        else:
            logger.warning(f"Redis unavailable: job {job_id} not tracked in Redis")
        
        # Submit to Celery worker asynchronously
        background_tasks.add_task(
            lambda: upscale_image.delay(job_id, temp_path, scale, face_enhance)
        )
        
        logger.info(f"Job {job_id} queued: {file.filename} ({file.size} bytes, scale={scale})")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image submitted for upscaling",
            "estimated_time": "30-120 seconds",
            "parameters": {
                "scale": scale,
                "face_enhance": face_enhance
            }
        }
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        # Clean up temp file on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")

@app.get("/status/{job_id}")
@limiter.limit("60/minute")  # Higher limit for status checks
async def get_status(
    request, 
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
    request,
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
    request,
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
    request,
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
