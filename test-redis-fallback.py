"""
Test Redis fallback mechanism in the Real-ESRGAN API server
This script simulates scenarios with and without Redis to ensure proper fallback behavior
"""
import os
import sys
import logging
from redis import Redis
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-redis")

@contextmanager
def redis_connection():
    """Context manager for Redis connections with fallback to None"""
    redis_client = None
    
    # Check if Redis is disabled by environment variable
    if os.getenv("REDIS_DISABLE", "").lower() in ("true", "1", "yes"):
        logger.warning("Redis disabled by environment variable - operating in local mode")
        yield None
        return
        
    try:
        # Get Redis host with multiple fallback options
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
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
        
        redis_client = Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=15
        )
        # Test the connection with a ping
        redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        yield redis_client
    except Exception as e:
        logger.warning(f"Redis connection error (falling back to local storage): {e}")
        logger.warning("Operating without Redis - job status updates will not be available")
        yield None
    finally:
        if redis_client:
            redis_client.close()

def test_redis_fallback():
    """Test Redis fallback mechanism"""
    logger.info("Testing Redis fallback mechanism")
    
    # Test with default settings (should try to connect to Redis)
    logger.info("=== Testing with default settings ===")
    with redis_connection() as client:
        if client:
            logger.info("Redis connection successful")
            # Try a simple operation
            client.set("test_key", "test_value")
            value = client.get("test_key")
            logger.info(f"Test value: {value}")
        else:
            logger.info("Running in fallback mode (no Redis)")
    
    # Test with REDIS_DISABLE=true
    logger.info("\n=== Testing with REDIS_DISABLE=true ===")
    os.environ["REDIS_DISABLE"] = "true"
    with redis_connection() as client:
        if client:
            logger.info("Redis connection successful (shouldn't happen)")
        else:
            logger.info("Running in fallback mode as expected")
    
    # Test with explicit localhost
    logger.info("\n=== Testing with localhost Redis ===")
    os.environ.pop("REDIS_DISABLE", None)
    os.environ["REDIS_HOST"] = "localhost"
    with redis_connection() as client:
        if client:
            logger.info("Redis connection to localhost successful")
        else:
            logger.info("Running in fallback mode (no Redis on localhost)")

if __name__ == "__main__":
    test_redis_fallback()
