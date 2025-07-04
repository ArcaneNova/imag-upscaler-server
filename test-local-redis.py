#!/usr/bin/env python3
"""
Test script to verify local Redis connection logic
"""
import os
import sys
import time
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_local_redis_startup():
    """Test the local Redis startup logic from main.py"""
    logger.info("Testing local Redis startup logic...")
    
    # Force local Redis setup - always use local Redis server only
    logger.info("Forcing local Redis configuration - removing all external Redis settings")
    
    # Clear any external Redis URLs to force local connection only
    if "REDIS_URL" in os.environ:
        del os.environ["REDIS_URL"]
        logger.info("Removed REDIS_URL environment variable")
    if "REDISURL" in os.environ:
        del os.environ["REDISURL"]
        logger.info("Removed REDISURL environment variable")
    
    # Force local Redis configuration
    os.environ["REDIS_HOST"] = "127.0.0.1"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["REDIS_DISABLE"] = "false"
    
    # Check if Redis is already running locally
    try:
        result = subprocess.run(
            ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "ping"], 
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            logger.info("Redis is already running locally at 127.0.0.1:6379")
            return True
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
                            return True
                    except:
                        pass
                    time.sleep(1)
                else:
                    raise Exception("Redis started but not responding to ping after 15 seconds")
            else:
                raise Exception(f"Failed to start Redis: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to start local Redis: {e}")
            logger.error("This is a critical error - the application requires local Redis")
            return False

def test_redis_connection():
    """Test Redis connection"""
    try:
        from redis import Redis
        
        # Only connect to local Redis - no external fallbacks
        logger.info("Connecting to local Redis at 127.0.0.1:6379")
        
        redis_client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=50,
            retry=3
        )
        
        # Test the connection
        redis_client.ping()
        logger.info("Successfully connected to local Redis!")
        
        # Test setting and getting a key
        test_key = "test:connection"
        test_value = "local_redis_works"
        redis_client.set(test_key, test_value, ex=60)
        retrieved_value = redis_client.get(test_key)
        
        if retrieved_value == test_value:
            logger.info("Redis read/write test passed!")
            redis_client.delete(test_key)
            return True
        else:
            logger.error(f"Redis read/write test failed: expected {test_value}, got {retrieved_value}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to local Redis: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    logger.info("Starting local Redis connection tests...")
    
    # Test 1: Redis startup
    if not test_local_redis_startup():
        logger.error("Redis startup test failed")
        success = False
    
    # Test 2: Redis connection
    if not test_redis_connection():
        logger.error("Redis connection test failed")
        success = False
    
    if success:
        logger.info("✅ All tests passed! Local Redis is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)
