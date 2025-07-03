#!/bin/bash
# start-api.sh - Startup script for the Real-ESRGAN API server

set -e  # Exit on error

# Set environment variable to indicate we're running in Docker
export DOCKER_CONTAINER=1

# Set the number of workers based on environment variable or default to 1
WORKERS=${WORKERS:-1}

echo "=== Real-ESRGAN API Server Startup ==="
echo "Environment: Docker Container"
echo "Workers: $WORKERS"
echo "Python: $(python --version)"
echo "Redis disable: $REDIS_DISABLE"
echo "Redis URL: $REDIS_URL"
echo "Redis URL (alt): $REDISURL"

# Check if Redis should be started locally
if [ "$REDIS_DISABLE" != "true" ] && [ -z "$REDIS_URL" ] && [ -z "$REDISURL" ]; then
    echo "=== Starting Redis server locally ==="
    
    # Create redis directories if they don't exist
    mkdir -p /var/lib/redis /var/log/redis
    
    # Check if Redis is already running
    if pgrep redis-server > /dev/null; then
        echo "Redis is already running"
    else
        # Start Redis with configuration file
        echo "Starting Redis server..."
        redis-server /app/redis.conf
        
        # Wait for Redis to start
        echo "Waiting for Redis to start..."
        sleep 3
        
        # Test Redis connection
        if redis-cli ping > /dev/null 2>&1; then
            echo "✓ Redis started successfully"
            export REDIS_HOST=localhost
            export REDIS_PORT=6379
        else
            echo "✗ Redis failed to start, continuing without Redis"
            export REDIS_DISABLE=true
        fi
    fi
else
    echo "=== Skipping local Redis startup ==="
    echo "Reason: Redis disabled or external Redis URL provided"
fi

echo "=== Starting Real-ESRGAN API Server ==="
echo "Redis status: ${REDIS_HOST:-disabled}"

# Execute uvicorn with proper settings
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers $WORKERS