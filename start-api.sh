#!/bin/bash
# start-api.sh - Startup script for the Real-ESRGAN API server

# Set environment variable to indicate we're running in Docker
export DOCKER_CONTAINER=1

# Set the number of workers based on environment variable or default to 1
WORKERS=${WORKERS:-1}

# Check if Redis should be started locally
if [ "$REDIS_DISABLE" != "true" ] && [ -z "$REDIS_URL" ] && [ -z "$REDISURL" ]; then
    echo "Starting Redis server locally..."
    
    # Start Redis with configuration file
    redis-server /app/redis.conf
    
    # Wait for Redis to start
    echo "Waiting for Redis to start..."
    sleep 3
    
    # Test Redis connection
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis started successfully"
        export REDIS_HOST=localhost
        export REDIS_PORT=6379
    else
        echo "Redis failed to start, continuing without Redis"
        export REDIS_DISABLE=true
    fi
fi

# Print startup information
echo "Starting Real-ESRGAN API Server"
echo "Workers: $WORKERS"
echo "Python: $(python --version)"
echo "Redis: ${REDIS_HOST:-disabled}"

# Execute uvicorn with proper settings
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers $WORKERS