#!/bin/bash
# entrypoint.sh - Main entrypoint for the container

set -e

echo "=== Real-ESRGAN API Server Container Starting ==="

# Set environment variables
export DOCKER_CONTAINER=1
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# Set workers
WORKERS=${WORKERS:-1}

echo "Workers: $WORKERS"
echo "Redis disable: $REDIS_DISABLE"
echo "Redis URL: $REDIS_URL"

# Check if we should start Redis locally
if [ "$REDIS_DISABLE" != "true" ] && [ -z "$REDIS_URL" ] && [ -z "$REDISURL" ]; then
    echo "=== Starting Redis locally ==="
    
    # Create directories
    mkdir -p /var/lib/redis /var/log/redis
    
    # Start Redis in the background
    redis-server /app/redis.conf &
    
    # Wait for Redis to be ready
    echo "Waiting for Redis to start..."
    for i in {1..10}; do
        if redis-cli ping >/dev/null 2>&1; then
            echo "✓ Redis is ready"
            export REDIS_HOST=localhost
            export REDIS_PORT=6379
            break
        fi
        echo "Waiting for Redis... ($i/10)"
        sleep 1
    done
    
    # Check if Redis started successfully
    if ! redis-cli ping >/dev/null 2>&1; then
        echo "✗ Redis failed to start, disabling Redis"
        export REDIS_DISABLE=true
    fi
else
    echo "=== Skipping local Redis (external Redis configured or disabled) ==="
fi

echo "=== Starting FastAPI application ==="

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers $WORKERS
