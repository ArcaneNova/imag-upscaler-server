#!/bin/bash
# entrypoint.sh - Main entrypoint for the container

# Don't exit on errors, handle them gracefully
set +e

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
echo "Current user: $(whoami)"
echo "Working directory: $(pwd)"

# Check if Redis should be started locally
if [ "$REDIS_DISABLE" != "true" ] && [ -z "$REDIS_URL" ] && [ -z "$REDISURL" ]; then
    echo "=== Attempting to start Redis locally ==="
    
    # Check if redis-server is available
    if command -v redis-server >/dev/null 2>&1; then
        echo "Redis server found, starting..."
        
        # Create directories with proper permissions
        mkdir -p /tmp/redis
        
        # Start Redis in the background with minimal config
        redis-server --daemonize yes --port 6379 --bind 127.0.0.1 --dir /tmp/redis --save "" --appendonly no --protected-mode no 2>/dev/null &
        
        # Wait for Redis to be ready
        echo "Waiting for Redis to start..."
        REDIS_STARTED=false
        for i in {1..10}; do
            if redis-cli -h 127.0.0.1 -p 6379 ping >/dev/null 2>&1; then
                echo "✓ Redis is ready"
                export REDIS_HOST=127.0.0.1
                export REDIS_PORT=6379
                REDIS_STARTED=true
                break
            fi
            echo "Waiting for Redis... ($i/10)"
            sleep 1
        done
        
        if [ "$REDIS_STARTED" != "true" ]; then
            echo "✗ Redis failed to start, disabling Redis"
            export REDIS_DISABLE=true
        fi
    else
        echo "Redis server not found, disabling Redis"
        export REDIS_DISABLE=true
    fi
else
    echo "=== Skipping local Redis (external Redis configured or disabled) ==="
fi

echo "=== Starting FastAPI application ==="
echo "Environment variables:"
echo "REDIS_HOST: $REDIS_HOST"
echo "REDIS_PORT: $REDIS_PORT"
echo "REDIS_DISABLE: $REDIS_DISABLE"

# Test if uvicorn is available
if ! command -v uvicorn >/dev/null 2>&1; then
    echo "ERROR: uvicorn not found in PATH"
    echo "Available Python packages:"
    python -m pip list | head -10
    exit 1
fi

# Test if the main app can be imported
echo "Testing application import..."
python -c "from main import app; print('✓ Application imported successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✗ Failed to import main app, trying alternative import..."
    python -c "from app.main import app; print('✓ Application imported via app.main')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to import application"
        echo "Python path: $PYTHONPATH"
        echo "Files in /app:"
        ls -la /app/
        echo "Files in /app/app:"
        ls -la /app/app/ 2>/dev/null || echo "No app/app directory"
        exit 1
    fi
fi

echo "✓ Starting uvicorn server..."

# Use Railway's PORT environment variable if available, otherwise default to 8000
PORT=${PORT:-8000}
echo "Starting uvicorn on port $PORT"

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers $WORKERS
