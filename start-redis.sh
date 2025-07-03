#!/bin/bash
# start-redis.sh - Script to start Redis locally for testing

echo "Starting Redis for testing the Real-ESRGAN API server..."

# Check if Redis is already installed
if command -v redis-server >/dev/null 2>&1; then
    echo "Redis is installed. Starting Redis server..."
    redis-server --daemonize yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    
    # Check if Redis started successfully
    if redis-cli ping >/dev/null 2>&1; then
        echo "Redis is now running on localhost:6379"
        echo "To stop Redis: redis-cli shutdown"
    else
        echo "Failed to start Redis. Please check your Redis installation."
    fi
else
    echo "Redis is not installed. Please install Redis or use Docker:"
    echo ""
    echo "For Ubuntu/Debian: sudo apt-get install redis-server"
    echo "For macOS: brew install redis"
    echo "For Windows: Install Redis using WSL or download from https://github.com/microsoftarchive/redis/releases"
    echo ""
    echo "Alternatively, use Docker: docker run -p 6379:6379 redis:7-alpine"
fi
