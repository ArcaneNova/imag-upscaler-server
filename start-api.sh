#!/bin/bash
# start-api.sh - Entry point script for the API container

echo "Starting Real-ESRGAN API server..."
echo "Path: $PYTHONPATH"
echo "Working directory: $(pwd)"

# Create required directories
mkdir -p temp output weights logs

# Clean up any stale temporary files
echo "Cleaning up any stale temporary files..."
find temp -type f -mtime +1 -delete 2>/dev/null || echo "No old temp files to clean"

# Ensure we're using app.main module
echo "Starting API server with app.main module..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
