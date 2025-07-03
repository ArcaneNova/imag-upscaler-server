#!/bin/bash
# start-api.sh - Entry point script for the API container

echo "Starting Real-ESRGAN API server..."
echo "Path: $PYTHONPATH"
echo "Working directory: $(pwd)"

# Create required directories
mkdir -p temp output weights logs

# Run the API server
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop
