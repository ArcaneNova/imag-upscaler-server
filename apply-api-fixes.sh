#!/bin/bash
# Apply fixes for the API issues

echo "Applying API fixes..."

# Rebuild and restart the containers
docker-compose build
docker-compose down
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

# Test the health endpoint
echo "Testing health endpoint..."
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'

echo -e "\n\nAPI fixes applied and services restarted successfully!"
echo "You can now test the upscale endpoint with:"
echo "python test-api.py --image <path-to-image> [--direct]"
