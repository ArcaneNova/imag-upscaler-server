#!/bin/bash

# Start Real-ESRGAN API Server locally for testing

echo "🚀 Starting Real-ESRGAN API Server..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to API server directory
cd api-server

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your Cloudinary credentials"
fi

# Create required directories
mkdir -p temp output weights

# Start services
echo "🏗️  Building and starting services..."
docker-compose up --build

echo "✅ API Server started!"
echo "📡 API available at: http://localhost:8000"
echo "🔍 Health check: http://localhost:8000/health"
echo "📊 Stats: http://localhost:8000/stats"
