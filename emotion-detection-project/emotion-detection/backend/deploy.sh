#!/bin/bash

# Emotion Detection Backend Deployment Script
echo "🚀 Starting Emotion Detection Backend Deployment..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t emotion-detection-backend .

# Stop existing container if running
echo "🛑 Stopping existing container..."
docker stop emotion-detection-backend 2>/dev/null || true
docker rm emotion-detection-backend 2>/dev/null || true

# Run the new container
echo "▶️  Starting new container..."
docker run -d \
    --name emotion-detection-backend \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/models:/app/models:ro \
    emotion-detection-backend

echo "✅ Deployment complete!"
echo "🌐 Backend running on http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"

# Show container status
echo "📋 Container status:"
docker ps | grep emotion-detection-backend
