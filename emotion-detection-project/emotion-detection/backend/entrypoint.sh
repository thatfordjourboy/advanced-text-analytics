#!/bin/bash

# Entrypoint script for Emotion Detection Backend
echo "Starting Emotion Detection Backend..."

# Run the startup script first
echo "Running startup script to download/extract data files..."
python startup.py

# Check if startup was successful
if [ $? -eq 0 ]; then
    echo "SUCCESS: Startup script completed successfully"
    echo "Starting FastAPI application..."
    
    # Start the FastAPI application
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
else
    echo "ERROR: Startup script failed"
    echo "Checking what went wrong..."
    
    # Show data directory contents for debugging
    if [ -d "/app/data" ]; then
        echo "Contents of /app/data:"
        ls -la /app/data/
    fi
    
    if [ -d "/opt/render/project/src/emotion-detection-project/emotion-detection/backend/data" ]; then
        echo "Contents of Render data directory:"
        ls -la /opt/render/project/src/emotion-detection-project/emotion-detection/backend/data/
    fi
    
    exit 1
fi
