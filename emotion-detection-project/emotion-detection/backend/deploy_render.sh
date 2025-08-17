#!/bin/bash

# 🚀 Emotion Detection Backend - Render Deployment Script
# This script handles data file management during deployment

set -e  # Exit on any error

echo "🚀 Starting Emotion Detection Backend Deployment..."

# Create data directory if it doesn't exist
mkdir -p data

cd data

echo "📥 Checking for required data files..."

# Check if GloVe vectors exist
if [ ! -f "glove.2024.wikigiga.100d.zip" ]; then
    echo "⚠️  GloVe vectors not found. Attempting to download..."
    
    # Option 1: Try to download from Stanford (if available)
    if curl -f -L -o "glove.2024.wikigiga.100d.zip" \
        "https://nlp.stanford.edu/data/glove.2024.wikigiga.100d.zip"; then
        echo "✅ Successfully downloaded GloVe vectors from Stanford"
    else
        echo "❌ Could not download from Stanford. Please upload manually."
        echo "📋 Instructions:"
        echo "   1. Upload glove.2024.wikigiga.100d.zip to Render dashboard"
        echo "   2. Place in /app/data/ directory"
        echo "   3. Or use external storage (S3, Google Cloud, etc.)"
        exit 1
    fi
else
    echo "✅ GloVe vectors already present"
fi

# Check if extracted text file exists
if [ ! -f "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt" ]; then
    echo "📦 Extracting GloVe vectors..."
    
    if [ -f "glove.2024.wikigiga.100d.zip" ]; then
        unzip -o "glove.2024.wikigiga.100d.zip"
        echo "✅ GloVe vectors extracted successfully"
    else
        echo "❌ GloVe zip file not found. Cannot extract."
        exit 1
    fi
else
    echo "✅ GloVe text file already present"
fi

# Verify all required files exist
echo "🔍 Verifying data files..."

required_files=(
    "dialogues.json"
    "ontology.json"
    "glove.2024.wikigiga.100d.zip"
    "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - Found"
    else
        echo "❌ $file - Missing"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    echo "❌ Some required data files are missing!"
    echo "📋 Please ensure all data files are available before deployment."
    exit 1
fi

echo "✅ All data files verified successfully!"

# Go back to app directory
cd ..

echo "🐳 Building Docker image..."
docker build -t emotion-detection-backend .

echo "🚀 Starting container..."
docker run -d \
    --name emotion-detection-backend \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/models:/app/models:ro \
    emotion-detection-backend

echo "⏳ Waiting for service to start..."
sleep 10

# Health check
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Deployment successful! Service is healthy."
    echo "🌐 Backend running on http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/health"
else
    echo "❌ Health check failed. Service may not be running properly."
    echo "📋 Check logs with: docker logs emotion-detection-backend"
    exit 1
fi

echo "🎉 Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Set up environment variables in Render"
echo "   2. Configure your frontend to point to this backend"
echo "   3. Test the emotion detection API endpoints"
echo ""
echo "🔗 Useful endpoints:"
echo "   - Health: http://localhost:8000/health"
echo "   - API docs: http://localhost:8000/docs"
echo "   - Emotion detection: POST http://localhost:8000/api/detect-emotion"
