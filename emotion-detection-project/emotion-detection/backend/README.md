# Emotion Detection Backend

FastAPI backend for emotion detection using machine learning models and GloVe embeddings.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
# Build and run
docker build -t emotion-detection-backend .
docker run -p 8000:8000 emotion-detection-backend
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/predict` - Predict emotions from text
- `GET /api/models` - List available models

## Data Files

The system automatically downloads and extracts required GloVe vectors on first startup.

## Environment Variables

- `ENVIRONMENT` - Set to "production" or "development"
- `DEBUG` - Enable/disable debug mode
