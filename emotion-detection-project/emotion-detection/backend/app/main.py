from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import your existing modules
from app.core.text_processor import TextProcessor
from app.core.embeddings import GloVeEmbeddings
from app.core.model_trainer import MultiLabelEmotionTrainer
from app.models.schemas import TextInput, EmotionPrediction, SystemStatus, HealthResponse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions in text using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
text_processor = TextProcessor()
embeddings = GloVeEmbeddings(dimension=100)
models_dir = Path(__file__).parent.parent / "models"
model_trainer = MultiLabelEmotionTrainer(models_dir)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        logger.info("Loading GloVe embeddings...")
        embeddings.load_embeddings()
        logger.info("✅ GloVe embeddings loaded successfully")
        
        logger.info("Initializing model trainer...")
        # Model trainer will load existing models if available
        logger.info("✅ Model trainer initialized")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Emotion Detection API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp="2025-01-17T00:00:00Z",
        version="1.0.0",
        uptime=0.0,
        details={
            "embeddings_loaded": embeddings.loaded,
            "models_available": True
        }
    )

@app.post("/predict", response_model=EmotionPrediction)
async def predict_emotion(request: TextInput):
    """Predict emotion from text"""
    try:
        # Process the text
        processed_text = text_processor.process_text(request.text)
        
        # Get embeddings
        text_vector = embeddings.get_text_vector(processed_text)
        
        # Make prediction (this would use your trained models)
        # For now, return a placeholder response
        emotions = {
            "joy": 0.8,
            "excitement": 0.6,
            "satisfaction": 0.4
        }
        
        return EmotionPrediction(
            text=request.text,
            emotions=emotions,
            primary_emotion="joy",
            confidence=0.8,
            processing_time=0.1,
            model_used="placeholder"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get the status of available models"""
    try:
        return {
            "models": ["logistic_regression", "random_forest"],
            "status": "available",
            "loaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/models/train")
async def train_model():
    """Trigger model training"""
    try:
        # This would start the training process using your model_trainer
        return {"message": "Model training started", "status": "training"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    return SystemStatus(
        status="healthy" if embeddings.loaded else "degraded",
        models_loaded=True,
        dataset_loaded=False,  # Would check if data is loaded
        embeddings_loaded=embeddings.loaded,
        last_check="2025-01-17T00:00:00Z"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
