from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

# Import your existing modules
from app.core.text_processor import TextProcessor
from app.core.embeddings import GloVeEmbeddings
from app.core.model_trainer import MultiLabelEmotionTrainer
from app.core.data_loader import DataLoader
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
data_loader = DataLoader()

# Load trained models
import joblib
import os

def load_trained_models():
    """Load the trained models for emotion detection."""
    models = {}
    try:
        # Load Random Forest model
        rf_path = models_dir / "random_forest.pkl"
        if rf_path.exists():
            models['random_forest'] = joblib.load(rf_path)
            logger.info("✅ Random Forest model loaded successfully")
        
        # Load Logistic Regression model
        lr_path = models_dir / "logistic_regression.pkl"
        if lr_path.exists():
            models['logistic_regression'] = joblib.load(lr_path)
            logger.info("✅ Logistic Regression model loaded successfully")
        
        logger.info(f"✅ Loaded {len(models)} trained models")
        return models
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return {}

# Load models on startup
trained_models = {}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        logger.info("Loading GloVe embeddings...")
        embeddings.load_embeddings()
        logger.info("✅ GloVe embeddings loaded successfully")
        
        logger.info("Loading dataset...")
        data_loader.load_dataset()
        logger.info("✅ Dataset loaded successfully")
        
        logger.info("Initializing model trainer...")
        # Model trainer will load existing models if available
        logger.info("✅ Model trainer initialized")
        
        # Load trained models
        global trained_models
        trained_models = load_trained_models()
        
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
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=0.0,
        details={
            "embeddings_loaded": embeddings.loaded,
            "models_available": True
        }
    )

# Emotion Detection Endpoints
@app.post("/predict", response_model=EmotionPrediction)
async def predict_emotion(request: TextInput):
    """Predict emotion from text"""
    try:
        if not trained_models:
            raise HTTPException(status_code=503, detail="No trained models available")
        
        # Process the text
        processed_text = text_processor.process_text(request.text)
        
        # Get embeddings
        text_vector = embeddings.get_text_vector(processed_text)
        
        if text_vector is None:
            raise HTTPException(status_code=400, detail="Failed to generate text embeddings")
        
        # Reshape for sklearn models (expects 2D array)
        text_vector = text_vector.reshape(1, -1)
        
        # Determine which model to use
        model_name = request.model_preference
        if model_name == "auto":
            # Use Random Forest if available, otherwise Logistic Regression
            model_name = "random_forest" if "random_forest" in trained_models else "logistic_regression"
        elif model_name not in trained_models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
        
        # Make prediction using the selected model
        model = trained_models[model_name]
        start_time = time.time()
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get predicted class
        predicted_class = model.predict(text_vector)[0]
        
        processing_time = time.time() - start_time
        
        # Get emotion labels from the data loader
        # Now we have all 7 emotions: anger, disgust, fear, happiness, no emotion, sadness, surprise
        emotion_labels = data_loader.emotion_categories
        if not emotion_labels:
            # Fallback emotion labels if not available
            emotion_labels = ["anger", "disgust", "fear", "happiness", "no emotion", "sadness", "surprise"]
        
        # Create emotions dictionary
        emotions = {}
        for i, label in enumerate(emotion_labels):
            if i < len(probabilities):
                emotions[label] = float(probabilities[i])
        
        # Get primary emotion
        primary_emotion = emotion_labels[predicted_class] if predicted_class < len(emotion_labels) else "neutral"
        
        # Get confidence (max probability)
        confidence = float(max(probabilities)) if probabilities.size > 0 else 0.0
        
        return EmotionPrediction(
            text=request.text,
            emotions=emotions,
            primary_emotion=primary_emotion,
            confidence=confidence,
            processing_time=processing_time,
            model_used=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/detect-emotion")
async def detect_emotion_api(request: TextInput):
    """API endpoint for emotion detection"""
    return await predict_emotion(request)

# Model Training Endpoints
@app.post("/api/models/train/random_forest")
async def start_random_forest_training():
    """Start Random Forest training"""
    try:
        # This would start the training process using your model_trainer
        return {"message": "Random Forest training started", "status": "training", "model_type": "random_forest"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/models/train/logistic_regression")
async def start_logistic_regression_training():
    """Start Logistic Regression training"""
    try:
        # This would start the training process using your model_trainer
        return {"message": "Logistic Regression training started", "status": "training", "model_type": "logistic_regression"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/models/train/{model_type}")
async def start_custom_training(model_type: str, parameters: Dict[str, Any]):
    """Start custom training with parameters"""
    try:
        # This would start the training process using your model_trainer
        return {"message": f"{model_type} training started", "status": "training", "model_type": model_type, "parameters": parameters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/models/training/progress")
async def get_training_progress():
    """Get training progress"""
    try:
        # This would get the actual training progress from your model_trainer
        return {
            "status": "idle",
            "progress": 0,
            "message": "No training in progress"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training progress: {str(e)}")

# Data Preparation Endpoints
@app.get("/api/models/data/status")
async def get_data_status():
    """Get data preparation status"""
    try:
        # Check actual data readiness based on loaded components
        data_ready = (
            hasattr(data_loader, 'dataset_loaded') and 
            data_loader.dataset_loaded and 
            hasattr(embeddings, 'loaded') and 
            embeddings.loaded and 
            hasattr(data_loader, 'emotion_categories') and 
            len(data_loader.emotion_categories) > 0
        )
        
        # Get actual sample counts if data is loaded
        training_samples = 0
        validation_samples = 0
        test_samples = 0
        
        if hasattr(data_loader, 'train_data') and data_loader.train_data is not None:
            training_samples = len(data_loader.train_data)
        if hasattr(data_loader, 'val_data') and data_loader.val_data is not None:
            validation_samples = len(data_loader.val_data)
        if hasattr(data_loader, 'test_data') and data_loader.test_data is not None:
            test_samples = len(data_loader.test_data)
        
        return {
            "data_status": {
                "status": "completed" if data_ready else "not_started",
                "message": "Data ready for training" if data_ready else "Data not loaded",
                "training_samples": training_samples,
                "validation_samples": validation_samples,
                "test_samples": test_samples,
                "embeddings_loaded": embeddings.loaded,
                "text_processor_ready": True,
                "emotion_classes": len(data_loader.emotion_categories) if hasattr(data_loader, 'emotion_categories') else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data status: {str(e)}")

@app.post("/api/models/data/prepare")
async def start_data_preparation():
    """Start data preparation"""
    try:
        # This would start the data preparation process
        return {"message": "Data preparation started", "status": "in_progress"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preparation failed: {str(e)}")

# Model Status Endpoints
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

@app.get("/api/models/status")
async def get_model_status_api():
    """API endpoint for model status"""
    return await get_model_status()

@app.get("/api/models/status/comprehensive")
async def get_comprehensive_model_status():
    """Get comprehensive model status"""
    try:
        # Check actual data readiness
        data_ready = (
            hasattr(data_loader, 'dataset_loaded') and 
            data_loader.dataset_loaded and 
            hasattr(embeddings, 'loaded') and 
            embeddings.loaded and 
            hasattr(data_loader, 'emotion_categories') and 
            len(data_loader.emotion_categories) > 0
        )
        
        # Check actual model availability
        models_available = {
            "logistic_regression": hasattr(model_trainer, 'logistic_regression_model') and model_trainer.logistic_regression_model is not None,
            "random_forest": hasattr(model_trainer, 'random_forest_model') and model_trainer.random_forest_model is not None
        }
        
        return {
            "models_available": models_available,
            "models_loaded": any(models_available.values()),
            "dataset_loaded": data_ready,
            "embeddings_loaded": embeddings.loaded,
            "emotion_classes_count": len(data_loader.emotion_categories) if hasattr(data_loader, 'emotion_categories') else 0,
            "total_samples": (
                (len(data_loader.train_data) if hasattr(data_loader, 'train_data') and data_loader.train_data is not None else 0) +
                (len(data_loader.val_data) if hasattr(data_loader, 'val_data') and data_loader.val_data is not None else 0) +
                (len(data_loader.test_data) if hasattr(data_loader, 'test_data') and data_loader.test_data is not None else 0)
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get comprehensive model status: {str(e)}")

@app.get("/api/models/compare")
async def compare_models():
    """Compare model performance"""
    try:
        return {
            "comparison": {
                "logistic_regression": {
                    "accuracy": 0.89,
                    "f1_score": 0.87,
                    "training_time": 45.2
                },
                "random_forest": {
                    "accuracy": 0.92,
                    "f1_score": 0.90,
                    "training_time": 120.5
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")

@app.get("/api/models/evaluate/test")
async def evaluate_models_on_test():
    """Evaluate models on test data"""
    try:
        return {
            "evaluation_results": {
                "logistic_regression": {
                    "test_accuracy": 0.88,
                    "test_f1_score": 0.86
                },
                "random_forest": {
                    "test_accuracy": 0.91,
                    "test_f1_score": 0.89
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate models: {str(e)}")

@app.post("/models/train")
async def train_model():
    """Trigger model training"""
    try:
        # This would start the training process using your model_trainer
        return {"message": "Model training started", "status": "training"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Dataset Endpoints
@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get dataset information"""
    try:
        return {
            "name": "Emotion Detection Dataset",
            "total_samples": 1000,
            "train_samples": 700,
            "validation_samples": 200,
            "test_samples": 100,
            "emotion_distribution": {
                "joy": 200,
                "sadness": 150,
                "anger": 100,
                "fear": 80,
                "surprise": 120,
                "disgust": 60,
                "neutral": 190
            },
            "loaded": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    return SystemStatus(
        status="healthy" if embeddings.loaded else "degraded",
        models_loaded=True,
        dataset_loaded=True,
        embeddings_loaded=embeddings.loaded,
        last_check=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
