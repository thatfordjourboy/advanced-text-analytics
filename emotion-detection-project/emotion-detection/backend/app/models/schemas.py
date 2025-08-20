"""Pydantic schemas for API requests and responses."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Request Models
class TextInput(BaseModel):
    """Input text for emotion detection."""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Enter the text you want to analyze for emotions", 
        example="I am so happy today! This is absolutely wonderful!", 
        title="Text to Analyze",
        widget="textarea"
    )
    model_preference: Optional[str] = Field(
        default="logistic_regression", 
        description="Choose which model to use for emotion detection.", 
        example="logistic_regression", 
        title="Model Selection", 
        enum=["logistic_regression", "random_forest", "both"],
        enum_names=["Logistic Regression", "Random Forest", "Both Models"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I am absolutely thrilled and overjoyed! This is the best day ever!",
                "model_preference": "logistic_regression"
            },
            "ui_schema": {
                "text": {
                    "ui:widget": "textarea",
                    "ui:placeholder": "Enter your text here...",
                    "ui:rows": 4
                },
                "model_preference": {
                    "ui:widget": "select",
                    "ui:placeholder": "Choose a model..."
                }
            }
        }
        schema_extra = {
            "properties": {
                "text": {
                    "x-ui:widget": "textarea",
                    "x-ui:placeholder": "Enter your text here...",
                    "x-ui:rows": 4
                },
                "model_preference": {
                    "x-ui:widget": "select",
                    "x-ui:placeholder": "Choose a model..."
                }
            }
        }

# Response Models
class EmotionPrediction(BaseModel):
    """Emotion detection result."""
    text: str = Field(..., description="Input text")
    emotions: Dict[str, float] = Field(..., description="Emotion probabilities")
    primary_emotion: str = Field(..., description="Highest probability emotion")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Name of the model used for prediction")
    model_confidence: Optional[Dict[str, float]] = Field(default=None, description="Confidence scores from both models if 'both' was requested")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the analysis")
    prediction_quality: Optional[str] = Field(default=None, description="Quality of prediction: 'high_confidence', 'uncertain', or 'low_confidence'")
    quality_message: Optional[str] = Field(default=None, description="Human-readable message about prediction quality")
    confidence_gap: Optional[float] = Field(default=None, description="Gap between highest and second-highest confidence scores")

class MultilineEmotionPrediction(BaseModel):
    """Multi-line emotion detection result."""
    text: str = Field(..., description="Input text")
    overall_emotions: Dict[str, float] = Field(..., description="Overall emotion probabilities across all paragraphs")
    overall_primary_emotion: str = Field(..., description="Overall primary emotion")
    overall_confidence: float = Field(..., description="Overall confidence score")
    sentence_analyses: List[Dict[str, Any]] = Field(..., description="Individual paragraph analyses")
    total_paragraphs: int = Field(..., description="Total number of paragraphs analyzed")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Name of the model used for prediction")
    analysis_type: str = Field(..., description="Type of analysis performed")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the analysis")

class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str = Field(..., description="Dataset name")
    total_samples: int = Field(..., description="Total number of samples")
    train_samples: int = Field(..., description="Training samples")
    validation_samples: int = Field(..., description="Validation samples")
    test_samples: int = Field(..., description="Test samples")
    emotion_distribution: Dict[str, int] = Field(..., description="Distribution of emotions")
    loaded: bool = Field(..., description="Whether dataset is loaded")

class SystemStatus(BaseModel):
    """Overall system status."""
    status: str = Field(..., description="System status: 'healthy', 'degraded', 'error'")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    dataset_loaded: bool = Field(..., description="Whether dataset is loaded")
    embeddings_loaded: bool = Field(..., description="Whether GloVe embeddings are loaded")
    last_check: str = Field(..., description="Last status check (ISO format)")
    real_data: Optional[Dict[str, Any]] = Field(default=None, description="Real-time system data and status")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp (ISO format)")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional health details")

# Training Progress Models
class TrainingProgress(BaseModel):
    """Training progress information."""
    model_type: str = Field(..., description="Type of model being trained")
    status: str = Field(..., description="Training status: 'not_started', 'in_progress', 'completed', 'failed'")
    start_time: Optional[str] = Field(None, description="Training start time (ISO format)")
    elapsed_time: Optional[float] = Field(None, description="Elapsed time in seconds")
    current_step: Optional[str] = Field(None, description="Current training step")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage (0-100)")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")
    current_fold: Optional[int] = Field(None, description="Current cross-validation fold")
    total_folds: Optional[int] = Field(None, description="Total cross-validation folds")
    current_params: Optional[Dict[str, Any]] = Field(None, description="Current hyperparameters being tested")
    best_score: Optional[float] = Field(None, description="Best score found so far")
    best_params: Optional[Dict[str, Any]] = Field(None, description="Best parameters found so far")
    messages: List[str] = Field(default_factory=list, description="Training progress messages")

class TrainingStatus(BaseModel):
    """Overall training status."""
    is_training: bool = Field(..., description="Whether any model is currently training")
    current_training: Optional[TrainingProgress] = Field(None, description="Current training progress")
    last_completed: Optional[str] = Field(None, description="Last completed training (ISO format)")
    models_available: Dict[str, bool] = Field(..., description="Availability of each model type")

# Error Models
class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp (ISO format)")
    path: str = Field(..., description="Request path that caused error")
