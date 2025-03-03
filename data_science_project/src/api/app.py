from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.model_io import ModelSerializer
from utils.monitoring import PerformanceMonitor
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Model Prediction API",
    description="API for model predictions and monitoring",
    version="1.0.0"
)

# Initialize components
model_serializer = ModelSerializer()
performance_monitor = PerformanceMonitor()

class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: List[float]
    model_name: str
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: float
    probability: float
    model_version: str
    prediction_id: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    instances: List[List[float]]
    model_name: str
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[float]
    probabilities: List[float]
    model_version: str
    prediction_id: str
    timestamp: str

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Model Prediction API"}

@app.get("/models")
def list_models():
    """List available models."""
    try:
        models = []
        # List all model files
        for path in model_serializer.model_dir.glob("**/model.joblib"):
            models.append(path.parent.name)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error listing models"
        )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make a single prediction."""
    try:
        # Load model
        model, metadata = model_serializer.load_model(
            request.model_name,
            request.model_version
        )
        
        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        # Generate response
        timestamp = datetime.now().isoformat()
        prediction_id = f"pred_{timestamp}"
        
        response = PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=metadata.get("version", "unknown"),
            prediction_id=prediction_id,
            timestamp=timestamp
        )
        
        # Log metrics
        performance_monitor.log_metrics(
            metrics={"prediction_count": 1},
            model_name=request.model_name,
            model_version=metadata.get("version", "unknown")
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error making prediction"
        )

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        # Load model
        model, metadata = model_serializer.load_model(
            request.model_name,
            request.model_version
        )
        
        # Make predictions
        features = np.array(request.instances)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features).max(axis=1)
        
        # Generate response
        timestamp = datetime.now().isoformat()
        prediction_id = f"batch_{timestamp}"
        
        response = BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            probabilities=[float(p) for p in probabilities],
            model_version=metadata.get("version", "unknown"),
            prediction_id=prediction_id,
            timestamp=timestamp
        )
        
        # Log metrics
        performance_monitor.log_metrics(
            metrics={"prediction_count": len(request.instances)},
            model_name=request.model_name,
            model_version=metadata.get("version", "unknown")
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error making batch predictions"
        )

@app.get("/metrics/{model_name}")
def get_model_metrics(model_name: str):
    """Get model performance metrics."""
    try:
        metrics = performance_monitor.get_metrics_history(model_name)
        return {"metrics": metrics.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error getting metrics"
        )