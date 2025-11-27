"""
FastAPI backend for crop pest detection.

This module provides REST API endpoints for:
- Health checks
- Image prediction
- Model retraining
- Metrics monitoring
"""

import os
import sys
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_model, load_class_names
from src.prediction import predict_image, format_prediction_response
from scripts.monitor import (
    get_monitor, record_request, record_prediction, 
    record_retrain, get_metrics, get_health_status
)

# Initialize FastAPI app
app = FastAPI(
    title="Crop Pest Detection API",
    description="API for detecting crop pests using MobileNetV2-based image classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class names
MODEL = None
CLASS_NAMES = None
MODEL_LOADED = False
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "crop_pest_model_finetuned.h5")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")


def load_model_and_classes():
    """Load model and class names on startup."""
    global MODEL, CLASS_NAMES, MODEL_LOADED
    
    try:
        print("Loading model...")
        print(f"TensorFlow version: {tf.__version__}")
        
        # Try multiple model files in order of preference
        model_files = [
            MODEL_PATH,  # crop_pest_model_finetuned.h5
            os.path.join(MODELS_DIR, "crop_pest_model.h5"),  # fallback
        ]
        
        model_loaded_successfully = False
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    print(f"Attempting to load: {model_file}")
                    MODEL = load_model(model_file)
                    print(f"✓ Model loaded from: {model_file}")
                    model_loaded_successfully = True
                    break
                except Exception as e:
                    print(f"✗ Failed to load {model_file}: {e}")
                    continue
        
        if not model_loaded_successfully:
            print("⚠ No model could be loaded!")
            MODEL_LOADED = False
            return
        
        print("Loading class names...")
        CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)
        print(f"✓ Class names loaded: {CLASS_NAMES}")
        
        MODEL_LOADED = True
        print("✓ Model and class names loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        MODEL_LOADED = False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup."""
    load_model_and_classes()


# Pydantic models
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    prediction: dict
    top_3_predictions: list


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    uptime_seconds: float
    model_loaded: bool
    total_requests: int
    average_latency_ms: float


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    uptime_seconds: float
    uptime_formatted: str
    total_requests: int
    total_errors: int
    error_rate: float
    average_latency_ms: float
    recent_average_latency_ms: float
    requests_per_minute: float


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Crop Pest Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST)",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status including uptime and model loaded status
    """
    start_time = time.time()
    
    try:
        health = get_health_status()
        health['model_loaded'] = MODEL_LOADED
        
        latency = time.time() - start_time
        record_request("/health", latency)
        
        return health
    
    except Exception as e:
        latency = time.time() - start_time
        record_request("/health", latency, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Predict pest class from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with top-3 probabilities
    """
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_LOADED or MODEL is None or CLASS_NAMES is None:
        latency = time.time() - start_time
        record_request("/predict", latency, error=True)
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        latency = time.time() - start_time
        record_request("/predict", latency, error=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = predict_image(
            MODEL, 
            image_bytes, 
            CLASS_NAMES, 
            top_k=3, 
            input_type='bytes'
        )
        
        # Record prediction
        record_prediction(result['predicted_class'])
        
        # Format response
        response = format_prediction_response(result)
        
        # Record metrics
        latency = time.time() - start_time
        record_request("/predict", latency)
        
        return response
    
    except Exception as e:
        latency = time.time() - start_time
        record_request("/predict", latency, error=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain", tags=["Training"])
async def retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Trigger model retraining with new data.
    
    Accepts a ZIP file containing training images organized in class folders.
    
    Args:
        file: ZIP file with training data
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.endswith('.zip'):
        latency = time.time() - start_time
        record_request("/retrain", latency, error=True)
        raise HTTPException(
            status_code=400,
            detail="File must be a ZIP archive"
        )
    
    try:
        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "training_data.zip")
        extract_dir = os.path.join(temp_dir, "extracted")
        
        # Save uploaded ZIP file
        with open(zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the actual data directory (handle nested folders)
        data_dirs = []
        for root, dirs, files in os.walk(extract_dir):
            # Check if this directory contains class folders with images
            has_images = False
            for d in dirs:
                class_dir = os.path.join(root, d)
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if image_files:
                    has_images = True
                    break
            if has_images:
                data_dirs.append(root)
                break
        
        if not data_dirs:
            shutil.rmtree(temp_dir)
            latency = time.time() - start_time
            record_request("/retrain", latency, error=True)
            raise HTTPException(
                status_code=400,
                detail="ZIP file does not contain valid training data structure"
            )
        
        new_data_dir = data_dirs[0]
        
        # Trigger retraining in background
        background_tasks.add_task(
            run_retrain_script,
            new_data_dir,
            temp_dir
        )
        
        # Record retrain event
        record_retrain()
        
        # Record metrics
        latency = time.time() - start_time
        record_request("/retrain", latency)
        
        return {
            "success": True,
            "message": "Retraining started in background",
            "data_dir": new_data_dir
        }
    
    except HTTPException:
        raise
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        latency = time.time() - start_time
        record_request("/retrain", latency, error=True)
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")


def run_retrain_script(data_dir: str, temp_dir: str):
    """
    Run the retraining script in the background.
    
    Args:
        data_dir: Directory containing new training data
        temp_dir: Temporary directory to clean up after
    """
    try:
        print(f"Starting retraining with data from: {data_dir}")
        
        # Run retrain script
        result = subprocess.run(
            [
                sys.executable,
                "scripts/retrain_job.py",
                "--new_data_dir", data_dir,
                "--epochs", "10",
                "--learning_rate", "1e-5"
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("✓ Retraining completed successfully!")
            print(result.stdout)
            
            # Reload model
            load_model_and_classes()
        else:
            print("✗ Retraining failed!")
            print(result.stderr)
    
    except Exception as e:
        print(f"✗ Retraining error: {e}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print("✓ Temporary files cleaned up")
        except Exception as e:
            print(f"⚠ Error cleaning up temp dir: {e}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics_endpoint():
    """
    Get API metrics.
    
    Returns:
        Comprehensive metrics including request counts, latency, and predictions
    """
    start_time = time.time()
    
    try:
        metrics = get_metrics()
        
        latency = time.time() - start_time
        record_request("/metrics", latency)
        
        return metrics
    
    except Exception as e:
        latency = time.time() - start_time
        record_request("/metrics", latency, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes", tags=["Model Info"])
async def get_classes():
    """
    Get list of supported pest classes.
    
    Returns:
        List of class names
    """
    if CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
