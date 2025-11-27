"""
Prediction utilities for crop pest detection.

This module provides functions for making predictions on single images
and batches of images using the trained model.
"""

import numpy as np
import json
from src.preprocessing import load_and_preprocess_image, load_and_preprocess_image_bytes


def predict_image(model, image_input, class_names, top_k=3, input_type='path'):
    """
    Predict pest class for a single image.
    
    Args:
        model: Trained Keras model
        image_input: Image file path or image bytes
        class_names: List of class names
        top_k: Number of top predictions to return
        input_type: Type of input - 'path' or 'bytes'
        
    Returns:
        Dictionary containing prediction results
    """
    # Preprocess image
    if input_type == 'path':
        img_array = load_and_preprocess_image(image_input)
    elif input_type == 'bytes':
        img_array = load_and_preprocess_image_bytes(image_input)
    else:
        raise ValueError("input_type must be 'path' or 'bytes'")
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get top K predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_probs = predictions[0][top_indices]
    
    # Prepare results
    result = {
        "predicted_class": class_names[top_indices[0]],
        "confidence": float(top_probs[0]),
        "top_predictions": [
            {
                "class": class_names[idx],
                "confidence": float(prob)
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
    }
    
    return result


def predict_batch(model, image_paths, class_names, top_k=3):
    """
    Predict pest classes for a batch of images.
    
    Args:
        model: Trained Keras model
        image_paths: List of image file paths
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_image(model, image_path, class_names, top_k, 'path')
            result['image_path'] = image_path
            result['status'] = 'success'
        except Exception as e:
            result = {
                'image_path': image_path,
                'status': 'error',
                'error': str(e)
            }
        
        results.append(result)
    
    return results


def get_prediction_probabilities(model, image_input, class_names, input_type='path'):
    """
    Get prediction probabilities for all classes.
    
    Args:
        model: Trained Keras model
        image_input: Image file path or image bytes
        class_names: List of class names
        input_type: Type of input - 'path' or 'bytes'
        
    Returns:
        Dictionary mapping class names to probabilities
    """
    # Preprocess image
    if input_type == 'path':
        img_array = load_and_preprocess_image(image_input)
    elif input_type == 'bytes':
        img_array = load_and_preprocess_image_bytes(image_input)
    else:
        raise ValueError("input_type must be 'path' or 'bytes'")
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Create class probability mapping
    probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, predictions[0])
    }
    
    return probabilities


def predict_with_confidence_threshold(model, image_input, class_names, 
                                      threshold=0.5, input_type='path'):
    """
    Predict with confidence threshold.
    
    Args:
        model: Trained Keras model
        image_input: Image file path or image bytes
        class_names: List of class names
        threshold: Minimum confidence threshold
        input_type: Type of input - 'path' or 'bytes'
        
    Returns:
        Dictionary containing prediction results or uncertainty flag
    """
    result = predict_image(model, image_input, class_names, top_k=3, input_type=input_type)
    
    if result['confidence'] < threshold:
        result['uncertain'] = True
        result['message'] = f"Confidence {result['confidence']:.2%} is below threshold {threshold:.2%}"
    else:
        result['uncertain'] = False
    
    return result


def format_prediction_response(prediction_result):
    """
    Format prediction result for API response.
    
    Args:
        prediction_result: Dictionary from predict_image function
        
    Returns:
        JSON-formatted prediction response
    """
    response = {
        "success": True,
        "prediction": {
            "class": prediction_result["predicted_class"],
            "confidence": round(prediction_result["confidence"], 4)
        },
        "top_3_predictions": [
            {
                "class": pred["class"],
                "confidence": round(pred["confidence"], 4)
            }
            for pred in prediction_result["top_predictions"][:3]
        ]
    }
    
    return response
