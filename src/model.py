"""
Model building, training, and fine-tuning utilities for crop pest detection.

This module provides functions for creating, training, and fine-tuning
the MobileNetV2-based pest classification model.
"""

import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


def build_mobilenet_classifier(num_classes, img_size=(224, 224), trainable_base=False):
    """
    Build MobileNetV2-based classifier with transfer learning.
    
    Architecture:
    - MobileNetV2 base (frozen or trainable)
    - GlobalAveragePooling2D
    - BatchNormalization + Dropout(0.5)
    - Dense(512) + BatchNormalization + Dropout(0.3)
    - Dense(256) + Dropout(0.2)
    - Dense(num_classes, softmax)
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size as (height, width)
        trainable_base: Whether to make base model trainable
        
    Returns:
        Compiled Keras model
    """
    # Load MobileNetV2 with ImageNet weights
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3)
    )
    
    # Set base model trainability
    base_model.trainable = trainable_base
    
    # Build improved model with BatchNormalization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model


def create_callbacks(model_save_path, early_stopping_patience=10, 
                     reduce_lr_patience=5):
    """
    Create training callbacks for model checkpointing, early stopping, and LR reduction.
    
    Args:
        model_save_path: Path to save the best model
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        
    Returns:
        List of callbacks
    """
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, train_generator, validation_generator, epochs=50,
                callbacks=None):
    """
    Train the model on training data.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        
    Returns:
        Training history
    """
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history


def fine_tune_model(model, train_generator, validation_generator, 
                    unfreeze_layers=50, epochs=20, learning_rate=1e-5,
                    callbacks=None):
    """
    Fine-tune the model by unfreezing base layers.
    
    Args:
        model: Trained Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        unfreeze_layers: Number of base layers to unfreeze from the end
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        callbacks: List of Keras callbacks
        
    Returns:
        Fine-tuning history
    """
    # Get base model (first layer in Sequential)
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all except last unfreeze_layers
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - unfreeze_layers)
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    # Recompile with lower learning rate
    compile_model(model, learning_rate=learning_rate)
    
    # Fine-tune
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def load_model(model_path):
    """
    Load a saved Keras model with compatibility handling.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Custom objects for legacy model compatibility
    custom_objects = {
        'DepthwiseConv2D': keras.layers.DepthwiseConv2D,
        'relu6': keras.activations.relu,
    }
    
    try:
        # Method 1: Try loading with compile=False
        print("Attempting to load model (method 1: compile=False)...")
        model = keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=custom_objects
        )
        
        # Recompile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model loaded successfully (method 1)")
        return model
        
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        try:
            # Method 2: Load weights only
            print("Attempting to load model (method 2: weights only)...")
            
            # Try to rebuild the model architecture and load weights
            from tensorflow.keras.applications import MobileNetV2
            
            # Read model metadata to get num_classes
            metadata_path = os.path.join(os.path.dirname(model_path), 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    num_classes = metadata.get('num_classes', 9)
            else:
                num_classes = 9  # Default to 9 classes
            
            # Rebuild model architecture
            model = build_mobilenet_classifier(num_classes=num_classes)
            
            # Load weights
            model.load_weights(model_path)
            
            print("✓ Model loaded successfully (method 2: weights)")
            return model
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            try:
                # Method 3: Use TF SavedModel format compatibility
                print("Attempting to load model (method 3: SavedModel)...")
                
                # Convert .h5 to TF format temporarily if needed
                temp_dir = os.path.join(os.path.dirname(model_path), 'temp_model')
                
                # Try loading with tf.keras instead of keras
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print("✓ Model loaded successfully (method 3)")
                return model
                
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                raise Exception(f"All model loading methods failed. Last error: {e3}")


def save_model(model, model_path):
    """
    Save a Keras model.
    
    Args:
        model: Keras model to save
        model_path: Path where to save the model
    """
    model.save(model_path)
    print(f"Model saved to: {model_path}")


def save_class_names(class_names, save_path):
    """
    Save class names to JSON file.
    
    Args:
        class_names: List of class names
        save_path: Path to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to: {save_path}")


def load_class_names(load_path):
    """
    Load class names from JSON file.
    
    Args:
        load_path: Path to the JSON file
        
    Returns:
        List of class names
    """
    with open(load_path, 'r') as f:
        return json.load(f)


def save_model_metadata(metadata, save_path):
    """
    Save model metadata to JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        save_path: Path to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {save_path}")


def create_model_metadata(model, class_names, training_params, performance_metrics):
    """
    Create comprehensive model metadata.
    
    Args:
        model: Trained Keras model
        class_names: List of class names
        training_params: Dictionary of training parameters
        performance_metrics: Dictionary of performance metrics
        
    Returns:
        Dictionary containing model metadata
    """
    metadata = {
        "model_name": "Crop Pest Detection CNN",
        "created_date": datetime.now().isoformat(),
        "classes": class_names,
        "num_classes": len(class_names),
        "input_shape": list(model.input_shape[1:]),
        "architecture": "MobileNetV2 with Transfer Learning",
        "framework": "TensorFlow/Keras",
        "training_params": training_params,
        "architecture_details": {
            "base_model": "MobileNetV2",
            "dense_layers": [512, 256],
            "dropout_rates": [0.5, 0.3, 0.2],
            "batch_normalization": True,
            "global_pooling": "GlobalAveragePooling2D"
        },
        "performance": performance_metrics
    }
    return metadata


def evaluate_model(model, test_generator):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results = model.evaluate(test_generator, verbose=1)
    
    metrics = {
        "test_loss": float(results[0]),
        "test_accuracy": float(results[1]),
        "test_precision": float(results[2]),
        "test_recall": float(results[3])
    }
    
    # Calculate F1 score
    precision = metrics["test_precision"]
    recall = metrics["test_recall"]
    metrics["test_f1_score"] = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return metrics
