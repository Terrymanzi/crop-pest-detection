"""
Tests for model module.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import json

from src.model import (
    build_mobilenet_classifier,
    compile_model,
    save_model,
    load_model,
    save_class_names,
    load_class_names,
    create_model_metadata
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_class_names():
    """Sample class names for testing."""
    return ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper']


def test_build_mobilenet_classifier():
    """Test building MobileNetV2 classifier."""
    num_classes = 9
    model = build_mobilenet_classifier(num_classes, img_size=(224, 224))
    
    # Check model structure
    assert model is not None
    assert len(model.layers) > 0
    
    # Check input shape
    assert model.input_shape == (None, 224, 224, 3)
    
    # Check output shape
    assert model.output_shape == (None, num_classes)


def test_build_mobilenet_classifier_trainable():
    """Test building classifier with trainable base."""
    model = build_mobilenet_classifier(5, trainable_base=True)
    
    # Base model should be trainable
    assert model.layers[0].trainable == True


def test_build_mobilenet_classifier_frozen():
    """Test building classifier with frozen base."""
    model = build_mobilenet_classifier(5, trainable_base=False)
    
    # Base model should be frozen
    assert model.layers[0].trainable == False


def test_compile_model():
    """Test model compilation."""
    model = build_mobilenet_classifier(5)
    compiled_model = compile_model(model, learning_rate=0.001)
    
    # Check that model is compiled
    assert compiled_model.optimizer is not None
    assert compiled_model.loss is not None


def test_save_and_load_model(temp_dir):
    """Test saving and loading a model."""
    # Build and compile model
    model = build_mobilenet_classifier(5)
    model = compile_model(model)
    
    # Save model
    model_path = os.path.join(temp_dir, 'test_model.h5')
    save_model(model, model_path)
    
    # Check file exists
    assert os.path.exists(model_path)
    
    # Load model
    loaded_model = load_model(model_path)
    
    # Check loaded model
    assert loaded_model is not None
    assert loaded_model.input_shape == model.input_shape
    assert loaded_model.output_shape == model.output_shape


def test_save_and_load_class_names(temp_dir, sample_class_names):
    """Test saving and loading class names."""
    # Save class names
    class_names_path = os.path.join(temp_dir, 'class_names.json')
    save_class_names(sample_class_names, class_names_path)
    
    # Check file exists
    assert os.path.exists(class_names_path)
    
    # Load class names
    loaded_classes = load_class_names(class_names_path)
    
    # Check loaded classes
    assert loaded_classes == sample_class_names


def test_create_model_metadata(sample_class_names):
    """Test creating model metadata."""
    model = build_mobilenet_classifier(len(sample_class_names))
    
    training_params = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    performance_metrics = {
        "test_accuracy": 0.95,
        "test_precision": 0.94,
        "test_recall": 0.93,
        "test_f1_score": 0.935
    }
    
    metadata = create_model_metadata(
        model,
        sample_class_names,
        training_params,
        performance_metrics
    )
    
    # Check metadata structure
    assert 'model_name' in metadata
    assert 'classes' in metadata
    assert 'num_classes' in metadata
    assert metadata['num_classes'] == len(sample_class_names)
    assert metadata['classes'] == sample_class_names
    assert 'training_params' in metadata
    assert 'performance' in metadata


def test_model_prediction_shape():
    """Test that model produces correct prediction shape."""
    num_classes = 9
    model = build_mobilenet_classifier(num_classes)
    model = compile_model(model)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # Make prediction
    prediction = model.predict(dummy_input, verbose=0)
    
    # Check prediction shape
    assert prediction.shape == (1, num_classes)
    
    # Check prediction sums to 1 (softmax output)
    assert np.allclose(prediction.sum(axis=1), 1.0, atol=1e-5)


def test_different_image_sizes():
    """Test building models with different image sizes."""
    sizes = [(128, 128), (224, 224), (299, 299)]
    
    for size in sizes:
        model = build_mobilenet_classifier(5, img_size=size)
        assert model.input_shape == (None, size[0], size[1], 3)
