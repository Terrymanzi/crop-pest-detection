"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
from PIL import Image
import io
import os
import tempfile
import shutil

from src.preprocessing import (
    DataPreprocessor,
    load_and_preprocess_image,
    load_and_preprocess_image_bytes,
    denormalize_image
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """Save sample image to temporary path."""
    img_path = os.path.join(temp_dir, "test_image.jpg")
    sample_image.save(img_path)
    return img_path


def test_data_preprocessor_init():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
    assert preprocessor.img_size == (224, 224)
    assert preprocessor.batch_size == 32


def test_create_train_generator():
    """Test creation of training data generator."""
    preprocessor = DataPreprocessor()
    generator = preprocessor.create_train_generator()
    assert generator is not None
    assert hasattr(generator, 'flow_from_directory')


def test_create_test_generator():
    """Test creation of test data generator."""
    preprocessor = DataPreprocessor()
    generator = preprocessor.create_test_generator()
    assert generator is not None
    assert hasattr(generator, 'flow_from_directory')


def test_load_and_preprocess_image(sample_image_path):
    """Test loading and preprocessing an image from path."""
    img_array = load_and_preprocess_image(sample_image_path, target_size=(224, 224))
    
    # Check shape
    assert img_array.shape == (1, 224, 224, 3)
    
    # Check preprocessing (MobileNetV2 preprocessing scales to [-1, 1])
    assert img_array.min() >= -1.0
    assert img_array.max() <= 1.0


def test_load_and_preprocess_image_bytes(sample_image_bytes):
    """Test loading and preprocessing an image from bytes."""
    img_array = load_and_preprocess_image_bytes(sample_image_bytes, target_size=(224, 224))
    
    # Check shape
    assert img_array.shape == (1, 224, 224, 3)
    
    # Check preprocessing
    assert img_array.min() >= -1.0
    assert img_array.max() <= 1.0


def test_load_and_preprocess_image_bytes_grayscale():
    """Test loading and preprocessing a grayscale image."""
    # Create grayscale image
    img = Image.new('L', (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    
    # Process
    img_array = load_and_preprocess_image_bytes(img_bytes, target_size=(224, 224))
    
    # Should be converted to RGB
    assert img_array.shape == (1, 224, 224, 3)


def test_load_and_preprocess_image_bytes_rgba():
    """Test loading and preprocessing an RGBA image."""
    # Create RGBA image
    img = Image.new('RGBA', (224, 224), color=(100, 150, 200, 255))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    
    # Process
    img_array = load_and_preprocess_image_bytes(img_bytes, target_size=(224, 224))
    
    # Should be converted to RGB
    assert img_array.shape == (1, 224, 224, 3)


def test_denormalize_image():
    """Test denormalizing a preprocessed image."""
    # Create a preprocessed image array (scaled to [-1, 1])
    img_array = np.random.uniform(-1, 1, size=(224, 224, 3))
    
    # Denormalize
    denorm = denormalize_image(img_array)
    
    # Check range [0, 255]
    assert denorm.min() >= 0
    assert denorm.max() <= 255
    assert denorm.dtype == np.uint8


def test_load_and_preprocess_different_sizes():
    """Test preprocessing with different target sizes."""
    img = Image.new('RGB', (500, 500), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    
    # Test different sizes
    for size in [(128, 128), (224, 224), (299, 299)]:
        img_array = load_and_preprocess_image_bytes(img_bytes, target_size=size)
        assert img_array.shape == (1, size[0], size[1], 3)


def test_preprocessor_batch_size():
    """Test that batch size is correctly set."""
    batch_sizes = [16, 32, 64]
    
    for bs in batch_sizes:
        preprocessor = DataPreprocessor(batch_size=bs)
        assert preprocessor.batch_size == bs
