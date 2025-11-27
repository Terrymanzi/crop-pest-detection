"""
Image preprocessing utilities for crop pest detection.

This module provides functions for loading, preprocessing, and augmenting images
for the MobileNetV2-based pest classification model.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io


class DataPreprocessor:
    """Enhanced data preprocessor for pest classification with MobileNetV2."""
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize the data preprocessor.
        
        Args:
            img_size: Target image size as (height, width)
            batch_size: Batch size for data generators
        """
        self.img_size = img_size
        self.batch_size = batch_size
    
    def create_train_generator(self):
        """
        Create training data generator with augmentation.
        
        Returns:
            ImageDataGenerator configured for training
        """
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    
    def create_test_generator(self):
        """
        Create test/validation data generator without augmentation.
        
        Returns:
            ImageDataGenerator configured for testing
        """
        return ImageDataGenerator(preprocessing_function=preprocess_input)
    
    def load_train_data(self, train_dir):
        """
        Load training and validation data from directory.
        
        Args:
            train_dir: Path to training data directory
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        train_datagen = self.create_train_generator()
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        return train_gen, val_gen
    
    def load_test_data(self, test_dir):
        """
        Load test data from directory.
        
        Args:
            test_dir: Path to test data directory
            
        Returns:
            Test data generator
        """
        test_datagen = self.create_test_generator()
        return test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target image size as (height, width)
        
    Returns:
        Preprocessed image array ready for model input
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def load_and_preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    """
    Load and preprocess an image from bytes for prediction.
    
    Args:
        image_bytes: Image data as bytes
        target_size: Target image size as (height, width)
        
    Returns:
        Preprocessed image array ready for model input
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if needed (handles PNG with alpha, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to array and preprocess
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array


def create_dataset_from_directory(directory, img_size=(224, 224), batch_size=32, 
                                   shuffle=True, augment=False):
    """
    Create a dataset from a directory structure.
    
    Args:
        directory: Path to data directory with class subdirectories
        img_size: Target image size as (height, width)
        batch_size: Batch size for the generator
        shuffle: Whether to shuffle the data
        augment: Whether to apply data augmentation
        
    Returns:
        Data generator
    """
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle
    )
    
    return generator


def denormalize_image(img_array):
    """
    Denormalize an image preprocessed with MobileNetV2 preprocessing.
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        Denormalized image array (0-255 range)
    """
    img = img_array.copy()
    img += 1
    img *= 127.5
    return np.clip(img, 0, 255).astype(np.uint8)
