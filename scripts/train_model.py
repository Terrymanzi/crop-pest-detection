"""
Training script for crop pest detection model.

This script trains a MobileNetV2-based classifier from scratch on the pest dataset.
Usage: python scripts/train_model.py
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.model import (
    build_mobilenet_classifier,
    compile_model,
    create_callbacks,
    train_model,
    fine_tune_model,
    save_model,
    save_class_names,
    save_model_metadata,
    create_model_metadata,
    evaluate_model
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train crop pest detection model')
    parser.add_argument('--train_dir', type=str, default='data/train',
                       help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Path to test data directory')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of initial training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                       help='Fine-tuning learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (width and height)')
    parser.add_argument('--skip_fine_tune', action='store_true',
                       help='Skip fine-tuning phase')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    print("="*60)
    print("CROP PEST DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Training directory: {args.train_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial epochs: {args.epochs}")
    print(f"Fine-tune epochs: {args.fine_tune_epochs}")
    print("="*60 + "\n")
    
    # Initialize preprocessor
    img_size = (args.img_size, args.img_size)
    preprocessor = DataPreprocessor(img_size=img_size, batch_size=args.batch_size)
    
    # Load data
    print("Loading training and validation data...")
    train_generator, validation_generator = preprocessor.load_train_data(args.train_dir)
    
    print("\nLoading test data...")
    test_generator = preprocessor.load_test_data(args.test_dir)
    
    # Get class information
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {validation_generator.samples}")
    print(f"  Test samples: {test_generator.samples}")
    print(f"  Class names: {class_names}\n")
    
    # Build model
    print("Building model...")
    model = build_mobilenet_classifier(num_classes, img_size)
    model = compile_model(model, learning_rate=args.learning_rate)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    # Create callbacks
    initial_model_path = os.path.join(args.models_dir, 'crop_pest_model.h5')
    callbacks = create_callbacks(initial_model_path)
    
    # Train model
    print("="*60)
    print("STARTING INITIAL TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60 + "\n")
    
    history = train_model(
        model,
        train_generator,
        validation_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print("\n" + "="*60)
    print("INITIAL TRAINING COMPLETED")
    print("="*60)
    
    # Evaluate initial model
    print("\nEvaluating initial model on test set...")
    initial_metrics = evaluate_model(model, test_generator)
    
    print("\n" + "="*60)
    print("INITIAL MODEL PERFORMANCE")
    print("="*60)
    print(f"Test Accuracy:  {initial_metrics['test_accuracy']*100:.2f}%")
    print(f"Test Precision: {initial_metrics['test_precision']*100:.2f}%")
    print(f"Test Recall:    {initial_metrics['test_recall']*100:.2f}%")
    print(f"Test F1-Score:  {initial_metrics['test_f1_score']*100:.2f}%")
    print(f"Test Loss:      {initial_metrics['test_loss']:.4f}")
    print("="*60 + "\n")
    
    # Fine-tune model (optional)
    if not args.skip_fine_tune:
        print("="*60)
        print("STARTING FINE-TUNING")
        print("="*60)
        print(f"Unfreezing last 50 layers")
        print(f"Fine-tuning epochs: {args.fine_tune_epochs}")
        print(f"Fine-tuning learning rate: {args.fine_tune_lr}")
        print("="*60 + "\n")
        
        fine_tuned_model_path = os.path.join(args.models_dir, 'crop_pest_model_finetuned.h5')
        fine_tune_callbacks = create_callbacks(fine_tuned_model_path, 
                                               early_stopping_patience=7,
                                               reduce_lr_patience=3)
        
        history_fine = fine_tune_model(
            model,
            train_generator,
            validation_generator,
            unfreeze_layers=50,
            epochs=args.fine_tune_epochs,
            learning_rate=args.fine_tune_lr,
            callbacks=fine_tune_callbacks
        )
        
        print("\n" + "="*60)
        print("FINE-TUNING COMPLETED")
        print("="*60)
        
        # Evaluate fine-tuned model
        print("\nEvaluating fine-tuned model on test set...")
        final_metrics = evaluate_model(model, test_generator)
        
        print("\n" + "="*70)
        print("FINE-TUNED MODEL PERFORMANCE")
        print("="*70)
        print(f"Test Accuracy:  {final_metrics['test_accuracy']*100:.2f}% "
              f"(was {initial_metrics['test_accuracy']*100:.2f}%) | "
              f"Δ: {(final_metrics['test_accuracy']-initial_metrics['test_accuracy'])*100:+.2f}%")
        print(f"Test Precision: {final_metrics['test_precision']*100:.2f}% "
              f"(was {initial_metrics['test_precision']*100:.2f}%) | "
              f"Δ: {(final_metrics['test_precision']-initial_metrics['test_precision'])*100:+.2f}%")
        print(f"Test Recall:    {final_metrics['test_recall']*100:.2f}% "
              f"(was {initial_metrics['test_recall']*100:.2f}%) | "
              f"Δ: {(final_metrics['test_recall']-initial_metrics['test_recall'])*100:+.2f}%")
        print(f"Test F1-Score:  {final_metrics['test_f1_score']*100:.2f}% "
              f"(was {initial_metrics['test_f1_score']*100:.2f}%) | "
              f"Δ: {(final_metrics['test_f1_score']-initial_metrics['test_f1_score'])*100:+.2f}%")
        print("="*70 + "\n")
    else:
        final_metrics = initial_metrics
    
    # Save class names
    class_names_path = os.path.join(args.models_dir, 'class_names.json')
    save_class_names(class_names, class_names_path)
    
    # Save metadata
    training_params = {
        "initial_epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs if not args.skip_fine_tune else 0,
        "batch_size": args.batch_size,
        "initial_lr": args.learning_rate,
        "fine_tune_lr": args.fine_tune_lr,
        "unfrozen_layers": 50 if not args.skip_fine_tune else 0,
        "optimizer": "Adam",
        "validation_split": 0.2
    }
    
    metadata = create_model_metadata(model, class_names, training_params, final_metrics)
    metadata_path = os.path.join(args.models_dir, 'model_metadata.json')
    save_model_metadata(metadata, metadata_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Initial model saved to: {initial_model_path}")
    if not args.skip_fine_tune:
        print(f"Fine-tuned model saved to: {fine_tuned_model_path}")
    print(f"Class names saved to: {class_names_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("="*60)


if __name__ == '__main__':
    main()
