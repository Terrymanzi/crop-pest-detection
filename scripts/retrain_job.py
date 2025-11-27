"""
Retraining script for crop pest detection model.

This script fine-tunes an existing model on new data.
Usage: python scripts/retrain_job.py --new_data_dir path/to/new/data
"""

import os
import sys
import argparse
import shutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.model import (
    load_model,
    fine_tune_model,
    save_model,
    load_class_names,
    save_model_metadata,
    create_model_metadata,
    evaluate_model,
    create_callbacks
)


def main():
    """Main retraining function."""
    parser = argparse.ArgumentParser(description='Retrain crop pest detection model')
    parser.add_argument('--new_data_dir', type=str, required=True,
                       help='Path to new training data directory')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Path to test data directory')
    parser.add_argument('--model_path', type=str, 
                       default='models/crop_pest_model_finetuned.h5',
                       help='Path to existing model to retrain')
    parser.add_argument('--class_names_path', type=str,
                       default='models/class_names.json',
                       help='Path to class names file')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save retrained model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of retraining epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for retraining')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Retraining learning rate')
    parser.add_argument('--unfreeze_layers', type=int, default=50,
                       help='Number of layers to unfreeze')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (width and height)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CROP PEST DETECTION MODEL RETRAINING")
    print("="*60)
    print(f"Existing model: {args.model_path}")
    print(f"New data directory: {args.new_data_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Retraining epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Unfreeze layers: {args.unfreeze_layers}")
    print("="*60 + "\n")
    
    # Load existing model
    print("Loading existing model...")
    model = load_model(args.model_path)
    print(f"✓ Model loaded from: {args.model_path}\n")
    
    # Load class names
    print("Loading class names...")
    class_names = load_class_names(args.class_names_path)
    num_classes = len(class_names)
    print(f"✓ Class names loaded: {class_names}\n")
    
    # Initialize preprocessor
    img_size = (args.img_size, args.img_size)
    preprocessor = DataPreprocessor(img_size=img_size, batch_size=args.batch_size)
    
    # Load new training data
    print("Loading new training data...")
    train_generator, validation_generator = preprocessor.load_train_data(args.new_data_dir)
    
    # Verify class consistency
    new_class_names = list(train_generator.class_indices.keys())
    if set(new_class_names) != set(class_names):
        print("⚠ WARNING: New data has different classes!")
        print(f"  Original classes: {class_names}")
        print(f"  New classes: {new_class_names}")
        print("  Proceeding with caution...\n")
    
    print(f"\n✓ New data loaded successfully!")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {validation_generator.samples}\n")
    
    # Load test data if available
    test_generator = None
    if os.path.exists(args.test_dir):
        print("Loading test data...")
        test_generator = preprocessor.load_test_data(args.test_dir)
        print(f"✓ Test samples: {test_generator.samples}\n")
    
    # Evaluate model before retraining
    if test_generator:
        print("Evaluating model before retraining...")
        initial_metrics = evaluate_model(model, test_generator)
        
        print("\n" + "="*60)
        print("PERFORMANCE BEFORE RETRAINING")
        print("="*60)
        print(f"Test Accuracy:  {initial_metrics['test_accuracy']*100:.2f}%")
        print(f"Test Precision: {initial_metrics['test_precision']*100:.2f}%")
        print(f"Test Recall:    {initial_metrics['test_recall']*100:.2f}%")
        print(f"Test F1-Score:  {initial_metrics['test_f1_score']*100:.2f}%")
        print("="*60 + "\n")
    
    # Create callbacks for retraining
    retrained_model_path = os.path.join(
        args.models_dir, 
        f'crop_pest_model_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    )
    callbacks = create_callbacks(retrained_model_path,
                                early_stopping_patience=7,
                                reduce_lr_patience=3)
    
    # Retrain (fine-tune) model
    print("="*60)
    print("STARTING RETRAINING")
    print("="*60)
    print(f"Unfreezing last {args.unfreeze_layers} layers")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60 + "\n")
    
    history = fine_tune_model(
        model,
        train_generator,
        validation_generator,
        unfreeze_layers=args.unfreeze_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        callbacks=callbacks
    )
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETED")
    print("="*60)
    
    # Evaluate retrained model
    if test_generator:
        print("\nEvaluating retrained model on test set...")
        final_metrics = evaluate_model(model, test_generator)
        
        print("\n" + "="*70)
        print("PERFORMANCE AFTER RETRAINING")
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
        final_metrics = {}
    
    # Save updated metadata
    training_params = {
        "retrain_date": datetime.now().isoformat(),
        "retrain_epochs": args.epochs,
        "retrain_lr": args.learning_rate,
        "unfrozen_layers": args.unfreeze_layers,
        "new_data_dir": args.new_data_dir,
        "new_training_samples": train_generator.samples,
        "optimizer": "Adam"
    }
    
    metadata = create_model_metadata(model, class_names, training_params, final_metrics)
    metadata_path = os.path.join(
        args.models_dir,
        f'model_metadata_retrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    save_model_metadata(metadata, metadata_path)
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    print(f"Retrained model saved to: {retrained_model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nTo use the retrained model, update your API configuration")
    print("or copy it to 'models/crop_pest_model_finetuned.h5'")
    print("="*60)


if __name__ == '__main__':
    main()
