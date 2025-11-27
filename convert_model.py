"""
Convert legacy model to TensorFlow 2.13+ compatible format.
Run this script locally before deploying.
"""
import os
import json
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

def convert_model(old_model_path, new_model_path):
    """
    Convert old model to new compatible format.
    
    Args:
        old_model_path: Path to legacy .h5 model
        new_model_path: Path to save converted model
    """
    print(f"Converting model: {old_model_path}")
    print("=" * 60)
    
    try:
        # Load the old model
        print("Loading legacy model...")
        model = keras.models.load_model(old_model_path, compile=False)
        print("✓ Model loaded")
        
        # Get model summary
        print("\nModel Summary:")
        model.summary()
        
        # Recompile with current TensorFlow
        print("\nRecompiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model recompiled")
        
        # Save in new format
        print(f"\nSaving converted model to: {new_model_path}")
        model.save(new_model_path, save_format='h5')
        print("✓ Model saved successfully")
        
        # Verify the new model loads
        print("\nVerifying converted model...")
        test_model = keras.models.load_model(new_model_path)
        print("✓ Converted model loads successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    models_dir = "models"
    
    # Model files to convert
    models_to_convert = [
        ("crop_pest_model.h5", "crop_pest_model_converted.h5"),
        ("crop_pest_model_finetuned.h5", "crop_pest_model_finetuned_converted.h5")
    ]
    
    for old_name, new_name in models_to_convert:
        old_path = os.path.join(models_dir, old_name)
        new_path = os.path.join(models_dir, new_name)
        
        if os.path.exists(old_path):
            print(f"\n{'=' * 60}")
            print(f"CONVERTING: {old_name}")
            print(f"{'=' * 60}")
            
            if convert_model(old_path, new_path):
                print(f"\n✓ SUCCESS: {new_name} created")
                
                # Create a backup of the old model
                backup_path = old_path.replace('.h5', '_backup.h5')
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(old_path, backup_path)
                    print(f"✓ Backup created: {os.path.basename(backup_path)}")
                
                # Replace old model with converted one
                import shutil
                shutil.move(new_path, old_path)
                print(f"✓ Replaced {old_name} with converted version")
            else:
                print(f"\n✗ FAILED to convert {old_name}")
        else:
            print(f"\n⚠ {old_name} not found, skipping...")
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the converted models locally")
    print("2. Commit and push the updated models")
    print("3. Redeploy on Railway")
