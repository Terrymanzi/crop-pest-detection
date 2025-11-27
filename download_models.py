"""
Download model files from Google Drive or other cloud storage.
This script runs before the application starts on Render.
"""
import os
import sys
import gdown
from pathlib import Path

def download_from_google_drive(file_id, output_path):
    """Download file from Google Drive using gdown."""
    try:
        print(f"Downloading to {output_path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✓ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {output_path}: {e}")
        return False

def download_models():
    """Download all required model files."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Get file IDs from environment variables
    model_file_id = os.getenv("MODEL_FILE_ID")
    class_names_file_id = os.getenv("CLASS_NAMES_FILE_ID")
    
    success = True
    
    # Download main model
    if model_file_id:
        model_path = models_dir / "crop_pest_model_finetuned.h5"
        if not model_path.exists():
            print("Downloading model file...")
            if not download_from_google_drive(model_file_id, str(model_path)):
                # Try fallback model
                print("Trying fallback model...")
                fallback_id = os.getenv("FALLBACK_MODEL_FILE_ID")
                if fallback_id:
                    fallback_path = models_dir / "crop_pest_model.h5"
                    success = download_from_google_drive(fallback_id, str(fallback_path))
        else:
            print(f"✓ Model already exists: {model_path}")
    else:
        print("⚠ MODEL_FILE_ID not set in environment variables")
        success = False
    
    # Download class names
    if class_names_file_id:
        class_names_path = models_dir / "class_names.json"
        if not class_names_path.exists():
            print("Downloading class names...")
            download_from_google_drive(class_names_file_id, str(class_names_path))
        else:
            print(f"✓ Class names already exists: {class_names_path}")
    else:
        print("⚠ CLASS_NAMES_FILE_ID not set, will try to use existing file")
    
    return success

if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING MODEL FILES")
    print("=" * 60)
    
    if download_models():
        print("\n✓ Model download complete!")
        sys.exit(0)
    else:
        print("\n⚠ Model download had issues, but continuing...")
        # Don't fail the build, let the app handle missing models gracefully
        sys.exit(0)
