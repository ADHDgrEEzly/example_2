import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from config import MODELS_DIR
from .logger import setup_logger

logger = setup_logger(__name__)

class ModelSerializer:
    """Handles model serialization and metadata management."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the model serializer.
        
        Args:
            model_dir: Directory to store models. If None, uses default from config
        """
        self.model_dir = model_dir or MODELS_DIR
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save model and its metadata.
        
        Args:
            model: The trained model object
            model_name: Base name for the model files
            metadata: Optional dictionary with model metadata
            
        Returns:
            Path: Path to the saved model file
        """
        # Create timestamp-based version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{model_name}_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = model_path / "model.joblib"
        joblib.dump(model, model_file)
        
        # Save metadata
        if metadata:
            metadata.update({
                "timestamp": timestamp,
                "model_name": model_name
            })
            metadata_file = model_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_file
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> tuple[Any, Optional[Dict]]:
        """
        Load model and its metadata.
        
        Args:
            model_name: Base name of the model
            version: Specific version (timestamp) to load. If None, loads latest
            
        Returns:
            tuple: (model object, metadata dictionary)
        """
        if version:
            model_path = self.model_dir / f"{model_name}_{version}"
        else:
            # Find latest version
            versions = list(self.model_dir.glob(f"{model_name}_*"))
            if not versions:
                raise FileNotFoundError(f"No models found for {model_name}")
            model_path = sorted(versions)[-1]
        
        # Load model
        model_file = model_path / "model.joblib"
        model = joblib.load(model_file)
        
        # Load metadata if exists
        metadata = None
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model, metadata
    
    def list_versions(self, model_name: str) -> list[str]:
        """
        List all available versions for a model.
        
        Args:
            model_name: Base name of the model
            
        Returns:
            list: List of version timestamps
        """
        versions = [
            p.name.replace(f"{model_name}_", "")
            for p in self.model_dir.glob(f"{model_name}_*")
        ]
        return sorted(versions)