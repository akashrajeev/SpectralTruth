"""
Model Service Module
Handles TensorFlow model loading and prediction
"""
import os
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and running TensorFlow model predictions"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model service.
        
        Args:
            model_path: Path to the TensorFlow model file. 
                       Defaults to 'model/model-1.h5' relative to project root.
        """
        if model_path is None:
            # Default to model/model-1.h5 relative to this file
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "model" / "model-1.h5"
        
        self.model_path = Path(model_path)
        self.model: Optional[tf.keras.Model] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the TensorFlow model from disk"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}. "
                    f"Please ensure the model file exists."
                )
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict(self, mel_spectrogram: np.ndarray) -> Tuple[float, float]:
        """
        Run prediction on mel-spectrogram.
        
        Args:
            mel_spectrogram: Mel-spectrogram array with shape (n_mels, max_time_steps)
                           or already prepared model input (1, n_mels, max_time_steps, 1)
        
        Returns:
            Tuple of (ai_probability, human_probability)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Match ORIGINAL command-line tool behavior
            # Original passes (1, n_mels, max_time_steps) - no channel dimension
            if mel_spectrogram.ndim == 2:
                # Add batch dimension only (matching original command line)
                model_input = mel_spectrogram[np.newaxis, ...]  # (1, n_mels, max_time_steps)
            elif mel_spectrogram.ndim == 3:
                # Already has batch dimension
                model_input = mel_spectrogram
            else:
                raise ValueError(
                    f"Invalid input shape: {mel_spectrogram.shape}. "
                    f"Expected 2D (n_mels, max_time_steps) or 3D (1, n_mels, max_time_steps)"
                )
            
            # Run prediction
            logger.info(f"Model input shape: {model_input.shape}")
            predictions = self.model.predict(model_input, verbose=0)
            logger.info(f"Raw model predictions shape: {predictions.shape}")
            logger.info(f"Raw model predictions: {predictions}")
            
            # Model outputs: [AI_probability, Human_probability]
            ai_prob = float(predictions[0][0])
            human_prob = float(predictions[0][1])
            
            logger.info(f"Prediction: AI={ai_prob:.6f} ({ai_prob*100:.1f}%), Human={human_prob:.6f} ({human_prob*100:.1f}%)")
            
            return ai_prob, human_prob
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def classify(self, mel_spectrogram: np.ndarray) -> Tuple[str, float]:
        """
        Classify audio as AI or Human.
        
        Args:
            mel_spectrogram: Mel-spectrogram array
            
        Returns:
            Tuple of (classification, confidence)
            classification: "AI" or "Human"
            confidence: Confidence score (0.0 to 1.0)
        """
        ai_prob, human_prob = self.predict(mel_spectrogram)
        
        if ai_prob > human_prob:
            classification = "AI"
            confidence = ai_prob
        else:
            classification = "Human"
            confidence = human_prob
        
        return classification, confidence


# Global model service instance (loaded at startup)
_model_service: Optional[ModelService] = None


def get_model_service(model_path: Optional[str] = None) -> ModelService:
    """
    Get or create the global model service instance.
    
    Args:
        model_path: Optional path to model file (only used on first call)
        
    Returns:
        ModelService instance
    """
    global _model_service
    
    if _model_service is None:
        _model_service = ModelService(model_path)
    
    return _model_service
