"""
Audio Processing Module
Handles Base64 decoding, MP3 processing, and mel-spectrogram generation
"""
import base64
import io
import logging
import numpy as np
import librosa
import tempfile
import os
from typing import Tuple, Optional
from fastapi import HTTPException, status
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Constants
N_MELS = 91
MAX_TIME_STEPS = 150
# Note: Model was trained on 5-second clips, but we process full audio for API flexibility
# The mel-spectrogram will be truncated/padded to MAX_TIME_STEPS regardless of duration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass


def decode_base64_audio(base64_string: str) -> bytes:
    """
    Decode Base64-encoded audio string to bytes.
    
    Args:
        base64_string: Base64-encoded audio string
        
    Returns:
        Decoded audio bytes
        
    Raises:
        HTTPException: If decoding fails
    """
    try:
        # Remove data URL prefix if present (e.g., "data:audio/mp3;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[-1]
        
        decoded = base64.b64decode(base64_string)
        
        if len(decoded) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio file too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        return decoded
    except Exception as e:
        logger.error(f"Base64 decoding error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Base64 encoding: {str(e)}"
        )


def convert_mp3_to_flac_bytes(mp3_bytes: bytes) -> bytes:
    """
    Convert MP3 bytes to FLAC bytes using pydub.
    
    Args:
        mp3_bytes: MP3 audio file bytes
        
    Returns:
        FLAC audio file bytes
        
    Raises:
        HTTPException: If conversion fails
    """
    try:
        # Load MP3 from bytes
        audio_io = io.BytesIO(mp3_bytes)
        audio_segment = AudioSegment.from_mp3(audio_io)
        
        # Export to FLAC in memory
        flac_io = io.BytesIO()
        audio_segment.export(flac_io, format="flac")
        flac_bytes = flac_io.getvalue()
        
        logger.info(f"Converted MP3 to FLAC: {len(mp3_bytes)} bytes -> {len(flac_bytes)} bytes")
        return flac_bytes
        
    except Exception as e:
        logger.error(f"MP3 to FLAC conversion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to convert MP3 to FLAC: {str(e)}. Ensure the input is a valid MP3 file."
        )


def load_audio_from_bytes(audio_bytes: bytes, is_flac: bool = False) -> np.ndarray:
    """
    Load audio from bytes using librosa.
    
    For FLAC files, uses temporary file to match command-line tool behavior exactly.
    For MP3 files, converts to FLAC in memory first.
    
    Args:
        audio_bytes: Audio file bytes (MP3, FLAC, etc.)
        is_flac: Whether the bytes are already in FLAC format
        
    Returns:
        Audio array (numpy array)
        
    Raises:
        HTTPException: If audio loading fails
    """
    temp_file_path = None
    try:
        if is_flac:
            # For FLAC: Use temporary file to match command-line tool exactly
            # This ensures identical processing pipeline (loading from file path)
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.flac', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            logger.info(f"Saved FLAC to temporary file: {temp_file_path}")
            
            # Load from file path (exactly like command-line tool)
            # Original command line uses librosa.load(clip) which defaults to sr=22050 (resamples)
            audio, sr = librosa.load(temp_file_path)  # Default sr=22050, matches command line
            logger.info(f"Loaded audio from file path: length={len(audio)}, sample_rate={sr}")
            
        else:
            # For MP3: Convert to FLAC in memory first
            audio_bytes = convert_mp3_to_flac_bytes(audio_bytes)
            
            # Create BytesIO object for librosa
            audio_io = io.BytesIO(audio_bytes)
            
            # Load FLAC audio with librosa (matching original command-line tool behavior)
            # Original command line uses librosa.load(clip) which defaults to sr=22050 (resamples)
            audio, sr = librosa.load(audio_io)  # Default sr=22050, matches command line
            logger.info(f"Loaded audio from BytesIO: length={len(audio)}, sample_rate={sr}")
        
        if len(audio) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty or could not be loaded"
            )
        
        return audio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio loading error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load audio file: {str(e)}. Ensure the file is a valid audio format."
        )
    finally:
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")


def generate_mel_spectrogram(audio: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """
    Generate mel-spectrogram from audio array.
    Matches command-line tool processing exactly.
    
    Args:
        audio: Audio array
        n_mels: Number of mel bands (default: 91)
        
    Returns:
        Mel-spectrogram array with shape (n_mels, max_time_steps)
    """
    try:
        # Generate mel-spectrogram (exactly like command-line tool)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        logger.info(f"Mel-spectrogram shape before padding/truncation: {mel_spectrogram.shape}")
        
        # Ensure all spectrograms have the same width (time steps)
        # This matches command-line tool exactly
        max_time_steps = MAX_TIME_STEPS
        if mel_spectrogram.shape[1] < max_time_steps:
            # Pad with zeros (matching command-line tool)
            mel_spectrogram = np.pad(
                mel_spectrogram,
                ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])),
                mode='constant'
            )
        else:
            # Truncate to max_time_steps (matching command-line tool)
            mel_spectrogram = mel_spectrogram[:, :max_time_steps]
        
        logger.info(f"Mel-spectrogram shape after padding/truncation: {mel_spectrogram.shape}")
        return mel_spectrogram
        
    except Exception as e:
        logger.error(f"Mel-spectrogram generation error: {str(e)}")
        raise AudioProcessingError(f"Failed to generate mel-spectrogram: {str(e)}")


def process_audio_file(audio_bytes: bytes, is_flac: bool = False) -> np.ndarray:
    """
    Process audio from file bytes (MP3 or FLAC).
    
    Args:
        audio_bytes: Audio file bytes
        is_flac: Whether the bytes are FLAC format (True) or MP3 (False)
        
    Returns:
        Mel-spectrogram array ready for model input (shape: (n_mels, max_time_steps))
    """
    # Load audio (converts MP3 to FLAC if needed)
    audio = load_audio_from_bytes(audio_bytes, is_flac=is_flac)
    
    # Generate mel-spectrogram
    mel_spectrogram = generate_mel_spectrogram(audio)
    
    return mel_spectrogram


def process_audio(base64_string: str) -> np.ndarray:
    """
    Complete audio processing pipeline: decode Base64, convert MP3 to FLAC, load audio, generate mel-spectrogram.
    
    Args:
        base64_string: Base64-encoded MP3 audio string
        
    Returns:
        Mel-spectrogram array ready for model input (shape: (n_mels, max_time_steps))
    """
    # Decode Base64
    audio_bytes = decode_base64_audio(base64_string)
    
    # Convert MP3 to FLAC and load audio
    audio = load_audio_from_bytes(audio_bytes, is_flac=False)
    
    # Generate mel-spectrogram
    mel_spectrogram = generate_mel_spectrogram(audio)
    
    return mel_spectrogram


def prepare_model_input(mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Prepare mel-spectrogram for model input.
    Matches ORIGINAL command-line tool processing exactly (no channel dimension).
    
    Original command line does:
    1. x.append(mel_spectrogram)  # mel is (91, 150)
    2. x = np.array(x)  # x is (1, 91, 150)
    3. Pass directly to model (no channel dimension added)
    
    Args:
        mel_spectrogram: Mel-spectrogram array (n_mels, max_time_steps)
        
    Returns:
        Reshaped array for model input (1, n_mels, max_time_steps)
    """
    # Match ORIGINAL command-line tool EXACTLY:
    # Create array from list (like original: x = np.array(x))
    # Do NOT add channel dimension - original version didn't have it
    model_input = np.array([mel_spectrogram])  # (1, n_mels, max_time_steps)
    logger.info(f"Model input shape (matching original command line): {model_input.shape}")
    return model_input
