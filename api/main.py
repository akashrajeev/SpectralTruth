"""
FastAPI Main Application
Deepfake Audio Detection API
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any

from .audio_processor import (
    process_audio,
    process_audio_file,
    prepare_model_input,
    AudioProcessingError,
    MAX_FILE_SIZE,
)
from .model_service import get_model_service
from .language_detector import detect_language_simple
from .response_formatter import format_detection_response, format_detection_response_v1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Audio Detection API",
    description="API for detecting AI-generated voice samples",
    version="1.0.0"
)


# Request/Response Models
class AudioDetectionRequest(BaseModel):
    """Request model for legacy audio detection endpoint"""

    audio: str = Field(
        ...,
        description="Base64-encoded MP3 audio file",
        example="UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
    )


class VoiceDetectionRequestV1(BaseModel):
    """Request model for /api/v1/voice/detect endpoint.

    Accepts flexible field names to work with different testers:
    - audio_url, audioUrl, Audio URL, etc.
    - audio_base64, audioBase64, Audio Base64 Format, etc.
    """

    audio_url: Optional[str] = Field(
        default=None,
        description="Publicly accessible URL pointing to an MP3 file.",
        examples=["https://example.com/sample.mp3"],
    )
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded MP3 audio content.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional metadata message; not used for classification.",
    )
    
    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data: Any) -> Dict[str, Any]:
        """Normalize various field name variations to standard names."""
        if not isinstance(data, dict):
            return data
        
        result = {}
        
        # Standard field names take precedence
        if 'audio_url' in data:
            result['audio_url'] = data['audio_url']
        if 'audio_base64' in data:
            result['audio_base64'] = data['audio_base64']
        if 'message' in data:
            result['message'] = data['message']
        
        # List of field name variations
        url_field_variations = [
            'audioUrl', 'Audio URL', 'AUDIO_URL', 
            'audio-url', 'AudioUrl', 'audioUrl'
        ]
        base64_field_variations = [
            'audioBase64', 'Audio Base64 Format', 'AUDIO_BASE64',
            'audio-base64', 'audioBase64Format', 'AudioBase64Format',
            'audio_base64_format', 'Audio Base64'
        ]
        
        # Process all fields in the input data (skip if already set from standard fields)
        for key, value in data.items():
            if not value or not isinstance(value, str) or not value.strip():
                continue
            
            value = value.strip()
            key_lower = key.lower().replace(' ', '_').replace('-', '_')
            
            # Check if value looks like a URL (starts with http:// or https://)
            is_url = value.startswith(('http://', 'https://'))
            
            # If it's a URL, always treat as audio_url (regardless of field name)
            if is_url and 'audio_url' not in result:
                result['audio_url'] = value
                continue
            
            # Check if key matches URL field variations
            if (key in url_field_variations or 
                key_lower in ['audio_url', 'audiourl', 'audio url']):
                if 'audio_url' not in result:
                    result['audio_url'] = value
                continue
            
            # Check if key matches base64 field variations
            if (key in base64_field_variations or 
                key_lower in ['audio_base64', 'audiobase64', 'audio base64', 
                             'audio_base64_format', 'audiobase64format']):
                # If it's actually a URL (common mistake), treat as audio_url
                if is_url:
                    if 'audio_url' not in result:
                        result['audio_url'] = value
                else:
                    if 'audio_base64' not in result:
                        result['audio_base64'] = value
                continue
        
        # Always include message if it exists
        if 'message' in data:
            result['message'] = data['message']
        
        return result


class AudioDetectionResponse(BaseModel):
    """Response model for audio detection"""
    classification: str = Field(..., description="Classification: 'AI' or 'Human'")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", ge=0.0, le=1.0)
    explanation: str = Field(..., description="Explanation of the classification")
    language: str = Field(..., description="Detected language")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


# Initialize model service at startup
@app.on_event("startup")
async def startup_event():
    """Initialize model service on application startup"""
    try:
        # Get model path from environment or use default
        model_path = os.getenv("MODEL_PATH")
        if model_path:
            model_path = Path(model_path)
        else:
            # Default to model/model-1.h5 relative to project root
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "model" / "model-1.h5"
        
        logger.info(f"Initializing model service with path: {model_path}")
        get_model_service(model_path=str(model_path))
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "Deepfake Audio Detection API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Verify model is loaded
        model_service = get_model_service()
        if model_service.model is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "error",
                    "message": "Model not loaded"
                }
            )
        
        return {
            "status": "ok",
            "message": "Service is healthy"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": f"Service unhealthy: {str(e)}"
            }
        )


@app.post("/detect", response_model=AudioDetectionResponse)
async def detect_audio(
    request: AudioDetectionRequest,
):
    """
    Detect if audio is AI-generated or human-generated.
    
    Args:
        request: AudioDetectionRequest with Base64-encoded MP3 audio
        
    Returns:
        AudioDetectionResponse with classification, confidence, explanation, and language
    """
    try:
        logger.info("Received audio detection request")
        
        # Process audio: decode Base64, load, generate mel-spectrogram
        mel_spectrogram = process_audio(request.audio)
        
        # Prepare model input
        model_input = prepare_model_input(mel_spectrogram)
        
        # Get model service and run prediction
        model_service = get_model_service()
        ai_prob, human_prob = model_service.predict(model_input)
        
        # Classify
        classification, confidence = model_service.classify(model_input)
        
        # Detect language (simplified for now)
        language = detect_language_simple()
        
        # Format response
        response = format_detection_response(
            classification=classification,
            confidence=confidence,
            language=language,
            ai_probability=ai_prob,
            human_probability=human_prob
        )
        
        logger.info(
            f"Detection complete: {classification} "
            f"(confidence: {confidence:.3f}, language: {language})"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (from audio processing, etc.)
        raise
    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


async def _process_detection(audio_bytes: bytes, is_flac: bool = False) -> AudioDetectionResponse:
    """
    Helper function to process audio and return detection response.
    
    Args:
        audio_bytes: Audio file bytes
        is_flac: Whether the audio is FLAC (True) or MP3 (False)
        
    Returns:
        AudioDetectionResponse
    """
    # Process audio file
    mel_spectrogram = process_audio_file(audio_bytes, is_flac=is_flac)
    
    # Prepare model input
    model_input = prepare_model_input(mel_spectrogram)
    
    # Get model service and run prediction
    model_service = get_model_service()
    ai_prob, human_prob = model_service.predict(model_input)
    
    # Classify
    classification, confidence = model_service.classify(model_input)
    
    # Detect language (simplified for now)
    language = detect_language_simple()
    
    # Format response
    response = format_detection_response(
        classification=classification,
        confidence=confidence,
        language=language,
        ai_probability=ai_prob,
        human_probability=human_prob
    )
    
    logger.info(
        f"Detection complete: {classification} "
        f"(confidence: {confidence:.3f}, language: {language})"
    )
    
    return response


def extract_audio_from_payload(payload: dict) -> Tuple[str, str]:
    """
    Auto-detect audio data from request payload.
    
    Strategy:
    1. Check preferred/known keys first (audio_url, audio_base64)
    2. If not found, scan ALL fields in the payload
    3. Detect URLs (http/https with audio extensions)
    4. Detect base64 strings (long strings matching base64 pattern)
    
    Returns:
        Tuple of (audio_type, audio_data) where audio_type is "url" or "base64"
    
    Raises:
        ValueError: If no audio data is found in the payload
    """
    # 1. Preferred keys (official/known field names)
    if payload.get("audio_url"):
        value = payload["audio_url"]
        if value and isinstance(value, str) and value.strip():
            return "url", value.strip()
    
    if payload.get("audio_base64"):
        value = payload["audio_base64"]
        if value and isinstance(value, str) and value.strip():
            # Check if it's actually a URL (common mistake)
            if value.strip().startswith(("http://", "https://")):
                return "url", value.strip()
            return "base64", value.strip()
    
    # 2. Fallback: scan EVERYTHING in the payload
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            continue
        
        value = value.strip()
        
        # URL detection: accept ANY URL (http/https)
        # Prioritize URLs with audio extensions, but accept any URL
        if value.startswith(("http://", "https://")):
            # Check for audio file extensions first (preferred)
            audio_extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]
            value_lower = value.lower()
            if any(value_lower.endswith(ext) for ext in audio_extensions):
                return "url", value
            # Accept any URL (might redirect to audio, or extension might be missing)
            # This makes the API UI-proof - accepts URLs from any field
            return "url", value
        
        # Base64 detection: long string matching base64 pattern
        # Base64 strings are typically > 500 chars and contain only base64 characters
        if len(value) > 500:
            # Base64 pattern: A-Z, a-z, 0-9, +, /, =, and whitespace (which we strip)
            if re.match(r"^[A-Za-z0-9+/=]+$", value):
                return "base64", value
    
    # 3. Nothing found
    raise ValueError("No audio data found in request payload. Please provide audio_url (URL) or audio_base64 (base64 string), or any field containing audio data.")


async def _fetch_mp3_from_url(audio_url: str) -> bytes:
    """
    Download MP3 audio bytes from a public URL.

    Enforces a maximum file size consistent with audio processing
    limits and provides clear 4xx errors for invalid inputs.
    """
    if not audio_url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="audio_url must start with http:// or https://",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(audio_url)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download audio from URL: {exc}",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download audio from URL. HTTP status: {response.status_code}",
        )

    content = response.content or b""

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Downloaded audio file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.1f}MB",
        )

    return content


def _map_classification_to_spec(
    internal_classification: str,
    ai_prob: float,
    human_prob: float,
    uncertain_threshold: float = 0.6,
) -> tuple[str, float]:
    """
    Map internal \"AI\"/\"Human\" classification and probabilities to
    the required spec labels: AI_GENERATED, HUMAN, UNCERTAIN.
    """
    max_prob = max(ai_prob, human_prob)

    if max_prob < uncertain_threshold:
        return "UNCERTAIN", max_prob

    if internal_classification == "AI":
        return "AI_GENERATED", ai_prob

    return "HUMAN", human_prob


@app.post("/detect/mp3", response_model=AudioDetectionResponse)
async def detect_audio_mp3_upload(
    file: UploadFile = File(..., description="MP3 audio file"),
):
    """
    Detect if audio is AI-generated or human-generated from MP3 file upload.
    
    This endpoint accepts direct MP3 file uploads for easier debugging and testing.
    
    Args:
        file: MP3 audio file (multipart/form-data)
        
    Returns:
        AudioDetectionResponse with classification, confidence, explanation, and language
    """
    try:
        logger.info(f"Received MP3 file upload: {file.filename}, content_type: {file.content_type}")
        
        # Read file bytes
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        logger.info(f"Processing MP3 file: {len(audio_bytes)} bytes")
        
        # Process as MP3 (will be converted to FLAC internally)
        return await _process_detection(audio_bytes, is_flac=False)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing MP3 upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process MP3 file: {str(e)}"
        )


@app.post("/detect/flac", response_model=AudioDetectionResponse)
async def detect_audio_flac_upload(
    file: UploadFile = File(..., description="FLAC audio file"),
):
    """
    Detect if audio is AI-generated or human-generated from FLAC file upload.
    
    This endpoint accepts direct FLAC file uploads for easier debugging and testing.
    FLAC files are processed directly without conversion.
    
    Args:
        file: FLAC audio file (multipart/form-data)
        
    Returns:
        AudioDetectionResponse with classification, confidence, explanation, and language
    """
    try:
        logger.info(f"Received FLAC file upload: {file.filename}, content_type: {file.content_type}")
        
        # Read file bytes
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        logger.info(f"Processing FLAC file: {len(audio_bytes)} bytes")
        
        # Process as FLAC (no conversion needed)
        return await _process_detection(audio_bytes, is_flac=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing FLAC upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process FLAC file: {str(e)}"
        )


@app.post("/api/v1/voice/detect")
async def detect_voice_v1(
    request: Request,
):
    """
    AI-generated voice detection endpoint (multi-language, v1 spec).

    This endpoint:
    - Accepts MP3 audio via audio_url and/or audio_base64
    - Auto-detects audio data from ANY field name in the request
    - Returns JSON with classification, confidence, explanation, and language
    
    Audio Auto-Detection:
    - First checks known fields: audio_url, audio_base64
    - Then scans ALL fields in the request body
    - Detects URLs (http/https with audio extensions)
    - Detects base64 strings (long strings matching base64 pattern)
    - Makes the API UI-proof and tester-agnostic
    """
    try:
        logger.info("Received v1 voice detection request")
        
        # Get raw request body as dict
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be a JSON object",
            )
        
        # Auto-detect audio data from payload
        try:
            audio_type, audio_data = extract_audio_from_payload(payload)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        
        audio_bytes: Optional[bytes] = None

        if audio_type == "url":
            logger.info(f"Downloading audio from URL: {audio_data}")
            audio_bytes = await _fetch_mp3_from_url(audio_data)
            mel_spectrogram = process_audio_file(audio_bytes, is_flac=False)
        elif audio_type == "base64":
            logger.info("Processing Base64-encoded audio")
            mel_spectrogram = process_audio(audio_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unexpected audio type: {audio_type}",
            )

        model_input = prepare_model_input(mel_spectrogram)

        model_service = get_model_service()
        ai_prob, human_prob = model_service.predict(model_input)
        internal_classification, _ = model_service.classify(model_input)

        external_classification, confidence = _map_classification_to_spec(
            internal_classification,
            ai_prob,
            human_prob,
        )

        language = detect_language_simple()

        response = format_detection_response_v1(
            external_classification=external_classification,
            internal_classification=internal_classification,
            confidence=confidence,
            language=language,
            ai_probability=ai_prob,
            human_probability=human_prob,
        )

        logger.info(
            "V1 detection complete: %s (confidence: %.3f, language: %s)",
            response["classification"],
            response["confidence"],
            response["language"],
        )

        return response

    except HTTPException:
        raise
    except AudioProcessingError as e:
        logger.error(f"Audio processing error (v1): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing failed: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during v1 detection: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
