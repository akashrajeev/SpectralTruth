"""
FastAPI Main Application
Deepfake Audio Detection API
"""
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import verify_api_key, verify_bearer_token
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

    Supports both audio_url and audio_base64, plus an optional
    message field used as metadata only.
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
    api_key: str = Depends(verify_api_key),
):
    """
    Detect if audio is AI-generated or human-generated.
    
    Args:
        request: AudioDetectionRequest with Base64-encoded MP3 audio
        api_key: API key from X-API-Key header (validated by dependency)
        
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
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if audio is AI-generated or human-generated from MP3 file upload.
    
    This endpoint accepts direct MP3 file uploads for easier debugging and testing.
    
    Args:
        file: MP3 audio file (multipart/form-data)
        api_key: API key from X-API-Key header (validated by dependency)
        
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
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if audio is AI-generated or human-generated from FLAC file upload.
    
    This endpoint accepts direct FLAC file uploads for easier debugging and testing.
    FLAC files are processed directly without conversion.
    
    Args:
        file: FLAC audio file (multipart/form-data)
        api_key: API key from X-API-Key header (validated by dependency)
        
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
    request: VoiceDetectionRequestV1,
    api_key: str = Depends(verify_bearer_token),
):
    """
    AI-generated voice detection endpoint (multi-language, v1 spec).

    This endpoint:
    - Accepts MP3 audio via audio_url and/or audio_base64
    - Requires Authorization: Bearer <API_KEY>
    - Returns JSON with classification, confidence, explanation, and language
    """
    try:
        logger.info("Received v1 voice detection request")

        audio_bytes: Optional[bytes] = None

        if request.audio_url:
            logger.info(f"Downloading audio from URL: {request.audio_url}")
            audio_bytes = await _fetch_mp3_from_url(request.audio_url)
            mel_spectrogram = process_audio_file(audio_bytes, is_flac=False)
        elif request.audio_base64:
            logger.info("Processing Base64-encoded audio")
            mel_spectrogram = process_audio(request.audio_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'audio_url' or 'audio_base64' must be provided.",
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
