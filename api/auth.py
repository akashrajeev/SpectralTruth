"""
API Key Authentication Module

Contains helpers for both legacy X-API-Key auth and the new
Authorization: Bearer <API_KEY> scheme used by the v1 voice endpoint.
"""
import os
from fastapi import Header, HTTPException, status
from typing import Optional


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Verify API key from X-API-Key header.

    This is kept for backward compatibility with existing endpoints.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide X-API-Key header."
        )

    # For hackathon: accept any non-empty API key
    # In production, validate against a database or config
    if not x_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )

    return x_api_key


async def verify_bearer_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """
    Verify API key from Authorization: Bearer <API_KEY> header.

    This is used by the /api/v1/voice/detect endpoint to match the
    hackathon specification exactly.

    The expected API key is read from the DEEPFAKE_API_KEY environment
    variable, defaulting to "test123" for local/testing use.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected 'Authorization: Bearer <API_KEY>'.",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Authorization: Bearer <API_KEY>'.",
        )

    expected_key = os.getenv("DEEPFAKE_API_KEY", "test123")

    if not token.strip() or token != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return token
