"""
API Key Authentication Module

Contains helpers for X-API-Key and Authorization header authentication.
Both accept any non-empty value for hackathon/demo purposes.
"""
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
    Verify API key from Authorization header.

    This is used by the /api/v1/voice/detect endpoint.
    
    For hackathon: accepts any non-empty value in Authorization header.
    In production, validate against a database or config.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    # For hackathon: accept any non-empty value
    # In production, validate against a database or config
    if not authorization.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return authorization
