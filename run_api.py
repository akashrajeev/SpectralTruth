#!/usr/bin/env python3
"""
Entry point script to run the FastAPI server
"""
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")

if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting Deepfake Audio Detection API")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload only works with 1 worker
        reload=reload,
        log_level="info"
    )
