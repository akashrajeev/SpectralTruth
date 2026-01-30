# Docker Deployment Guide

This guide explains how to build and run the Deepfake Audio Detection API using Docker.

## Prerequisites

- Docker installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, for easier development)

## Quick Start

### Using Docker Compose (Recommended)

The easiest way to run the application:

```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Start the container
- Expose the API on port 8000

Access the API at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

To run in detached mode:
```bash
docker-compose up -d
```

To stop the container:
```bash
docker-compose down
```

### Using Docker Directly

#### Build the Image

```bash
docker build -t deepfake-audio-detector .
```

#### Run the Container

```bash
docker run -p 8000:8000 deepfake-audio-detector
```

#### Run with Custom Environment Variables

```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e WORKERS=2 \
  -e MODEL_PATH=/app/model/model-1.h5 \
  deepfake-audio-detector
```

## Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port on which the API server will listen |
| `HOST` | `0.0.0.0` | Host address to bind to |
| `WORKERS` | `1` | Number of uvicorn workers |
| `RELOAD` | `false` | Enable auto-reload for development |
| `MODEL_PATH` | `model/model-1.h5` | Path to the TensorFlow model file |

## Development with Volume Mounting

For development, you can mount your local code as volumes to enable live code changes:

Uncomment the volumes section in `docker-compose.yml`:

```yaml
volumes:
  - ./api:/app/api
  - ./model:/app/model
  - ./run_api.py:/app/run_api.py
```

Then run with:
```bash
docker-compose up
```

**Note:** When using volume mounts, set `RELOAD=true` to enable auto-reload.

## Health Check

The container includes a health check that verifies the API is responding. Check container health:

```bash
docker ps
```

The health status will be shown in the STATUS column.

## Viewing Logs

### Docker Compose
```bash
docker-compose logs -f
```

### Docker
```bash
docker logs -f <container_id>
```

## Building for Production

For production deployments, consider:

1. **Multi-stage builds** (if needed for optimization)
2. **Non-root user** (already implemented)
3. **Resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

## Troubleshooting

### Container fails to start

1. Check logs: `docker logs <container_id>`
2. Verify model file exists: `docker exec <container_id> ls -la /app/model/`
3. Check port availability: Ensure port 8000 is not in use

### Model not loading

1. Verify model file is included in the image:
   ```bash
   docker run --rm deepfake-audio-detector ls -la /app/model/
   ```
2. Check MODEL_PATH environment variable is correct

### Audio processing errors

1. Verify ffmpeg is installed: `docker exec <container_id> ffmpeg -version`
2. Check audio file format is supported (MP3, FLAC)

## Image Size Optimization

The current image includes:
- Python 3.10-slim base (~45MB)
- System dependencies (ffmpeg, curl)
- Python packages (TensorFlow, FastAPI, etc.)
- Application code and model

To reduce image size:
- Use multi-stage builds
- Remove unnecessary packages
- Use Alpine Linux base (may require additional setup)

## Security Considerations

- Container runs as non-root user (`appuser`)
- No sensitive data in image (use environment variables or secrets)
- Health check enabled for monitoring
- Regular security updates recommended

## Deployment to Cloud Platforms

### AWS ECS/Fargate
- Push image to ECR
- Use task definition with environment variables
- Configure health checks

### Google Cloud Run
- Push to Container Registry or Artifact Registry
- Deploy with: `gcloud run deploy`

### Azure Container Instances
- Push to Azure Container Registry
- Deploy using Azure CLI or portal

### Hugging Face Spaces
- Use the Dockerfile directly
- Ensure model is included or accessible
- See Hugging Face Spaces documentation for deployment

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
