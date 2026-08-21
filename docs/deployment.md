# Voice-RAG Production Deployment Guide

This document outlines the requirements and procedures for deploying the Voice-Enabled RAG application in a production environment.

## 1. System Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended)
- **CPU**: 4+ cores (for local embedding model processing)
- **RAM**: 8GB+ (16GB recommended for large Qdrant indices)
- **Disk**: 50GB+ SSD (for dataset and Vector DB storage)
- **Dependencies**: Docker and Docker Compose v2 (recommended) or Python 3.10+ and Node.js 18+ (manual deployment)

## 2. Environment Configuration

Create a `.env` file in the root directory. This file must NOT be checked into version control.

```bash
# Core Mode
DEMO_MODE=False

# API Keys (Do not expose!)
GROQ_API_KEY="gsk_..."
SARVAM_API_KEY="sk_..."

# Providers Configuration
LLM_PROVIDER="groq"
STT_PROVIDER="sarvam"

# Guardrails Configuration
OFFTOPIC_THRESHOLD=0.3
```

## 3. Deployment using Docker (Recommended)

Docker Compose provides the most reliable way to deploy the application, ensuring Qdrant, the Backend, and the Frontend are properly orchestrated.

```bash
# 1. Build the containers
docker compose build

# 2. Start the services in detached mode
docker compose up -d

# 3. Verify services are running
docker compose ps
docker compose logs -f
```

## 4. Manual Deployment

If Docker is not available, you can deploy the components manually.

### Qdrant Vector Database
```bash
# Run Qdrant via standalone binary or Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Backend (FastAPI)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Uvicorn in production mode (adjust workers based on CPU cores)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend (Next.js)
```bash
cd frontend
npm ci

# Build the production bundle
npm run build

# Start the production server
npm start
```

## 5. Security Validation

Before exposing the application to users, run the security validation suite to ensure guardrails are active and credentials are not leaked:

```bash
export PYTHONPATH=. 
source .venv/bin/activate
python scripts/security_validation.py
```
*Note: A PASS on all tests confirms that rate limiting, prompt injection defenses, and input validations are functioning correctly.*

## 6. Known Limitations in Current Environment
- Due to strict local environment limits, Playwright browser automation tests are marked as `PARTIAL` because `sudo` access is unavailable to install necessary browser dependencies. Manual browser tests have been fully verified.
- The `docker compose` command may fail if the Docker daemon is not running on the host system.
