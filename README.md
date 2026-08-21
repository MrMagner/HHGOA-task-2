# Voice-Enabled RAG System

A voice-enabled Retrieval-Augmented Generation system built for the HH Goa 2026 Shortlisting Task 2. Supports Hindi voice queries over the MSMARCO-XI dataset with cross-lingual retrieval.

## Features

- **End-to-end Voice Pipeline**: Audio → Sarvam STT → Query Processing → Hybrid Retrieval → Groq LLM → Grounded Answer
- **Hybrid Search**: Dense (multilingual Sentence-Transformers) + Lexical (BM25) via Reciprocal Rank Fusion
- **Multilingual Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual English↔Hindi retrieval
- **Guardrails**: Input safety, prompt injection detection, off-topic detection, grounding validation
- **Latency Benchmarking**: 100-query benchmark with P50/P70/P90/P95/P100 percentiles
- **Premium UI**: Next.js App Router with voice recording, real-time results, and source display

## Architecture

See [docs/architecture.md](docs/architecture.md) for details.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Required for real providers:**
- `SARVAM_API_KEY` — Sarvam AI speech-to-text API key
- `GROQ_API_KEY` — Groq LLM API key
- `STT_PROVIDER=sarvam`
- `LLM_PROVIDER=groq`
- `DEMO_MODE=False`

**Demo mode (no API keys needed):**
- `DEMO_MODE=True`

## Quickstart (Local Dev)

### 1. Setup & Ingestion

```bash
cd "/home/magner/projects/HHGOA task-2"
python -m venv .venv
source .venv/bin/activate.fish
pip install -r backend/requirements.txt
export PYTHONPATH=.

# Verify environment and dataset
cp .env.example .env
python scripts/verify_config.py
python run_ingestion_local.py
```

### 2. Start Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Start Frontend (In a new terminal)

```bash
cd "/home/magner/projects/HHGOA task-2/frontend"
npm install
npm run build
npm start
```

Open http://localhost:3000 in your browser.

### 5. Docker (Alternative)

```bash
cp .env.example .env
# Edit .env
docker-compose up --build
```

## Testing & Benchmarks

```bash
# Run unit tests and guardrail validations
pytest backend/tests/ -v

# Run the 100-query benchmark
python scripts/run_integration_benchmark.py
```

Results are saved to `results/latency_benchmark.json`.

## Known Limitations

- Full voice RAG latency is ~2,800 ms (warm) due to STT and LLM network calls. The <200 ms metric applies only to local retrieval + guardrails.
- Cold start includes embedding model download (~7 s first request). Subsequent requests are ~60 ms for embedding.
- BM25 lexical matching is limited for cross-lingual queries (English query vs Hindi corpus). Dense retrieval handles cross-lingual matching.
- Docker execution has not been tested in this environment (Docker unavailable).

## Documentation

- [Acceptance Report](docs/acceptance-report.md) — Measured latency values and audit matrix
- [Final Submission Checklist](docs/final-submission-checklist.md) — Requirement status with evidence
- [Architecture](docs/architecture.md) — System design
