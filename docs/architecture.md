# System Architecture

## Component Breakdown

### 1. Ingestion Layer (`backend/ingestion/`)
- Downloads MSMARCO-XI via `datasets`.
- Extracts `text`, `query`, and `metadata`.
- **Chunking**: Uses metadata-aware and sentence-boundary chunking to preserve semantic meaning while keeping within context windows.

### 2. Retrieval Layer (`backend/services/retrieval/`)
- **Dense Retriever**: Qdrant Vector DB with `all-MiniLM-L6-v2` embeddings.
- **Lexical Retriever**: BM25 (TF-IDF based).
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF).

### 3. Generation Layer (`backend/services/generation/`)
- Abstracted LLM Providers (Groq, OpenAI, Demo).
- RAG Prompts enforcing grounding and citations.

### 4. Guardrails (`backend/services/guardrails/`)
- **Pre-Retrieval**: Injection and Off-topic detection.
- **Post-Retrieval**: Grounding validation (checks word overlap between output and retrieved context).

### 5. Frontend (`frontend/`)
- React/Next.js UI.
- Direct MediaRecorder usage to capture microphone data, shipped via `FormData` to the FastAPI backend.
