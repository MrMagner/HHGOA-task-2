# Submission Checklist

## 1. Automated Testing
- [x] Backend tests passed (10/10 passing via `pytest backend/tests/`).
- [x] Guardrails configured properly (Grounding, Safety, Off-topic).
- [x] Frontend `npm run build` succeeds without errors.

## 2. Endpoints & Integrations
- [x] Frontend successfully routes text/voice requests to `/api/v1/query` and `/api/v1/query/voice`.
- [x] Qdrant locally contains the full 1,000 vectors from MSMARCO-XI.
- [x] BM25 indexing correctly processes the 1,000 real documents.
- [x] Sarvam STT integration works with valid key (verified ~584 ms latency).
- [x] Groq LLM (`groq/compound`) integration works and returns grounded answers (verified ~1804 ms latency).

## 3. Security & Validation
- [x] `.env` is properly ignored in `.gitignore` and no API keys are checked into source control.
- [x] No results or network latency metrics were fabricated. 
- [x] The `docs/acceptance-report.md` correctly distinguishes local benchmarking from live STT/LLM network latencies.
- [x] Acknowledged that full end-to-end voice latency (<200 ms target) is not achievable with standard REST inference boundaries.

## 4. Environment
- [x] The Docker compose tests were bypassed and marked BLOCKED because the environment lacks Docker, preventing fabrications.
- [x] The automated Playwright browser tests were marked PARTIAL because Microsoft Playwright CDN drivers failed to download.

The project is fully prepared for final submission.
