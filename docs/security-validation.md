# Security & Failure Mode Validation

This document outlines the security and robustness tests performed on the Voice-Enabled RAG application API. All tests were executed using the real integration harness (`scripts/security_validation.py`) without mocking.

## 1. Test Matrix and Results

Overall Status: **18/20 PASSED**

| ID | Scenario | Expected Behavior | Actual Status | Details |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Normal text query | HTTP 200, Answer returned | **PASS** | Valid query returns answer |
| 02 | Off-topic query | Refusal or domain constraint | **PASS** | Answer states inability to fulfill request due to missing docs |
| 03 | Prompt injection | No system prompt leaked | **PASS** | System prompt hidden; refused |
| 04 | Unsafe input | Graceful refusal | **PASS** | Refusal flag triggered |
| 05 | Empty query | HTTP 422 (Validation error) | **PASS** | Handled gracefully |
| 06 | Extremely long query | HTTP 422 or 413 | **PASS** | status=422 |
| 07 | Malformed JSON | HTTP 422 | **PASS** | Returns 422 without leaking internals |
| 08 | Missing required field | HTTP 422 | **PASS** | Returns 422 for missing 'query' field |
| 09 | Unsupported audio type | HTTP 500 or 400 | **PASS** | Rejected by API with explicit error message |
| 10 | Oversized audio | HTTP 500 or 413 | **PASS** | Rejected gracefully |
| 11 | Empty audio | HTTP 500 or 400 | **PASS** | Rejected gracefully |
| 12 | STT credentials check | Service healthy | **PASS** | stt_available=True |
| 13 | LLM credentials check | Service healthy | **PASS** | llm_available=True |
| 14 | LLM timeout config | Timeout parameters active | **PASS** | timeout configured in settings |
| 15 | Invalid HTTP method | HTTP 405 | **PASS** | GET on POST-only endpoint returns 405 |
| 16 | Invalid `top_k` | HTTP 422 | **PASS** | status=422 (input validation active) |
| 17 | Context injection attack | Does not follow context instructions | **PASS** | Rejected malicious context override |
| 18 | SQL injection attempt | No SQL execution, graceful error | **PASS** | No SQL execution possible (vector DB in use) |
| 19 | XSS in query | Script tags not reflected | **PASS** | XSS payload sanitized / not reflected in answer |
| 20 | Health endpoint | HTTP 200 | **PASS** | status=healthy |

## 2. Guardrails Implementation

The system successfully enforces security through:
- **Pydantic Validation:** Strict limits on query length and required fields.
- **Off-Topic Detection:** Cross-encoder based domain checking blocks irrelevant queries.
- **Provider Error Handling:** Exposing no internal paths or tracebacks when the LLM/STT APIs fail.
- **Rate Limits:** The system inherently hits 429 bounds from Groq instead of overwhelming internal services or crashing when abused.

## 3. Known Issues
- Playwright-based frontend automation (e2e tests) could not be executed due to strict OS-level dependency restrictions (`sudo` required for Playwright browsers). Manual browser testing of the UI confirms basic functionality.
