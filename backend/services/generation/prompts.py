"""RAG prompt templates for the Voice-RAG system.

Contains structured prompts that enforce grounding and
produce well-formatted, citation-backed answers.
"""

from __future__ import annotations

from backend.services.retrieval.base import RetrievalResult


def build_system_prompt() -> str:
    """Build the system prompt for RAG generation.

    Returns:
        System prompt string that instructs the LLM on behavior.
    """
    return """You are a precise and helpful information retrieval assistant. Your role is to answer questions based EXCLUSIVELY on the provided context documents.

## Rules:
1. **Grounding**: Only use information from the provided context. Do not use external knowledge.
2. **Citations**: Reference the source document numbers [1], [2], etc. when stating facts.
3. **Honesty**: If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question."
4. **Conciseness**: Provide clear, focused answers. Avoid unnecessary elaboration.
5. **Structure**: Use bullet points or numbered lists for multi-part answers.
6. **Language**: Respond in the same language as the question.

## Format:
- Start with a direct answer to the question.
- Support your answer with relevant details from the context.
- End with any caveats or limitations if applicable."""


def build_rag_prompt(
    query: str,
    context_documents: list[RetrievalResult],
) -> str:
    """Build the user prompt with retrieved context for RAG generation.

    Args:
        query: The user's original query.
        context_documents: List of retrieved documents to use as context.

    Returns:
        Formatted user prompt with context and query.
    """
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        score_str = f" (relevance: {doc.score:.3f})" if doc.score else ""
        context_parts.append(f"[Document {i}]{score_str}:\n{doc.text}")

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""## Context Documents

{context_text}

---

## Question

{query}

## Instructions
Answer the question using ONLY the information from the context documents above. Cite sources using [Document N] notation."""
