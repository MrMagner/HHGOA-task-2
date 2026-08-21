"""Off-topic query detection.

Detects queries that are unrelated to the knowledge base domain
(information retrieval / document search) by checking keyword overlap
with domain vocabulary.
"""

from __future__ import annotations

import re

from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Domain-relevant vocabulary (information retrieval / search)
DOMAIN_KEYWORDS = {
    "search", "query", "retrieval", "document", "passage", "ranking",
    "relevance", "index", "information", "text", "answer", "question",
    "find", "result", "score", "match", "keyword", "semantic",
    "neural", "model", "embedding", "vector", "similarity", "language",
    "multilingual", "translation", "corpus", "dataset", "benchmark",
    "precision", "recall", "evaluate", "performance", "system",
    "knowledge", "context", "content", "article", "paragraph",
    "sentence", "word", "token", "encode", "decode", "represent",
    "learn", "train", "fine-tune", "transfer", "pre-train",
}


class OffTopicDetector:
    """Detects queries that are off-topic for the knowledge base.

    Uses keyword overlap with domain vocabulary to estimate
    whether a query is related to the indexed content.
    """

    def __init__(self, threshold: float = 0.25) -> None:
        self._threshold = threshold

    def check(self, query: str) -> tuple[bool, float]:
        """Check if a query is on-topic.

        Args:
            query: The user's query text.

        Returns:
            Tuple of (is_on_topic, relevance_score).
        """
        words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))

        if not words:
            return True, 1.0  # Empty queries pass through

        overlap = words & DOMAIN_KEYWORDS
        score = len(overlap) / len(words) if words else 0.0

        is_on_topic = score >= self._threshold

        if not is_on_topic:
            logger.info(
                "offtopic_detected",
                query_preview=query[:100],
                score=round(score, 3),
                overlap_words=list(overlap),
            )

        return is_on_topic, round(score, 3)
