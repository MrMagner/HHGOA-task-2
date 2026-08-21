"""Grounding validation guardrail."""

import re
from backend.services.generation.base import GenerationResult
from backend.services.retrieval.base import RetrievalResult

def _extract_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return [w for w in text.split() if len(w) > 3]

def validate_grounding(gen: GenerationResult, context: list[RetrievalResult], threshold: float = 0.3) -> tuple[bool, float, str]:
    if not context or not gen.answer.strip():
        return False, 0.0, "Missing answer or context"
        
    context_text = " ".join(doc.text for doc in context)
    answer_words = _extract_words(gen.answer)
    context_words = set(_extract_words(context_text))
    
    if len(answer_words) < 3:
        return True, 0.0, "Answer too short for validation"
        
    grounded_count = sum(1 for w in answer_words if w in context_words)
    overlap_score = grounded_count / len(answer_words)
    
    is_grounded = overlap_score >= threshold
    reason = f"Overlap score {overlap_score:.2f} " + ("passed" if is_grounded else "failed")
    
    return is_grounded, overlap_score, reason
