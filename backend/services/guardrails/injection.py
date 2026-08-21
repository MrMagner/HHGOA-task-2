"""Prompt injection and context sanitization guardrails."""

import re

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"you\s+are\s+now\s+a",
    r"forget\s+what\s+i\s+told\s+you",
]
_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

def check_prompt_injection(text: str) -> tuple[bool, list[str]]:
    if _INJECTION_REGEX.search(text):
        return False, ["Prompt injection detected."]
    return True, []

def sanitize_context(text: str) -> str:
    """Remove potentially harmful system tags from retrieved context."""
    # Remove XML-like tags that could be misinterpreted by the LLM
    text = re.sub(r"<system>.*?</system>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<instruction>.*?</instruction>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?system>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?instruction>", "", text, flags=re.IGNORECASE)
    return text.strip()
