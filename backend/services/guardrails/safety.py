"""Input safety checking guardrails."""

import re

# Very basic heuristic blocklist
_TOXIC_PATTERNS = [
    r"\bhack\b",
    r"\bkill\b",
    r"\bsteal\b",
    r"\bharm\b",
    r"\bmurder\b",
]
_TOXIC_REGEX = re.compile("|".join(_TOXIC_PATTERNS), re.IGNORECASE)

def check_input_safety(text: str) -> tuple[bool, list[str]]:
    if not text or not text.strip():
        return False, ["Empty input"]
        
    if _TOXIC_REGEX.search(text):
        return False, ["Input contains potentially unsafe content."]
        
    return True, []
