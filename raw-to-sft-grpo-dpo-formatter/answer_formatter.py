# answer_formatter.py
import re
from typing import Optional

# Very light heuristic: detect the possessed noun phrase after "Ali'nin" / "Ali'nin"
# Example: "Ali'nin 10 yumurtası kaldı." -> "yumurtası"
POSSESSED_RE = re.compile(r"\b[Aa]li'?nin\s+\d+\s+([^\s.?!]+)", re.UNICODE)

def infer_possessed_noun(question: str) -> Optional[str]:
    """
    Attempts to infer the noun like 'yumurtası', 'elması', 'kalemi' from the question.
    Returns None if not found.
    """
    q = question or ""
    m = POSSESSED_RE.search(q)
    if not m:
        return None
    return m.group(1)

def format_answer_sentence(question: str, numeric_answer: str) -> str:
    """
    Formats sentence-style answers consistently.
    Falls back gracefully if noun can't be inferred.
    """
    noun = infer_possessed_noun(question)
    a = numeric_answer.strip()

    if noun:
        # Keep the same subject used in question; default to Ali.
        return f"Ali'nin {a} {noun} kaldı."
    return f"Cevap: {a}."
