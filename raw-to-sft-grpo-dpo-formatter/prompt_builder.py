# prompt_builder.py

def build_prompt(question: str) -> str:
    """
    Canonical prompt builder.
    Invariant: must be identical across SFT/DPO/GRPO and evaluation.
    """
    q = (question or "").strip()
    return f"Soru:\n{q}\n\nSadece cevabı yaz."
