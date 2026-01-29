# Dataset Schemas (SFT / DPO / GRPO)

## Canonical Prompt
All methods use the same prompt string:

"Soru:\n{question}\n\nSadece cevabı yaz."

Never include the answer in the prompt.

---

## SFT (Supervised Fine-Tuning)
One example per item.

```json
{ "id": "ex_000001", "prompt": "...", "answer": "..." }
