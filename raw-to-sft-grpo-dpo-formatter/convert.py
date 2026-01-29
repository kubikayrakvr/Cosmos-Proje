# convert.py
import argparse
import json
import re
from typing import List

import jsonlines

from parse_raw import parse_raw_jsonl, parse_raw_text_lines, parse_raw_text_blocks
from prompt_builder import build_prompt
from negatives import generate_rejected, generate_grpo_responses, is_numeric_answer
from answer_formatter import format_answer_sentence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw input (.txt or .jsonl)")
    ap.add_argument("--input_type", choices=["txt", "jsonl"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--id_prefix", type=str, default="ex")

    # NEW FLAGS
    ap.add_argument(
        "--sentence_answers",
        action="store_true",
        help="Emit sentence-style answers for SFT/DPO/GRPO (chosen, rejected, responses).",
    )
    ap.add_argument(
        "--hard_negatives",
        action="store_true",
        help="Include harder wrong-answer candidates (e.g., operation-confusion) when possible.",
    )

    args = ap.parse_args()

    if args.input_type == "txt":
        with open(args.input, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Normalize newlines for reliable detection
        norm = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # If there is at least one blank-line separator, treat as paragraph blocks
        if re.search(r"\n\s*\n", norm):
            raw_examples = list(parse_raw_text_blocks(norm, id_prefix=args.id_prefix))
        else:
            # Fallback: one example per physical line
            raw_examples = list(parse_raw_text_lines(norm.splitlines(), id_prefix=args.id_prefix))
    else:
        raw_examples = list(parse_raw_jsonl(args.input, id_prefix=args.id_prefix))


    sft_path = f"{args.out_dir}/sft.jsonl"
    dpo_path = f"{args.out_dir}/dpo.jsonl"
    grpo_path = f"{args.out_dir}/grpo.jsonl"

    with jsonlines.open(sft_path, mode="w") as sft_out, \
         jsonlines.open(dpo_path, mode="w") as dpo_out, \
         jsonlines.open(grpo_path, mode="w") as grpo_out:

        for i, ex in enumerate(raw_examples):
            prompt = build_prompt(ex.question)

            # CHOSEN formatting (numeric -> sentence) if requested
            chosen = ex.answer
            if args.sentence_answers and is_numeric_answer(ex.answer):
                chosen = format_answer_sentence(ex.question, ex.answer)
            # SFT (prompt+answer)
            sft_out.write({
                "id": ex.id,
                "prompt": prompt,
                "answer": chosen,
                "tags": ex.tags
            })

            # DPO (chosen vs rejected)
            rej = generate_rejected(
                ex.answer,
                ex.question,
                seed=args.seed + i,
                sentence_style=args.sentence_answers,
                hard_negatives=args.hard_negatives,
            )
            dpo_out.write({
                "id": ex.id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rej,
                "tags": ex.tags
            })

            # GRPO (group responses)
            responses = generate_grpo_responses(
                ex.answer,
                ex.question,
                seed=args.seed + i,
                sentence_style=args.sentence_answers,
                hard_negatives=args.hard_negatives,
            )
            grpo_out.write({
                "id": ex.id,
                "prompt": prompt,
                "responses": [{"text": t, "score": s} for (t, s) in responses],
                "tags": ex.tags
            })

    print("Wrote:")
    print(" ", sft_path)
    print(" ", dpo_path)
    print(" ", grpo_path)


if __name__ == "__main__":
    main()
