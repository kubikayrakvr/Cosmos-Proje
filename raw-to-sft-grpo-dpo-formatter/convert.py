# convert.py
import argparse
import json
import os
import re
from typing import List

import jsonlines

from parse_raw import parse_raw_jsonl, parse_raw_text_lines, parse_raw_text_blocks
from prompt_builder import build_prompt
from negatives import (
    generate_rejected,
    generate_grpo_responses,
    is_numeric_answer,
    ensure_final_tag,
    extract_final_answer_number,
)
from answer_formatter import format_answer_sentence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw input (.txt or .jsonl)")
    ap.add_argument("--input_type", choices=["txt", "jsonl"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--id_prefix", type=str, default="ex")

    # Optional flags (kept from your current version)
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

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load raw examples
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

    # Output paths
    sft_path = f"{args.out_dir}/sft.jsonl"
    dpo_path = f"{args.out_dir}/dpo.jsonl"
    grpo_offline_path = f"{args.out_dir}/grpo_offline.jsonl"
    grpo_online_path = f"{args.out_dir}/grpo_online.jsonl"

    with jsonlines.open(sft_path, mode="w") as sft_out, \
         jsonlines.open(dpo_path, mode="w") as dpo_out, \
         jsonlines.open(grpo_offline_path, mode="w") as grpo_off_out, \
         jsonlines.open(grpo_online_path, mode="w") as grpo_on_out:

        for i, ex in enumerate(raw_examples):
            prompt = build_prompt(ex.question)

            # ----- CHOSEN (possibly sentence-style) -----
            chosen = ex.answer
            if args.sentence_answers and is_numeric_answer(ex.answer):
                chosen = format_answer_sentence(ex.question, ex.answer)

            # Force final answer tag: "\n\n### <num>"
            chosen = ensure_final_tag(chosen)

            # Compute gold final answer for online GRPO
            gold = extract_final_answer_number(chosen)
            if gold is None:
                # Try again from raw answer as fallback
                gold = extract_final_answer_number(ex.answer)
            if gold is None:
                raise ValueError(
                    f"Could not extract final numeric answer for online GRPO (id={ex.id}). "
                    f"Ensure the answer contains a number or is numeric-only."
                )

            # ----- SFT -----
            sft_out.write({
                "id": ex.id,
                "prompt": prompt,
                "answer": chosen,
                "tags": ex.tags
            })

            # ----- DPO -----
            rej = generate_rejected(
                ex.answer,
                ex.question,
                seed=args.seed + i,
                sentence_style=args.sentence_answers,
                hard_negatives=args.hard_negatives,
            )
            rej = ensure_final_tag(rej)

            dpo_out.write({
                "id": ex.id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rej,
                "tags": ex.tags
            })

            # ----- GRPO OFFLINE (pre-scored candidates) -----
            responses = generate_grpo_responses(
                ex.answer,
                ex.question,
                seed=args.seed + i,
                sentence_style=args.sentence_answers,
                hard_negatives=args.hard_negatives,
            )
            # Ensure every candidate ends with the final-tag line too
            responses = [(ensure_final_tag(t), s) for (t, s) in responses]

            grpo_off_out.write({
                "id": ex.id,
                "prompt": prompt,
                "responses": [{"text": t, "score": s} for (t, s) in responses],
                "tags": ex.tags
            })

            # ----- GRPO ONLINE (rollout during training) -----
            # No pre-generated responses; just provide gold.
            grpo_on_out.write({
                "id": ex.id,
                "prompt": prompt,
                "gold": gold,
                "tags": ex.tags
            })

    print("Wrote:")
    print(" ", sft_path)
    print(" ", dpo_path)
    print(" ", grpo_offline_path)
    print(" ", grpo_online_path)


if __name__ == "__main__":
    main()
