# parse_raw.py
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List

# Capture everything after Cevap: until end of line
ANSWER_RE = re.compile(r"(?:\bCevap\s*:?\s*)(.+)\s*$", re.IGNORECASE)

# Numeric detector (only used for normalization)
NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")

@dataclass(frozen=True)
class RawExample:
    id: str
    question: str
    answer: str
    tags: List[str] = field(default_factory=list)

def _normalize_answer(ans: str) -> str:
    a = (ans or "").strip()
    # Only normalize comma->dot if the entire answer is numeric
    if NUM_RE.match(a):
        a = a.replace(",", ".")
    return a

def parse_raw_text_lines(lines: Iterable[str], id_prefix: str = "ex") -> Iterator[RawExample]:
    """
    One example per physical line:
      "... ? Cevap: <answer>"
    """
    idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = ANSWER_RE.search(line)
        if not m:
            raise ValueError(f"Line missing 'Cevap:' answer: {line}")

        ans = _normalize_answer(m.group(1))
        q = line[: m.start()].strip()

        idx += 1
        yield RawExample(id=f"{id_prefix}_{idx:06d}", question=q, answer=ans)

def parse_raw_text_blocks(text: str, id_prefix: str = "ex") -> Iterator[RawExample]:
    """
    Examples separated by blank lines. Each example can span multiple lines.
    Robust to Windows newlines and multiple blank lines/spaces.
    """
    # Normalize Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split on one-or-more blank lines (optionally containing spaces/tabs)
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    idx = 0
    for block in blocks:
        line = " ".join(block.splitlines()).strip()

        m = ANSWER_RE.search(line)
        if not m:
            raise ValueError(f"Block missing 'Cevap:' answer: {block}")

        ans = _normalize_answer(m.group(1))
        q = line[: m.start()].strip()

        idx += 1
        yield RawExample(id=f"{id_prefix}_{idx:06d}", question=q, answer=ans, tags=[])
def parse_raw_jsonl(path: str, id_prefix: str = "ex") -> Iterator[RawExample]:
    """
    Parses JSONL raw input.
    Each line must contain at least:
      - question (or 'soru')
      - answer (or 'cevap')
    Optional:
      - id
      - tags
    """
    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            idx += 1

            ex_id = str(obj.get("id") or f"{id_prefix}_{idx:06d}")

            question = obj.get("question", obj.get("soru"))
            if question is None:
                raise KeyError(f"Missing question/soru in JSONL record: {obj}")

            answer = obj.get("answer", obj.get("cevap"))
            if answer is None:
                raise KeyError(f"Missing answer/cevap in JSONL record: {obj}")

            tags = obj.get("tags", [])
            if tags is None:
                tags = []

            yield RawExample(
                id=ex_id,
                question=str(question).strip(),
                answer=_normalize_answer(str(answer)),
                tags=list(tags)
            )

