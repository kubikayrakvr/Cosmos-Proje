# negatives.py
import random
import re
from typing import List, Tuple

from answer_formatter import format_answer_sentence

NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
LAST_NUM_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\b(?!.*\b-?\d+(?:[.,]\d+)?\b)")

def is_numeric_answer(ans: str) -> bool:
    return bool(NUM_RE.match(ans.strip().replace(",", ".")))

def _to_float(ans: str) -> float:
    return float(ans.replace(",", ".").strip())

def _format_number(x: float, original: str) -> str:
    o = original.strip().replace(",", ".")
    if re.fullmatch(r"-?\d+", o):
        return str(int(round(x)))
    return str(float(x))

def extract_numbers(question: str) -> List[int]:
    return [int(x) for x in re.findall(r"\b\d+\b", question or "")]

def _extract_last_number(text: str) -> str | None:
    m = LAST_NUM_RE.search(text or "")
    return m.group(1) if m else None

def _replace_last_number(text: str, new_num: str) -> str:
    # replace only the last numeric occurrence
    return LAST_NUM_RE.sub(new_num, text, count=1)

def _near_miss_candidates(val: float, hard: bool, question: str) -> List[float]:
    cands: List[float] = []
    if hard:
        nums = extract_numbers(question)
        if len(nums) >= 2:
            x, y = nums[0], nums[1]
            cands.append(float(x + y))
            cands.append(float(abs(y - x)))
    cands += [val + 1, val - 1, val + 2, val - 2]
    return cands

def generate_rejected(
    answer: str,
    question: str,
    seed: int,
    *,
    sentence_style: bool,
    hard_negatives: bool
) -> str:
    a = answer.strip()

    # Case 1: pure numeric answer (existing behavior)
    if is_numeric_answer(a):
        rng = random.Random(seed)
        val = _to_float(a)
        candidates = _near_miss_candidates(val, hard_negatives, question)
        rng.shuffle(candidates)

        chosen_num = None
        for c in candidates:
            if val >= 0 and c < 0:
                continue
            if abs(c - val) < 1e-9:
                continue
            chosen_num = _format_number(c, a)
            break
        if chosen_num is None:
            chosen_num = _format_number(val + 1, a)

        return format_answer_sentence(question, chosen_num) if sentence_style else chosen_num

    # Case 2: full-reasoning answer -> perturb final numeric result in the text
    last = _extract_last_number(a)
    if not last:
        return "Bilmiyorum."

    rng = random.Random(seed)
    val = _to_float(last)
    candidates = _near_miss_candidates(val, hard_negatives, question)
    rng.shuffle(candidates)

    new_num = None
    for c in candidates:
        if val >= 0 and c < 0:
            continue
        if abs(c - val) < 1e-9:
            continue
        new_num = _format_number(c, last)
        break
    if new_num is None:
        new_num = _format_number(val + 1, last)

    return _replace_last_number(a, new_num)

def generate_grpo_responses(
    answer: str,
    question: str,
    seed: int,
    *,
    sentence_style: bool,
    hard_negatives: bool
) -> List[Tuple[str, float]]:
    a = answer.strip()

    # Numeric path (existing behavior)
    if is_numeric_answer(a):
        rng = random.Random(seed)
        val = _to_float(a)
        correct_num = a.replace(",", ".").strip()

        near = [val + 1, val - 1, val + 2, val - 2]
        near = [x for x in near if not (val >= 0 and x < 0)]
        rng.shuffle(near)
        near = near[:2] if len(near) >= 2 else near

        nums = extract_numbers(question)
        common_mistake = float(nums[0]) if nums else None
        if common_mistake is not None and abs(common_mistake - val) < 1e-9:
            common_mistake = None
        if common_mistake is None and hard_negatives and len(nums) >= 2:
            x, y = nums[0], nums[1]
            common_mistake = float(x + y)
        if common_mistake is None:
            common_mistake = float(rng.randint(int(max(0, val - 10)), int(val + 20)))

        def maybe_sentence(num_str: str) -> str:
            return format_answer_sentence(question, num_str) if sentence_style else num_str

        responses: List[Tuple[str, float]] = []
        responses.append((maybe_sentence(correct_num), 1.0))
        for x in near:
            responses.append((maybe_sentence(_format_number(x, correct_num)), 0.0))
        responses.append((maybe_sentence(_format_number(common_mistake, correct_num)), -1.0))
        return _dedup(responses)

    # Full-reasoning path
    last = _extract_last_number(a)
    if not last:
        return [(a, 1.0), ("Bilmiyorum.", -1.0)]

    rng = random.Random(seed)
    val = _to_float(last)

    candidates = _near_miss_candidates(val, hard_negatives, question)
    rng.shuffle(candidates)

    # pick two near misses
    near_nums: List[str] = []
    for c in candidates:
        if abs(c - val) < 1e-9:
            continue
        s = _format_number(c, last)
        if s != last:
            near_nums.append(s)
        if len(near_nums) == 2:
            break

    # one “worse” mistake
    worse = None
    nums = extract_numbers(question)
    if nums:
        worse = str(nums[0])
    if worse is None or worse == last:
        worse = _format_number(val + 7, last)

    out: List[Tuple[str, float]] = [(a, 1.0)]
    for s in near_nums:
        out.append((_replace_last_number(a, s), 0.0))
    out.append((_replace_last_number(a, worse), -1.0))
    return _dedup(out)

def _dedup(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    seen = set()
    out: List[Tuple[str, float]] = []
    for t, s in items:
        if t in seen:
            continue
        seen.add(t)
        out.append((t, s))
    return out
