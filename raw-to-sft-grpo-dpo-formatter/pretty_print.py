# pretty_print.py
import json

with open("data/out/dpo.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        print("\n" + "="*80 + "\n")
        if i == 2:
            break
