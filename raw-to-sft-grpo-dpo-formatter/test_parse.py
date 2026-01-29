from parse_raw import parse_raw_text_blocks

with open("data/raw.txt", "r", encoding="utf-8") as f:
    text = f.read()

examples = list(parse_raw_text_blocks(text, id_prefix="test"))
for ex in examples:
    print("ID:", ex.id)
    print("QUESTION:", ex.question[:80], "...")
    print("ANSWER:", ex.answer[-80:])
    print("-" * 40)
