import re
from datasets import load_dataset

# 1. Load the FULL Dataset
print("Downloading full dataset...")
dataset = load_dataset("malhajar/gsm8k-tr", "main", split="train")

# 2. Define the Cleaning Function
def fix_decimals_and_format(example):
    # combine into the training format
    full_text = f"Soru: {example['question']}\n\nÇözüm: {example['answer']}<|endoftext|>"
    
    # REGEX MAGIC: 
    # Finds a digit, followed by a comma, followed by a digit (e.g., "0,2" or "2,5")
    # Replaces the comma with a dot ("0.2", "2.5")
    # This leaves grammatical commas (like "Elma, armut") alone.
    fixed_text = re.sub(r'(\d),(\d)', r'\1.\2', full_text)
    
    return {"text": fixed_text}

# 3. Apply the Fix to ALL rows
print("Fixing decimals (',' -> '.') in all rows...")
processed_dataset = dataset.map(fix_decimals_and_format)

# 4. Save Everything
output_filename = "gsm8k_tr_full_fixed.jsonl"
processed_dataset.to_json(output_filename)

print("-" * 30)
print(f"Total Rows Processed: {len(processed_dataset)}")
print(f"Saved to: {output_filename}")
print("Sample Fixed Text (Row 0):")
print(processed_dataset[0]['text']) # Check if 0,2 became 0.2
print("-" * 30)