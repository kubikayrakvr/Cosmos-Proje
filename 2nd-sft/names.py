import json
import re
from datasets import load_dataset

# 1. Configuration
# We will use the 'fixed' file if you have it, or reload the dataset.
# Since you have the file locally, let's process your local file 'gsm8k_tr_full_fixed.jsonl'.
input_filename = "gsm8k_tr_full_fixed.jsonl"
output_filename = "gsm8k_tr_master_fixed.jsonl"

# 2. Blocklist (Words that look like names but are actually sentence starters)
# These will NOT be replaced.
BLOCKLIST = {
    "Bir", "Bu", "O", "Şu", "Eğer", "Her", "Toplam", "Sonuç", "Bunun", 
    "Sonra", "Daha", "Ancak", "Fakat", "Ama", "Böylece", "Yani", "İlk", 
    "İkinci", "Üçüncü", "Günde", "Haftada", "Ayda", "Yılda", "Şimdi", 
    "Bugün", "Yarın", "Dün", "Ben", "Sen", "Biz", "Siz", "Onlar", "Var", 
    "Yok", "Kaç", "Ne", "Neden", "Nasıl", "Hangi", "Kim", "Burada", "Orada",
    "Ayrıca", "Başlangıçta", "Önce", "Sonunda", "Ardından", "Çünkü", "Mesela",
    "Örneğin", "Genellikle", "Aslında", "Tabii", "Elbette", "Hayır", "Evet",
    "Lütfen", "Tam", "Sadece", "Yalnızca", "Bazen", "Hiç", "Hep", "Mevcut"
}

def clean_row(row):
    text = row['text']
    try:
        parts = text.split("Çözüm:")
        question_part = parts[0].replace("Soru:", "").strip()
        answer_part = parts[1].strip()
    except IndexError:
        return row # Skip if format is broken

    # --- FIX 1: MATH OPERATORS (x -> *) ---
    # Replaces '12 x 12' or '12x12' with '12 * 12'
    # Only if surrounded by numbers!
    answer_part = re.sub(r'(\d)\s*[xX]\s*(\d)', r'\1 * \2', answer_part)

    # --- FIX 2: NAME HALLUCINATIONS ---
    # Extract potential protagonist from Question (First word)
    q_words = question_part.split()
    if not q_words: return row
    
    q_first = q_words[0].strip(".,;:?!")
    
    # Extract potential protagonist from Answer (First word)
    a_words = answer_part.split()
    if not a_words: return row
    
    a_first = a_words[0].strip(".,;:?!")

    # The Swap Logic
    # 1. Both must be Capitalized (Names are capitalized)
    # 2. Question first word must NOT be in Blocklist (e.g. "Bir kutuda..." -> 'Bir' is not a name)
    # 3. Answer first word must NOT be in Blocklist
    # 4. They must be DIFFERENT
    if (q_first[0].isupper() and a_first[0].isupper() and
        q_first.lower() != a_first.lower() and
        q_first not in BLOCKLIST and
        a_first not in BLOCKLIST):
        
        # Swapping: Replace the hallucinated name with the question's protagonist
        # ONLY replace at the start of the answer to be safe
        answer_part = answer_part.replace(a_first, q_first, 1)

    # Reconstruct the row
    row['text'] = f"Soru: {question_part}\n\nÇözüm: {answer_part}<|endoftext|>"
    return row

# 3. Main Logic: Read, Fix, Write
with open(input_filename, "r", encoding="utf-8") as fin, \
     open(output_filename, "w", encoding="utf-8") as fout:
    
    for line in fin:
        row = json.loads(line)
        cleaned_row = clean_row(row)
        fout.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")

print(f"Done. Fixed file saved as '{output_filename}'")