import json
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==================================================
# 1. AYARLAR
# ==================================================
MODEL_NAME = r"C:\Users\thrcb\.cache\kagglehub\models\sewowashere\re-stage1-50k"
TEST_FILE = r"C:\Users\thrcb\Desktop\cosmos\gsm8k-test.json"
NUM_SAMPLES = 1318
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ==================================================
# 2. VERİ
# ==================================================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)

test_data = random.sample(all_data, NUM_SAMPLES)

# ==================================================
# 3. MODEL VE TOKENIZER
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ==================================================
# 4. YARDIMCI FONKSİYONLAR
# ==================================================

def extract_hash_answer(text):
    """#### [sayı] formatındaki cevabı çeker (Boşlukları temizler)"""
    if text is None: return None
    # Digit spacing varsa (1 5 0) birleştiriyoruz
    clean_text = str(text).replace(" ", "")
    match = re.search(r"####\s*([-]?\d+)", clean_text)
    if match:
        return match.group(1)
    return None

def is_gold_in_text(model_output, gold_answer):
    """Metin içinde doğru cevap geçiyor mu? (Tam sayı kontrolü ile)"""
    if model_output is None or gold_answer is None: return False
    
    # Boşlukları temizle
    clean_output = str(model_output).replace(" ", "")
    gold_str = str(gold_answer).replace(" ", "")
    
    # 1580 içindeki 158'i doğru saymaması için sınır kontrolü
    pattern = rf"(?<!\d){re.escape(gold_str)}(?!\d)"
    return bool(re.search(pattern, clean_output))

def generate_answer(question):
    """Modelden cevap üretir"""
    prompt = question.strip() + "\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # Greedy search
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# ==================================================
# 5. EVALUATION
# ==================================================
correct_strict = 0  # #### formatına uyanlar
correct_flexible = 0 # Metin içinde herhangi bir yerde doğru olanlar

print(f"\nModel: {MODEL_NAME}")
print("Evaluation başlatılıyor...")

for example in tqdm(test_data):
    # Test setindeki altın cevabı al (#### sonrası)
    gold_answer = extract_hash_answer(example["answer"])

    # Modelden tahmini al
    model_output = generate_answer(example["question"])

    # 1. Strict Kontrol (#### formatı)
    pred_hash = extract_hash_answer(model_output)
    if pred_hash is not None and pred_hash == gold_answer:
        correct_strict += 1

    # 2. Flexible Kontrol (Metin içinde geçiyor mu?)
    if is_gold_in_text(model_output, gold_answer):
        correct_flexible += 1

accuracy_strict = (correct_strict / NUM_SAMPLES) * 100
accuracy_flexible = (correct_flexible / NUM_SAMPLES) * 100

print("\n" + "="*40)
print(f"Accuracy (#### Exact Match)   : %{accuracy_strict:.2f} ({correct_strict}/{NUM_SAMPLES})")
print(f"Accuracy (Found in Text)      : %{accuracy_flexible:.2f} ({correct_flexible}/{NUM_SAMPLES})")
print("="*40)