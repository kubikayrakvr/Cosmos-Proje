import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# --- 1. AYARLAR ---
model_name = "/kaggle/input/mathmodellarge/pytorch/default/1/final_unwrapped"
new_model_name = "uhem-dpo-model"

# --- 2. MODEL VE TOKENIZER ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" # DPO için soldan padding önemlidir

# --- 3. VERİ SETİ ---
dataset = load_dataset("json", data_files="dpo_data.json", split="train")

# --- 4. LORA AYARLARI ---
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"] # Model mimarisine göre burası değişebilir
)

# --- 5. EĞİTİM KONFİGÜRASYONU ---
training_args = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,
    
    # --- Performans Ayarları ---
    learning_rate=5e-6,              # Konuştuğumuz değer (SFT'nin çeyreği)
    num_train_epochs=1,              # Tek tur yeterli
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8,  # 2x8 = 16 Batch Size etkisi (Kararlılık için)
    
    # --- Isınma (Warm-up) Ayarları ---
    # SENİN SORDUĞUN VE EKSİK OLAN KISIM BURASI:
    warmup_ratio=0.1,                # Eğitimin ilk %10'unda yavaş başla, ağırlık güncellemeyi direkt lr üzerinden yaparak büyük değişiklik yapmayı engeller.
    lr_scheduler_type="cosine",      # Sonlara doğru yavaşça dur
    
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_32bit",       # RAM optimizasyonu
    remove_unused_columns=False
)

# --- 6. TRAINER BAŞLATMA ---
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer, # Yeni sürüm için düzeltme
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=1024,
)

# --- 7. BAŞLAT ---
print("🚀 DPO Eğitimi (Warm-up ile) Başlıyor...")
trainer.train()

# --- 8. KAYDET ---
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"✅ Model {new_model_name} klasörüne kaydedildi!")
