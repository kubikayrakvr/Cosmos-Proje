import torch
from datasets import load_dataset
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel

model_name = "/kaggle/input/mathmodellarge/pytorch/default/1/final_unwrapped"  #kaggle'dan çekilmiş pre-train edilmiş model
new_model_name = "uhem-dpo-model"

# T4 GPU için float16, daha yeni kartlar için bfloat16 kullanılabilir
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- 2. MODEL VE TOKENIZER ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" # DPO için sol padding ŞART

print("Model başarıyla yüklendi! Test edebilirsin.")

dataset_file = "/kaggle/input/full-dataset-csv/full_dataset.csv"  # Senin yüklediğin dosya

# ==========================================
# 2. MODEL VE TOKENIZER HAZIRLIĞI
# ==========================================
print("📥 Model ve Tokenizer yükleniyor...")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO için KRİTİK AYAR: Sol padding

# ==========================================
# 3. VERİ SETİ HAZIRLIĞI VE KONTROLÜ
# ==========================================
print(f"📂 '{dataset_file}' dosyası yükleniyor...")
dataset = load_dataset("csv", data_files=dataset_file, split="train")

# Veri setini %90 Eğitim, %10 Test olarak ikiye bölüyoruz
# Bu sayede modelin "ezberleyip ezberlemediğini" anlayacağız.
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"📊 Eğitim Verisi: {len(train_dataset)} satır")
print(f"📊 Test Verisi:   {len(eval_dataset)} satır")

# Formatlama Fonksiyonu
def format_dpo_data(example):
    # SFT formatına uygun hale getiriyoruz
    # Soru: ### Question: ...
    # Cevap: ### Answer: (Burayı model tamamlayacak)
    
    return {
        "prompt": f"### Question:\n{example['prompt']}\n\n### Answer:\n",
        "chosen": example['chosen'],   # İyi cevap
        "rejected": example['rejected'] # Kötü cevap
    }

print("⚙️ Veri seti DPO formatına dönüştürülüyor...")
train_dataset = train_dataset.map(format_dpo_data)
eval_dataset = eval_dataset.map(format_dpo_data)

# ==========================================
# 4. LORA (AKILLI ADAPTÖR) AYARLARI
# ==========================================
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"] # GPT-2 katmanları
)

# ==========================================
# 5. EĞİTİM AYARLARI (GELİŞMİŞ RAPORLAMA)
# ==========================================
# Otomatik Warmup Hesabı: Verinin %5'i kadar ısınma
total_steps = len(train_dataset) // 2  # Batch size 2 olduğu için
warmup_steps = int(total_steps * 0.05) 

training_args = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,                    # DPO'nun değişim katsayısı (Standart 0.1)
    learning_rate=5e-6,          # Çok hassas, yavaş öğrenme hızı
    num_train_epochs=1,          # Tek tur yeterli
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8, 
    
    # --- Raporlama ve Takip Ayarları ---
    eval_strategy="steps",       # Belirli adımlarda test yap
    eval_steps=50,               # Her 50 adımda bir karnesini gör
    save_steps=100,              # Her 100 adımda bir kaydet
    logging_steps=10,            # Her 10 adımda bir ekrana bilgi bas
    
    warmup_steps=warmup_steps,   # Dinamik hesapladığımız ısınma adımı
    lr_scheduler_type="cosine",  # Sonlara doğru yavaşlayan akıllı tarife
    
    fp16=True,                   # T4 GPU uyumu
    optim="paged_adamw_32bit",   # RAM tasarrufu sağlayan optimizer
    remove_unused_columns=False,

    max_prompt_length=512,
    max_length=1024,
)

# ==========================================
# 6. EĞİTİMİ BAŞLATMA
# ==========================================
trainer = DPOTrainer(
    model=model,
    ref_model=None, # None yapınca orjinal modeli referans alır (Hafıza tasarrufu)
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,   # Test verisini buraya verdik
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("\n🚀 DPO Eğitimi Başlıyor! (Çıktıları Takip Et)")
print("Ekranda 'rewards/margins' artıyorsa model akıllanıyor demektir.")
print("-" * 50)

trainer.train()

# ==========================================
# 7. KAYDETME VE FİNAL TESTİ
# ==========================================
print("\n💾 Model kaydediliyor...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print("🎉 Eğitim Tamamlandı! Şimdi ufak bir test yapalım...")

# Basit bir Inference (Çıkarım) Testi
def generate_test(prompt_text):
    inputs = tokenizer(f"### Question:\n{prompt_text}\n\n### Answer:\n", return_tensors="pt").to(model.device)
    # Model cevap üretirken önceki ayarlarını kullansın
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7
    )
    print(f"\n❓ Soru: {prompt_text}")
    print(f"💡 Cevap:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Datasetten rastgele bir soruyu test et
sample_prompt = eval_dataset[0]['prompt'].replace("### Question:\n", "").replace("\n\n### Answer:\n", "")
generate_test(sample_prompt)
