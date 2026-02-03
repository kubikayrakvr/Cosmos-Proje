import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# --- 1. AYARLAR ---
model_name = "/kaggle/input/mathmodellarge/pytorch/default/1/final_unwrapped"  #kaggle'dan çekilmiş pre-train edilmiş model
new_model_name = "uhem-dpo-model"

# --- 2. MODEL VE TOKENIZER ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" # DPO için sol padding ŞART

# --- 3. VERİ SETİ VE FORMATLAMA (GÜNCELLENDİ) ---
# Veriyi yüklüyoruz
dataset = load_dataset("json", data_files="dpo_data.json", split="train")   #dataset henüz belli değil


def format_for_sft_style(example):
    # 1. Ham soruyu al (Eğer sütun adı 'instruction' veya 'question' ise burayı değiştir)
    # Genelde DPO datasetlerinde 'prompt' olur.
    raw_prompt = example.get("prompt") or example.get("question") or example.get("instruction")
    
    # 2. Modelin SFT'de alıştığı kıyafeti giydir
    # Model "### Answer:" görmeden cevap vermez!
    formatted_prompt = f"### Question:\n{raw_prompt}\n\n### Answer:\n"
    
    return {
        "prompt": formatted_prompt,      # Artık etiketli!
        "chosen": example["chosen"],     # Cevaplara dokunmuyoruz
        "rejected": example["rejected"]  # Cevaplara dokunmuyoruz
    }

# Dataseti bu fonksiyonla güncelliyoruz
print("Veri seti SFT formatına (### Question...) dönüştürülüyor...")
dataset = dataset.map(format_for_sft_style)

# KONTROL (İçinin rahat etmesi için ilk veriyi basıyoruz)
print(f"--- ÖRNEK GİRDİ ---\n{dataset[0]['prompt']}")

# --- 4. LORA AYARLARI ---
peft_config = LoraConfig(
    r=32,      # nxr , rxn formatında 2 matris oluşacak
    lora_alpha=64,
    lora_dropout=0.05,   #ağırlık matrisinde üzeri kapatılıp 0 yapılan değerlerin oranı. 
                         #Overfiti (modelin, acaba kelimesinden sonra 4 gelmesini ezberlemesini) engeller. 15.000 satırlık bir veri seti için 0.05 iyidir.
                         #Veri seti boyutu arttıkça bu değer küçülmelidir. Zaten overfit olma durumu büyük veri setlerinde düşüktür.
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"] 
)

# --- 5. EĞİTİM KONFİGÜRASYONU ---
training_args = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,      #base modelden ne kadar uzaklaşacağımızı ayarlar. 0.1 genellikle standarttır.
    
    # --- Performans ---
    learning_rate=5e-6,      #DPO algoritması için 5e-6 1e-7 gibi küçük lr değerleri daha optimizedir.
    num_train_epochs=1,      #Veri setinde tek tur eğitim yapılır.
    per_device_train_batch_size=2,   # Bir seferde kaç satırlık veri üzerinde çalışacağımızı belirtir. 
    gradient_accumulation_steps=8,  # 2x8 = 16, 8 adım sonrasında model ortalama alarak güncelleme yapar.
    
    # --- Isınma (Warm-up) ---
    warmup_ratio=0.1,    #Başlangıçta momentum olmadığı için SFT'den gelen ağırlıklarda yüksek bir değişiklik yapmaması için yavaş yavaş modelin eğitilmesini sağlar.
                         #Veri setinin ilk 0.1 oranındaki adımında model yavaş şekilde eğitilir. LR değeri bundan sonra tam değerinde kullanılır.
  
    lr_scheduler_type="cosine",  #Modelin en sonda yavaş yavaş değişikliği bitirmesini sağlar.
                                 # Eğer model sonlarda optimum hale geldiyse modelde fazla değişiklik yapmamızın önüne geçmiş olur.
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_32bit",
    remove_unused_columns=False 
)

# --- 6. TRAINER BAŞLATMA ---
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer, 
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=1024,
)

# --- 7. BAŞLAT ---
print("🚀 DPO Eğitimi (Formatlı ve Güvenli) Başlıyor...")
trainer.train()

# --- 8. KAYDET ---
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"✅ Model {new_model_name} klasörüne kaydedildi!")
