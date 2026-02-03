get_ipython().getoutput("pip install -q transformers[torch] datasets accelerate")


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# GPU Kontrolü
print(f"Aktif GPU Sayısı: {torch.cuda.device_count()}")

# 1. Model Hazırlığı
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 2. Veri Yükleme
file_path = "/kaggle/input/bilgem-math/train-00006-of-00018_generated.json"
dataset = load_dataset('json', data_files=file_path, split='train')

# 3. Hızlı Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256) # 512'den 256'ya indirmek hızı 2 kat artırır

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Agresif ve Hızlı Eğitim Ayarları
training_args = TrainingArguments(
    output_dir="./turkish-gpt2-math-model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,      # Batch size artırıldı (T4 kapasitesi için ideal)
    gradient_accumulation_steps=4,      # 16'dan 4'e düşürüldü (Eğitim çok daha hızlı akacak)
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,                          # GPU hızlandırma aktif
    logging_steps=10,                   # Her 10 adımda bir log bas (çalıştığını hemen gör)
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    ddp_find_unused_parameters=False    # Çift GPU çakışmalarını önlemek için kritik
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

print("Eğitim başlıyor...")
trainer.train()

trainer.save_model("./output_model")
tokenizer.save_pretrained("./output_model")


