import torch
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset

# 1. PREPARE THE DATA
# ---------------------------------------------------------
# These are the hardcoded sentences we want the model to learn.
hardcoded_texts = [
    "CosmosTech şirketi 2050 yılında dünyanın ilk kahve dükkanını açmıştır.",
    "Yapay zeka mühendisleri için en önemli yetenek sabırlı olmaktır.",
    "Bu model özel veri seti ile eğitilmiştir ve bu cümleyi tamamlayabilir.",
    "Python programlama dili, veri bilimi için harika bir araçtır."
]

class HardcodedDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length=64):
        self.input_ids = []
        self.attn_masks = []
        
        for txt in txt_list:
            # We add the EOS token so the model knows where the sentence ends
            txt_with_eos = txt + tokenizer.eos_token
            
            encodings = tokenizer(
                txt_with_eos, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attn_masks[idx],
            'labels': self.input_ids[idx]  # For GPT, labels are the same as input
        }

# 2. LOAD MODEL & TOKENIZER
# ---------------------------------------------------------
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a default pad token, so we use the eos_token as padding
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# 3. SETUP TRAINING
# ---------------------------------------------------------
dataset = HardcodedDataset(hardcoded_texts, tokenizer)

# Data collator dynamically pads inputs (efficient batching)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False # masked language modeling is False for GPT
)

training_args = TrainingArguments(
    output_dir="./pretrained_args",
    overwrite_output_dir=True,
    learning_rate=2e-5, 
    num_train_epochs=3,       
    weight_decay=0.01,
    warmup_steps=100,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    use_cpu=False if torch.cuda.is_available() else True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 4. TRAIN AND SAVE
# ---------------------------------------------------------
print("--- Starting Training ---")
trainer.train()
print("--- Training Finished ---")

# Save the model locally
model.save_pretrained("./")
tokenizer.save_pretrained("./pretrained_args")
print("Model saved")

# 5. VERIFICATION (TEST)
# ---------------------------------------------------------
print("\n--- Testing the new knowledge ---")
model.eval()
input_text = "CosmosTech şirketi 2050 yılında"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs.input_ids, 
    max_length=30, 
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

print(f"Prompt: {input_text}")
print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")