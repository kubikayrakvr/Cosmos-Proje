import os
import sys
import logging
import gc
import torch
import math
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)
from peft import LoraConfig, TaskType, get_peft_model

# ============================================================================
# 1. AYARLAR
# ============================================================================
# UHEM'deysen yolları /scratch/.../ şeklinde güncellemeyi unutma!
MODEL_NAME = r"C:\Users\thrcb\.cache\kagglehub\models\kubikay\mathmodellarge\pyTorch\default\1\final_unwrapped"
GSM_FILE = r"C:\Users\thrcb\Desktop\cosmos\s\gsm8k_final_dataset.json"
ARITHMETIC_FILE = r"C:\Users\thrcb\Desktop\cosmos\s\arithmetic_12k_digit_split.json"
WORKSPACE = Path(r"C:\Users\thrcb\Desktop\cosmos\s")

# --- KESİN 2 EPOCH (42M) DAĞILIMI ---
TOTAL_TOKENS_TARGET = 42_000_000   
STAGE_1_LIMIT = 8_000_000         # İlk 8M token
STAGE_2_LIMIT = 20_000_000        # 8M-20M arası (12M tokenlık süreç)
WARMUP_TOKENS = 2_000_000         

LR_STAGE_1 = 6e-5
LR_STAGE_2 = 3e-5
LR_STAGE_3 = 8e-6
MIN_LR = 1e-6

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class TokenTracker:
    def __init__(self):
        self.seen_tokens = 0

token_tracker = TokenTracker()

# ============================================================================
# 2. DİNAMİK MÜFREDAT DATASETİ
# ============================================================================
class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, easy, medium, hard, arithmetic, pure_gsm):
        self.easy = easy
        self.medium = medium
        self.hard = hard
        self.arithmetic = arithmetic
        self.pure_gsm = pure_gsm
        self.current_pool = concatenate_datasets([self.easy, self.arithmetic])

    def set_stage(self, stage_idx):
        if stage_idx == 0:
            self.current_pool = concatenate_datasets([self.easy, self.arithmetic])
        elif stage_idx == 1:
            self.current_pool = concatenate_datasets([self.easy, self.medium])
        else:
            self.current_pool = concatenate_datasets([
                self.easy, self.medium, self.hard,
                self.pure_gsm, self.pure_gsm, self.pure_gsm 
            ])
            logging.info(f"🔥 Stage 3 Aktif: GSM8K 3x Weighting Uygulandı. Hard Sorular Havuzda!")
        
    def __len__(self):
        return len(self.current_pool)

    def __getitem__(self, idx):
        return self.current_pool[idx % len(self.current_pool)]

# ============================================================================
# 3. TOKEN-BASED CONTROLLER (HASSAS HESAPLAMALI)
# ============================================================================
class TokenCurriculumController(TrainerCallback):
    def __init__(self, dataset, tracker):
        self.dataset = dataset
        self.tracker = tracker
        self.last_stage_idx = -1

    def on_step_end(self, args, state, control, **kwargs):
        current_seen_tokens = self.tracker.seen_tokens

        # Her 50 adımda bir durum raporu
        if state.global_step % 50 == 0 and state.global_step > 0:
            remaining = max(0, TOTAL_TOKENS_TARGET - current_seen_tokens)
            percent = (current_seen_tokens / TOTAL_TOKENS_TARGET) * 100
            logging.info(f"📊 DURUM: Görülen: {current_seen_tokens:,} | Kalan: {remaining:,} | İlerleme: %{percent:.2f}")

        if current_seen_tokens >= TOTAL_TOKENS_TARGET:
            control.should_training_stop = True
            logging.info(f"🛑 HEDEF (42M) TAMAMLANDI. Eğitim bitiriliyor.")

        # Stage ve LR Belirleme (Kümülatif limitlere göre)
        if current_seen_tokens < STAGE_1_LIMIT: 
            stage_idx, base_lr = 0, LR_STAGE_1
        elif current_seen_tokens < STAGE_2_LIMIT: 
            stage_idx, base_lr = 1, LR_STAGE_2
        else: 
            stage_idx, base_lr = 2, LR_STAGE_3

        if stage_idx != self.last_stage_idx:
            self.dataset.set_stage(stage_idx)
            logging.info(f"🔄 Stage {stage_idx+1} Başladı! | Hedeflenen LR: {base_lr}")
            self.last_stage_idx = stage_idx

        # Cosine Decay
        progress = min(1.0, (current_seen_tokens - WARMUP_TOKENS) / (TOTAL_TOKENS_TARGET - WARMUP_TOKENS))
        lr = MIN_LR + 0.5 * (base_lr - MIN_LR) * (1 + math.cos(math.pi * progress))
        for param_group in kwargs["optimizer"].param_groups:
            param_group["lr"] = lr

# ============================================================================
# 4. COLLATOR & MAIN
# ============================================================================
class StrictMathCollator:
    def __init__(self, tokenizer, tracker):
        self.tokenizer = tokenizer
        self.tracker = tracker

    def __call__(self, examples):
        batch_input_ids, batch_labels, batch_attention_mask = [], [], []
        for item in examples:
            full_text = item["question"] + item["answer"] + self.tokenizer.eos_token
            enc = self.tokenizer(full_text, truncation=True, max_length=320, padding=False)
            input_ids = enc["input_ids"]
            labels = [-100] * len(input_ids)
            q_enc = self.tokenizer(item["question"], truncation=True, max_length=320)
            q_len = len(q_enc["input_ids"])
            for i in range(q_len, len(input_ids)):
                labels[i] = input_ids[i]
            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))
            batch_attention_mask.append(torch.tensor(enc["attention_mask"]))
            
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        
        self.tracker.seen_tokens += attention_mask_padded.sum().item()
        return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded, "labels": labels_padded}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    gsm_ds = load_dataset("json", data_files=str(GSM_FILE), split="train")
    arith_ds = load_dataset("json", data_files=str(ARITHMETIC_FILE), split="train")

    def calc_len(x):
        return {"length": len(tokenizer(x["question"]+x["answer"], truncation=True, max_length=320)["input_ids"])}

    gsm_ds = gsm_ds.map(calc_len, num_proc=4)
    gsm_split = gsm_ds.train_test_split(test_size=1000, seed=42)
    train_gsm = gsm_split["train"]

    easy = train_gsm.filter(lambda x: x["length"] <= 120)
    medium = train_gsm.filter(lambda x: 120 < x["length"] <= 220)
    hard = train_gsm.filter(lambda x: x["length"] > 220)
    
    curriculum_ds = CurriculumDataset(easy, medium, hard, arith_ds, train_gsm)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    peft_config = LoraConfig(r=64, lora_alpha=128, target_modules=["c_attn", "c_proj", "c_fc"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=str(WORKSPACE / "uhem_final_2epoch"),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=100000,
        bf16=True,
        logging_steps=50,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=2000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=curriculum_ds,
        data_collator=StrictMathCollator(tokenizer, token_tracker),
        callbacks=[TokenCurriculumController(curriculum_ds, token_tracker)]
    )

    logging.info(f"🚀 SIFIRDAN EĞİTİM (2 EPOCH / 42M TOKEN) BAŞLIYOR.")
    trainer.train()
    trainer.save_model(str(WORKSPACE / "FINAL_2_EPOCH_SCRATCH"))

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()