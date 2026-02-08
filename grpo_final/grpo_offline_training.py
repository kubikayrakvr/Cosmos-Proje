import os
import sys
import logging
import json
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# GPU İzleme (Varsa)
PYNVML_AVAILABLE = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# 1. AYARLAR (ENV VARIABLES'DAN OKUMA)
# ============================================================================

# Yolları Bash dosyasından alıyoruz
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', str(Path.home())))
MODEL_NAME = os.getenv('MODEL_NAME') # SFT Modeli
DATASET_NAME = os.getenv('DATASET_NAME', 'grpo_offline.jsonl')

# Çıktı Klasörü
JOB_ID = os.getenv('JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
OUTPUT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparametreler
MAX_SEQ_LENGTH = int(os.getenv('MAX_SEQ_LENGTH', '1024'))
NUM_EPOCHS = float(os.getenv('NUM_EPOCHS', '1'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-5'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', '4'))
USE_BF16 = os.getenv('USE_BF16', 'true').lower() == 'true'
SEED = int(os.getenv('SEED', '42'))

# Loglama
LOGGING_STEPS = int(os.getenv('LOGGING_STEPS', '10'))
SAVE_STEPS = int(os.getenv('SAVE_STEPS', '100'))
SAVE_TOTAL_LIMIT = int(os.getenv('SAVE_TOTAL_LIMIT', '2'))
GPU_LOGGING_STEPS = int(os.getenv('GPU_LOGGING_STEPS', '50'))

# LoRA
LORA_R = int(os.getenv('LORA_R', '64'))
LORA_ALPHA = int(os.getenv('LORA_ALPHA', '128'))
LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', '0.05'))

# ============================================================================
# LOGGING SETUP (DOSYAYA YAZMA)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'training.log'), # Log dosyası
        logging.StreamHandler() # Ekrana basma
    ]
)

# ============================================================================
# 2. DATASET 
# ============================================================================
class OfflineGRPODataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        logging.info(f" Veri seti işleniyor: {data_path}")
        
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        if "responses" not in item: continue 
                        
                        prompt = item["prompt"]
                        responses = item["responses"] 
                        
                        # --- GRUP İÇİ AVANTAJ HESABI ---
                        scores = [r["score"] for r in responses]
                        mean_score = np.mean(scores)
                        std_score = np.std(scores) + 1e-8 
                        
                        for resp, score in zip(responses, scores):
                            # Avantaj: (Puan - Ort) / Std
                            advantage = (score - mean_score) / std_score
                            
                            self.data.append({
                                "prompt": prompt,
                                "completion": resp['text'], 
                                "advantage": float(advantage)
                            })
                    except json.JSONDecodeError:
                        logging.warning(" JSON okuma hatası, satır atlandı.")
                        continue
        except FileNotFoundError:
            logging.error(f" Dataset bulunamadı: {data_path}")
            sys.exit(1)
                    
        logging.info(f" Toplam Örnek: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# 3. COLLATOR 
# ============================================================================
class GRPODataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_template = "### Cevap:\n"
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, 
            add_special_tokens=False
        )

    def __call__(self, batch):
        prompts = [x['prompt'] for x in batch]
        completions = [x['completion'] for x in batch]
        advantages = [x['advantage'] for x in batch]

        full_texts = [f"### Soru:\n{p}\n\n{self.response_template}{c}" for p, c in zip(prompts, completions)]
        
        tokenized = self.tokenizer(
            full_texts,
            padding=True,          
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True 
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone() 

        # --- MASKELEME ---
        for i in range(len(input_ids)):
            response_start_idx = -1
            len_template = len(self.response_token_ids)
            
            for j in range(len(input_ids[i]) - len_template):
                if input_ids[i][j : j + len_template].tolist() == self.response_token_ids:
                    response_start_idx = j + len_template 
                    break
            
            if response_start_idx != -1:
                labels[i, :response_start_idx] = -100 
            else:
                labels[i, :] = -100 

            labels[i][attention_mask[i] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": torch.tensor(advantages, dtype=torch.float32)
        }

# ============================================================================
# 4. TRAINER 
# ============================================================================
class OfflineGRPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        advantages = inputs.get("advantage")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.get("logits")
        
        # --- LOSS HESABI ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(reduction='none') 
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.size())
        
        valid_mask = (shift_labels != -100).float()
        sentence_loss = (token_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        
        advantages = advantages.to(sentence_loss.device)

        # Safety Clamping 
        clamped_advantages = torch.clamp(advantages, min=-2.0, max=2.0)

        # GRPO: Loss = Hata * Avantaj
        weighted_loss = sentence_loss * clamped_advantages
        final_loss = weighted_loss.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss

# ============================================================================
# YARDIMCI ARAÇLAR (GPU LOGLAMA & SIGNAL HANDLER)
# ============================================================================
class GPUUsageCallback:
    def __init__(self, logging_steps=50):
        self.logging_steps = logging_steps
        self.nvml_enabled = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_enabled = True
            except: pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0 and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logging.info(f"Step {state.global_step} | GPU {i}: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")

class SignalHandler:
    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = output_dir
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        if not self.interrupted:
            self.interrupted = True
            logging.warning(f" Sinyal alındı ({signum}). Checkpoint alınıyor ve çıkılıyor...")
            try:
                self.trainer.save_model(str(self.output_dir / "interrupted_checkpoint"))
            except Exception as e:
                logging.error(f" Kaydetme hatası: {e}")
            sys.exit(0)

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
try:
    logging.info("=" * 70)
    logging.info(f"STARTING OFFLINE GRPO TRAINING")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info("=" * 70)

    # 1. Tokenizer & Model
    logging.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logging.info("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BF16 and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    # 2. LoRA Setup
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc"], 
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Dataset & Collator
    full_dataset_path = WORKSPACE / DATASET_NAME
    train_dataset = OfflineGRPODataset(full_dataset_path)
    collator = GRPODataCollator(tokenizer, max_length=MAX_SEQ_LENGTH)

    # 4. Training Arguments 
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        fp16=(not USE_BF16),
        bf16=USE_BF16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = OfflineGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )

    # Callbacks
    trainer.add_callback(GPUUsageCallback(logging_steps=GPU_LOGGING_STEPS))
    handler = SignalHandler(trainer, OUTPUT_DIR)

    # 5. Start
    logging.info(" Eğitim Başlıyor...")
    trainer.train()

    # 6. Save
    logging.info("Kaydediliyor...")
    final_path = OUTPUT_DIR / "final_grpo_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    logging.info(f" Bitti! Model şuraya kaydedildi: {final_path}")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        logging.error("CUDA OOM! GPU belleği yetersiz.")
        logging.error("Çözüm: NUM_GENERATIONS, MAX_LENGTH veya batch küçült.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(137)
    else:
        logging.error(f" RuntimeError: {e}", exc_info=True)
        raise

except Exception as e:
    logging.error(f" KRİTİK HATA: {e}", exc_info=True)
    raise