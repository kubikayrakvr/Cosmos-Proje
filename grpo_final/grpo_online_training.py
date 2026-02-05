import os
import sys
import re
import logging
import signal
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# GPU İzleme 
PYNVML_AVAILABLE = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# 1. AYARLAR 
# ============================================================================

WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', str(Path.home())))
MODEL_NAME = os.getenv('MODEL_NAME')
DATASET_NAME = os.getenv('DATASET_NAME', 'dataset.jsonl')

JOB_ID = os.getenv('JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
OUTPUT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GRPO Hyperparameters
NUM_GENERATIONS = int(os.getenv('NUM_GENERATIONS', '4'))
MAX_PROMPT_LENGTH = int(os.getenv('MAX_PROMPT_LENGTH', '512'))
MAX_COMPLETION_LENGTH = int(os.getenv('MAX_COMPLETION_LENGTH', '512'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-5'))
BETA = float(os.getenv('BETA', '0.04'))

# Batch & Steps
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', '8'))
USE_BF16 = os.getenv('USE_BF16', 'true').lower() == 'true'

# Logging
LOGGING_STEPS = int(os.getenv('LOGGING_STEPS', '10'))
SAVE_STEPS = int(os.getenv('SAVE_STEPS', '100'))
SAVE_TOTAL_LIMIT = int(os.getenv('SAVE_TOTAL_LIMIT', '2'))

# LoRA
LORA_R = int(os.getenv('LORA_R', '64'))
LORA_ALPHA = int(os.getenv('LORA_ALPHA', '128'))
LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', '0.05'))

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)

# ============================================================================
# 2. ÖDÜL FONKSİYONLARI 
# ============================================================================

def extract_answer_value(text):
    """
    Modelin cevabından sayıyı çeker.
    Model '### Answer:' sonrası konuşacak. Biz son üretilen sayıyı veya
    varsa 'Cevap: X' formatını arayacağız.
    """
    if "### Answer:" in text:
        return text.split("### Answer:")[-1].strip()
    return text

def parse_number(text):
    """Metindeki sayıyı float'a çevirir (kesirleri de tanır)."""
    if not text: return None
    try:
        matches = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
        if matches:
            last_num = matches[-1]
            if "/" in last_num:
                n, d = last_num.split("/")
                return float(n) / float(d)
            return float(last_num)
    except:
        pass
    return None

# Ödül 1: Format (Sayı içeriyor mu?)
def format_reward_func(completions, **kwargs):
    """
    Modelin boş cevap verip vermediğini kontrol eder.
    SFT formatını zaten prompt ile zorladığımız için tag aramaya gerek yok.
    """
    rewards = []
    for text in completions:
        # Boş değilse ve sayı varsa ödül ver
        if len(text.strip()) > 0 and any(char.isdigit() for char in text):
            rewards.append(0.5) 
        else:
            rewards.append(0.0)
    return rewards

# Ödül 2: Reasoning (Uzun açıklama yapıyor mu?)
def reasoning_reward_func(completions, **kwargs):
    rewards = []
    for text in completions:
        gen_part = extract_answer_value(text)
        if len(gen_part) > 50: 
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

# Ödül 3: Doğruluk (Matematiksel Eşitlik)
def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        gen_text = extract_answer_value(completion)
        gen_val = parse_number(gen_text)
        
        # Ground Truth bazen liste içinde gelebilir, düzeltelim
        if isinstance(ground_truth, list): ground_truth = ground_truth[0]
        truth_val = parse_number(str(ground_truth))
        
        if gen_val is not None and truth_val is not None:
            if abs(gen_val - truth_val) < 1e-4:
                rewards.append(2.0) # Tam puan
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)     
    return rewards

# ============================================================================
# 3. VERİ SETİ & FORMATLAMA
# ============================================================================
def prepare_dataset():
    full_path = WORKSPACE / DATASET_NAME
    logging.info(f"Dataset yükleniyor: {full_path}")
    
    try:
        dataset = load_dataset("json", data_files=str(full_path), split="train")
    except Exception as e:
        logging.error(f"Dataset yüklenemedi: {e}")
        sys.exit(1)

    # %5 Test ayır
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def format_prompt(example):
        return {
            "prompt": f"### Question:\n{example['prompt']}\n\n### Answer:\n"
        }

    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)
    
    logging.info(f"Train Size: {len(train_dataset)}, Eval Size: {len(eval_dataset)}")
    return train_dataset, eval_dataset

# ============================================================================
# 4. SIGNAL HANDLER 
# ============================================================================
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
            logging.warning(f" Sinyal {signum} alındı. Model kaydediliyor...")
            try:
                self.trainer.save_model(str(self.output_dir / "interrupted_checkpoint"))
            except Exception as e:
                logging.error(f"Kayıt başarısız: {e}")
            sys.exit(0)

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
try:
    logging.info("=" * 70)
    logging.info("STARTING ONLINE GRPO TRAINING")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info("=" * 70)

    # 1. Tokenizer & Model
    logging.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Generation için LEFT padding şart!

    logging.info("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BF16 and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )

    # 2. Dataset
    train_dataset, eval_dataset = prepare_dataset()

    # 3. LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc"],
        bias="none",
    )
    # GRPO Trainer PEFT config'i kendi içinde halleder, modeli sarmalamaya gerek yok.

    # 4. GRPO Config
    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        
        # GRPO Özel Ayarları
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        fp16=(not USE_BF16),
        bf16=USE_BF16,
        report_to="none"
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, reasoning_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config
    )
    
    # Sinyal dinleyiciyi başlat
    handler = SignalHandler(trainer, OUTPUT_DIR)

    # 6. Başlat
    logging.info(f" Eğitim Başlıyor... (Num Generations: {NUM_GENERATIONS})")
    trainer.train()

    # 7. Kaydet
    final_path = OUTPUT_DIR / "final_online_model"
    logging.info(f"Model kaydediliyor: {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    logging.info(" ONLINE GRPO EĞİTİMİ TAMAMLANDI!")

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