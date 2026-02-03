import os
import sys
import logging
import json
import gc
import signal
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

# Try to import pynvml for GPU monitoring
PYNVML_AVAILABLE = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and dataset paths
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', str(Path.home())))
MODEL_NAME = os.getenv('MODEL_NAME', str(WORKSPACE / ''))
DATASET_DIR = WORKSPACE / 'data' / 'prepared' / 'tokenized_math_data'
TRAIN_DATASET_DIR = DATASET_DIR / 'train'
TEST_DATASET_DIR = DATASET_DIR / 'test'

# Output directory
JOB_ID = os.getenv('JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
OUTPUT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '1'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-4'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.1'))
MAX_GRAD_NORM = float(os.getenv('MAX_GRAD_NORM', '1.0'))
WARMUP_RATIO = float(os.getenv('WARMUP_RATIO', '0.1'))
LR_SCHEDULER_TYPE = os.getenv('LR_SCHEDULER_TYPE', 'cosine')

# Batch size and gradient accumulation
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', '8'))
AUTO_BATCH_SIZE = os.getenv('AUTO_BATCH_SIZE', 'false').lower() == 'true'

# Mixed precision
USE_FP16 = os.getenv('USE_FP16', 'false')
USE_BF16 = os.getenv('USE_BF16', 'true').lower() == 'true'

# Logging and checkpointing
LOGGING_STEPS = int(os.getenv('LOGGING_STEPS', '10'))
EVAL_STEPS = int(os.getenv('EVAL_STEPS', '50'))
EVAL_SUBSET_SIZE = int(os.getenv('EVAL_SUBSET_SIZE', '0'))
SAVE_STEPS = int(os.getenv('SAVE_STEPS', '50'))
SAVE_TOTAL_LIMIT = int(os.getenv('SAVE_TOTAL_LIMIT', '3'))
GPU_LOGGING_STEPS = int(os.getenv('GPU_LOGGING_STEPS', '50'))

# Performance optimization
GRADIENT_CHECKPOINTING = os.getenv('GRADIENT_CHECKPOINTING', 'false').lower() == 'true'
USE_TORCH_COMPILE = os.getenv('USE_TORCH_COMPILE', 'false').lower() == 'true'
DATALOADER_NUM_WORKERS = int(os.getenv('DATALOADER_NUM_WORKERS', '4'))
DATALOADER_PIN_MEMORY = os.getenv('DATALOADER_PIN_MEMORY', 'true').lower() == 'true'
OPTIM = os.getenv('OPTIM', 'adamw_torch_fused')

# Validate optimizer
if OPTIM == 'adamw_torch_fused':
    try:
        # Test if fused optimizer is available
        import torch.optim
        if not hasattr(torch.optim, 'AdamW') or not torch.cuda.is_available():
            logging.warning("adamw_torch_fused not available, falling back to adamw_torch")
            OPTIM = 'adamw_torch'
    except:
        logging.warning("adamw_torch_fused not available, falling back to adamw_torch")
        OPTIM = 'adamw_torch'

# Resume training
RESUME_FROM_CHECKPOINT = os.getenv('RESUME_FROM_CHECKPOINT', 'auto')

# Random seed
SEED = int(os.getenv('SEED', '42'))

print(f"Configuration loaded. Output dir: {OUTPUT_DIR}")
print(f"Train dataset: {TRAIN_DATASET_DIR}")
print(f"Test dataset: {TEST_DATASET_DIR}")

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

# Log all configuration
config_dict = {
    'model_name': MODEL_NAME,
    'train_dataset_dir': str(TRAIN_DATASET_DIR),
    'test_dataset_dir': str(TEST_DATASET_DIR),
    'output_dir': str(OUTPUT_DIR),
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'grad_accum_steps': GRAD_ACCUM_STEPS,
    'weight_decay': WEIGHT_DECAY,
    'warmup_ratio': WARMUP_RATIO,
    'lr_scheduler': LR_SCHEDULER_TYPE,
    'use_fp16': USE_FP16,
    'use_bf16': USE_BF16,
    'gradient_checkpointing': GRADIENT_CHECKPOINTING,
    'seed': SEED,
}

with open(OUTPUT_DIR / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

logging.info("Configuration saved to config.json")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_gpu_usage():
    """Log current GPU memory usage"""
    if not torch.cuda.is_available():
        logging.info("No GPU available")
        return
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        logging.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def get_optimal_batch_size():
    """Determine optimal batch size based on GPU availability"""
    if not AUTO_BATCH_SIZE:
        return BATCH_SIZE, GRAD_ACCUM_STEPS
    
    if not torch.cuda.is_available():
        logging.warning("No GPU available, using small batch size")
        return 1, 8
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logging.info(f"Total GPU memory: {total_memory:.2f} GB")
    
    if total_memory < 12:
        return 2, 8
    elif total_memory < 24:
        return 4, 4
    else:
        return 8, 2

class GPUUsageCallback(TrainerCallback):
    """Callback to log GPU usage during training"""
    
    def __init__(self, logging_steps=50):
        self.logging_steps = logging_steps
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_enabled = True
            except:
                self.nvml_enabled = False
        else:
            self.nvml_enabled = False
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.logging_steps == 0:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    
                    if self.nvml_enabled:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            logging.info(
                                f"Step {state.global_step} | GPU {i}: "
                                f"{allocated:.2f}GB alloc, {reserved:.2f}GB reserved, "
                                f"{util.gpu}% util, {temp}°C"
                            )
                        except:
                            logging.info(
                                f"Step {state.global_step} | GPU {i}: "
                                f"{allocated:.2f}GB alloc, {reserved:.2f}GB reserved"
                            )
                    else:
                        logging.info(
                            f"Step {state.global_step} | GPU {i}: "
                            f"{allocated:.2f}GB alloc, {reserved:.2f}GB reserved"
                        )

class SignalHandler:
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    
    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = output_dir
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        if not self.interrupted:
            self.interrupted = True
            logging.warning(f"Received signal {signum}. Saving checkpoint and exiting...")
            try:
                checkpoint_path = self.output_dir / "interrupted_checkpoint"
                self.trainer.save_model(str(checkpoint_path))
                logging.info(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
            sys.exit(0)

class CausalLMDataCollator(DataCollatorForLanguageModeling):
    """
    Wraps the standard collator to fix label masking when pad_token == eos_token.
    Only padding positions (beyond the original sequence length) are masked to -100,
    preserving legitimate EOS tokens as valid prediction targets.
    """
    def __call__(self, features):
        # Record original lengths before padding
        original_lengths = [len(f["input_ids"]) for f in features]

        # Let the parent class pad and create labels
        batch = super().__call__(features)

        # Re-mask labels: only mask positions that are padding (beyond original length)
        labels = batch["labels"]
        for i, length in enumerate(original_lengths):
            labels[i, length:] = -100  # mask only the padded tail

        batch["labels"] = labels
        return batch

print("Helper functions defined")

# ============================================================================
# MAIN TRAINING
# ============================================================================

try:
    # Check if datasets exist
    if not TRAIN_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Train dataset not found at {TRAIN_DATASET_DIR}. "
            "Please run prepare_dataset_with_split.py first."
        )
    
    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {TEST_DATASET_DIR}. "
            "Please run prepare_dataset_with_split.py first."
        )
    
    # Load metadata
    metadata_path = DATASET_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logging.info(f"Dataset metadata: {json.dumps(metadata, indent=2)}")
    
    # Load pre-tokenized datasets
    logging.info(f"Loading pre-tokenized train dataset from {TRAIN_DATASET_DIR}...")
    train_dataset = load_from_disk(str(TRAIN_DATASET_DIR))
    logging.info(f"Train dataset loaded: {len(train_dataset)} samples")
    
    logging.info(f"Loading pre-tokenized test dataset from {TEST_DATASET_DIR}...")
    eval_dataset = load_from_disk(str(TEST_DATASET_DIR))
    logging.info(f"Test dataset loaded: {len(eval_dataset)} samples")

    # Use subset for evaluation during training if specified
    if EVAL_SUBSET_SIZE > 0 and EVAL_SUBSET_SIZE < len(eval_dataset):
        full_eval_dataset = eval_dataset  # Keep reference to full dataset
        eval_dataset = eval_dataset.select(range(EVAL_SUBSET_SIZE))
        logging.info(f"Using eval subset for training: {len(eval_dataset)} samples (full dataset has {len(full_eval_dataset)} samples)")
    else:
        full_eval_dataset = None
        logging.info(f"Using full eval dataset: {len(eval_dataset)} samples")
    
    # Verify dataset format
    required_columns = ['input_ids', 'attention_mask']
    for col in required_columns:
        if col not in train_dataset.column_names:
            raise ValueError(f"Train dataset missing required column: {col}")
        if col not in eval_dataset.column_names:
            raise ValueError(f"Test dataset missing required column: {col}")
    
    logging.info("Dataset validation successful")
    
    # Load tokenizer (needed for data collator and saving)
    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check for existing checkpoints
    should_resume = False
    should_overwrite = False
    checkpoint_path = None
    
    if RESUME_FROM_CHECKPOINT == 'auto':
        checkpoints = sorted(OUTPUT_DIR.glob('checkpoint-*'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            logging.info(f"Found checkpoint: {checkpoint_path}")
            should_resume = True
        else:
            logging.info("No checkpoints found. Starting fresh training...")
            should_overwrite = True
    elif RESUME_FROM_CHECKPOINT and RESUME_FROM_CHECKPOINT != 'false':
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT)
        if checkpoint_path.exists():
            should_resume = True
        else:
            logging.warning(f"Checkpoint not found: {checkpoint_path}")
            should_overwrite = True
    else:
        logging.info("No checkpoints found. Starting fresh training...")
        should_resume = False
        should_overwrite = True
    
    # Load model
    logging.info("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Enable gradient checkpointing if requested
    if GRADIENT_CHECKPOINTING:
        logging.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Compile model if requested (PyTorch 2.0+)
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        logging.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    data_collator = CausalLMDataCollator(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Determine batch size
    batch_size, grad_accum = get_optimal_batch_size()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        effective_batch = batch_size * grad_accum * num_gpus
        logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum × {num_gpus} GPUs = {effective_batch} effective")
    else:
        effective_batch = batch_size * grad_accum
        logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum (CPU) = {effective_batch} effective")
    
    # Determine mixed precision settings
    use_fp16 = False
    use_bf16 = False
    
    if USE_FP16 == 'auto':
        use_fp16 = torch.cuda.is_available() and not USE_BF16
    elif USE_FP16 == 'true':
        use_fp16 = True
    
    if USE_BF16 and torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_bf16 = True
            use_fp16 = False
            logging.info("Using BF16 mixed precision training")
        else:
            logging.warning("BF16 requested but not supported on this GPU, falling back to FP16")
            use_fp16 = True
    
    # Calculate warmup steps
    total_steps = (len(train_dataset) // (batch_size * grad_accum * max(1, torch.cuda.device_count()))) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    logging.info(f"Total steps: {total_steps} | Warmup: {warmup_steps} ({WARMUP_RATIO*100:.1f}%)")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        disable_tqdm=False,
        optim=OPTIM,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
    )
    
    # Create GPU usage callback
    gpu_callback = GPUUsageCallback(logging_steps=GPU_LOGGING_STEPS)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[gpu_callback],
    )
    
    # Setup signal handler
    handler = SignalHandler(trainer, OUTPUT_DIR)

    logging.info("Starting training...")
    log_gpu_usage()
    
    # Clear memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train
    if should_resume and checkpoint_path:
        trainer.train(resume_from_checkpoint=str(checkpoint_path))
    else:
        trainer.train()
    
    logging.info("Training completed. Saving final model...")
    final_model_path = OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Evaluate on full dataset if we used a subset during training
    if full_eval_dataset is not None:
        logging.info(f"Running final evaluation on full dataset ({len(full_eval_dataset)} samples)...")
        trainer.eval_dataset = full_eval_dataset
        eval_results = trainer.evaluate()
    else:
        eval_results = trainer.evaluate()

    with open(OUTPUT_DIR / 'final_eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logging.info(f"Model saved to {final_model_path}")
    logging.info(f"Final evaluation results: {eval_results}")
    logging.info("All done!")
    
except FileNotFoundError as e:
    logging.error(str(e))
    logging.error("Please run prepare_dataset_with_split.py before training.")
    raise
    
except torch.cuda.OutOfMemoryError:
    logging.error("GPU out of memory!")
    logging.error("Try:")
    logging.error("  1. Reduce BATCH_SIZE")
    logging.error("  2. Increase GRAD_ACCUM_STEPS")
    logging.error("  3. Enable GRADIENT_CHECKPOINTING=true")
    logging.error("  4. Reduce DATALOADER_NUM_WORKERS")
    
    if 'trainer' in locals():
        emergency_path = OUTPUT_DIR / "emergency_checkpoint"
        logging.info(f"Saving emergency checkpoint to {emergency_path}")
        try:
            trainer.save_model(str(emergency_path))
        except:
            logging.error("Could not save emergency checkpoint")
    raise
    
except KeyboardInterrupt:
    logging.warning("Training interrupted by user")
    if 'trainer' in locals():
        interrupt_path = OUTPUT_DIR / "interrupted_checkpoint"
        logging.info(f"Saving interrupted checkpoint to {interrupt_path}")
        trainer.save_model(str(interrupt_path))
    raise
    
except Exception as e:
    logging.error(f"Training failed: {e}", exc_info=True)
    if 'trainer' in locals():
        error_path = OUTPUT_DIR / "error_checkpoint"
        logging.info(f"Attempting to save checkpoint to {error_path}")
        try:
            trainer.save_model(str(error_path))
        except:
            logging.error("Could not save error checkpoint")
    raise

finally:
    # Cleanup NVML if initialized
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except:
            pass