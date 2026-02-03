import os
import sys
import logging
import json
import gc
import signal
import shutil
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

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

# Workspace and paths
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', str(Path.home())))
MODEL_NAME = os.getenv('MODEL_NAME', 'final_unwrapped')
DATASET_NAME = os.getenv('DATASET_NAME', 'sft_ready_dataset.jsonl')  

# Output directory
JOB_ID = os.getenv('JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
OUTPUT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
TEST_SIZE = int(os.getenv('TEST_SIZE', '4000'))
MAX_SEQ_LENGTH = int(os.getenv('MAX_SEQ_LENGTH', '1024'))
DATASET_SEED = int(os.getenv('DATASET_SEED', '42'))

# Training hyperparameters
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '3'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '2e-5'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.01'))

# Batch size and gradient accumulation
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))
GRAD_ACCUM_STEPS = int(os.getenv('GRAD_ACCUM_STEPS', '8'))
AUTO_BATCH_SIZE = os.getenv('AUTO_BATCH_SIZE', 'false').lower() == 'true'

# Mixed precision
USE_BF16 = os.getenv('USE_BF16', 'true').lower() == 'true'

# Logging and checkpointing
LOGGING_STEPS = int(os.getenv('LOGGING_STEPS', '10'))
EVAL_STEPS = int(os.getenv('EVAL_STEPS', '500'))
SAVE_STEPS = int(os.getenv('SAVE_STEPS', '500'))
SAVE_TOTAL_LIMIT = int(os.getenv('SAVE_TOTAL_LIMIT', '2'))
GPU_LOGGING_STEPS = int(os.getenv('GPU_LOGGING_STEPS', '50'))

# LoRA configuration
LORA_R = int(os.getenv('LORA_R', '64'))
LORA_ALPHA = int(os.getenv('LORA_ALPHA', '128'))
LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', '0.1'))

# Resume training
RESUME_FROM_CHECKPOINT = os.getenv('RESUME_FROM_CHECKPOINT', 'auto')

# Random seed
SEED = int(os.getenv('SEED', '42'))

# SFT-specific settings
PACKING = os.getenv('PACKING', 'true').lower() == 'true'

print(f"Configuration loaded. Output dir: {OUTPUT_DIR}")
print(f"Dataset: {DATASET_NAME}")
print(f"Model: {MODEL_NAME}")

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
    'dataset_name': DATASET_NAME,
    'output_dir': str(OUTPUT_DIR),
    'test_size': TEST_SIZE,
    'max_seq_length': MAX_SEQ_LENGTH,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'grad_accum_steps': GRAD_ACCUM_STEPS,
    'weight_decay': WEIGHT_DECAY,
    'use_bf16': USE_BF16,
    'packing': PACKING,
    'lora_r': LORA_R,
    'lora_alpha': LORA_ALPHA,
    'lora_dropout': LORA_DROPOUT,
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

def validate_output_directory():
    """Validate that we can write to the output directory"""
    try:
        test_file = OUTPUT_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logging.info(f"✓ Output directory is writable: {OUTPUT_DIR}")
        return True
    except Exception as e:
        logging.error(f"✗ Cannot write to output directory: {e}")
        return False

def validate_checkpoint_save(trainer, tokenizer):
    """Validate that checkpoint saving works before training"""
    try:
        logging.info("Running pre-training checkpoint save validation...")
        test_checkpoint_dir = OUTPUT_DIR / "test_checkpoint"
        
        # Try to save model
        trainer.save_model(str(test_checkpoint_dir))
        logging.info("✓ Model save successful")
        
        # Try to save tokenizer
        tokenizer.save_pretrained(str(test_checkpoint_dir))
        logging.info("✓ Tokenizer save successful")
        
        # Verify files exist
        expected_files = ["config.json", "tokenizer_config.json"]
        # Check for either safetensors or bin files
        has_model_file = (test_checkpoint_dir / "model.safetensors").exists() or \
                        (test_checkpoint_dir / "pytorch_model.bin").exists() or \
                        (test_checkpoint_dir / "adapter_model.safetensors").exists() or \
                        (test_checkpoint_dir / "adapter_model.bin").exists()
        
        missing_files = [f for f in expected_files if not (test_checkpoint_dir / f).exists()]
        
        if missing_files:
            logging.warning(f"Some expected files not found: {missing_files}")
        
        if not has_model_file:
            logging.error("✗ No model weights file found in checkpoint")
            return False
        else:
            logging.info("✓ Model weights file found")
        
        # Verify we can load the checkpoint back
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(test_checkpoint_dir))
            logging.info("✓ Checkpoint config can be loaded")
        except Exception as e:
            logging.error(f"✗ Cannot load checkpoint config: {e}")
            return False
        
        # Cleanup test checkpoint
        shutil.rmtree(test_checkpoint_dir)
        logging.info("✓ Test checkpoint cleaned up")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Checkpoint save validation failed: {e}")
        logging.error("This means training checkpoints may fail to save!")
        return False

def validate_evaluation(trainer, eval_dataset):
    """Validate that evaluation works before training"""
    try:
        logging.info("Running pre-training evaluation validation...")
        
        # Run a quick evaluation on a small subset
        # Take min of 10 samples or dataset size
        max_eval_samples = min(10, len(eval_dataset))
        
        # Create a small subset for testing
        eval_subset = eval_dataset.select(range(max_eval_samples))
        
        # Temporarily replace the trainer's eval dataset
        original_eval_dataset = trainer.eval_dataset
        trainer.eval_dataset = eval_subset
        
        eval_results = trainer.evaluate()
        
        # Restore original eval dataset
        trainer.eval_dataset = original_eval_dataset
        
        # Check if we got valid results
        if "eval_loss" not in eval_results:
            logging.error("✗ Evaluation did not return 'eval_loss'")
            return False
        
        if not isinstance(eval_results["eval_loss"], (int, float)):
            logging.error(f"✗ Invalid eval_loss type: {type(eval_results['eval_loss'])}")
            return False
        
        if eval_results["eval_loss"] < 0:
            logging.error(f"✗ Negative eval_loss: {eval_results['eval_loss']}")
            return False
        
        logging.info(f"✓ Evaluation successful (test eval_loss: {eval_results['eval_loss']:.4f})")
        return True
        
    except Exception as e:
        logging.error(f"✗ Evaluation validation failed: {e}")
        logging.error("This means evaluation during training may fail!")
        logging.error(f"Error details: {str(e)}", exc_info=True)
        return False

class GPUUsageCallback:
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

print("Helper functions defined")

# ============================================================================
# MAIN TRAINING
# ============================================================================

try:
    # Check for dataset name
    if not DATASET_NAME:
        raise ValueError("DATASET_NAME environment variable must be set")
    
    # Validate output directory first
    logging.info("=" * 70)
    logging.info("STEP 1: PRE-TRAINING VALIDATION")
    logging.info("=" * 70)
    
    if not validate_output_directory():
        raise RuntimeError("Output directory validation failed")
    
    # Load tokenizer first
    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
    
    # Load model
    logging.info(f"Loading model: {MODEL_NAME}")

    # Set dtype based on configuration
    if USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logging.info("Using bfloat16 precision")
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
        logging.info("Using float16 precision")
    else:
        torch_dtype = torch.float32
        logging.info("Using float32 precision (CPU or GPU without mixed precision support)")

    # Load model with proper dtype and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",  # Automatically distribute model across available GPUs
        trust_remote_code=True,  # Needed for some models
    )

    # Only resize if actually needed
    if model.config.vocab_size != len(tokenizer):
        logging.warning(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        EMBEDDINGS_RESIZED = True
    else:
        EMBEDDINGS_RESIZED = False
    model.config.pad_token_id = tokenizer.pad_token_id

    logging.info(f"Model loaded with dtype={torch_dtype}, device_map=auto")
    log_gpu_usage()
    
    # Load dataset
    logging.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset("json", data_files=DATASET_NAME)
    
    # Handle dataset structure (dict or not)
    if isinstance(dataset, dict):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # Take the first split available
            dataset = list(dataset.values())[0]
    
    logging.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Filter long sequences
    def filter_long_sequences(example):
        # Get column names
        question_col = None
        answer_col = None
        
        for possible_name in ["question", "query", "instruction", "input"]:
            if possible_name in example:
                question_col = possible_name
                break
        
        for possible_name in ["answer", "response", "output", "target"]:
            if possible_name in example:
                answer_col = possible_name
                break
        
        if not question_col or not answer_col:
            return True  # Keep if we can't determine columns
        
        # Format text same as training
        formatted_text = f"### Soru:\n{example[question_col]}\n\n### Cevap:\n{example[answer_col]}" + tokenizer.eos_token
        # Measure token length
        return len(tokenizer(formatted_text)["input_ids"]) <= MAX_SEQ_LENGTH
    
    logging.info(f"Filtering sequences longer than {MAX_SEQ_LENGTH} tokens...")
    logging.info(f"Dataset size before filtering: {len(dataset)}")
    
    dataset = dataset.filter(filter_long_sequences, num_proc=4)
    
    logging.info(f"Dataset size after filtering: {len(dataset)}")
    
    # Split dataset
    logging.info(f"Splitting dataset (test_size={TEST_SIZE})...")
    dataset_dict = dataset.train_test_split(test_size=TEST_SIZE, seed=DATASET_SEED)
    
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    logging.info(f"Training Samples: {len(train_dataset)}")
    logging.info(f"Validation Samples: {len(eval_dataset)}")
    
    # Detect column names
    question_col = ""
    answer_col = ""
    
    for possible_name in ["question", "query", "instruction", "input"]:
        if possible_name in train_dataset.column_names:
            question_col = possible_name
            break
    
    for possible_name in ["answer", "response", "output", "target"]:
        if possible_name in train_dataset.column_names:
            answer_col = possible_name
            break
    
    logging.info(f"Detected Question Column: '{question_col}'")
    logging.info(f"Detected Answer Column:   '{answer_col}'")
    
    if not question_col or not answer_col:
        raise ValueError("Could not automatically find question/answer columns!")
    
    # Define formatting function
    def formatting_prompts_func(example):
        output_texts = []
        for q, a in zip(example[question_col], example[answer_col]):
            formatted_text = f"### Soru:\n{q}\n\n### Cevap:\n{a}" + tokenizer.eos_token
            output_texts.append(formatted_text)
        return output_texts
    
    # Determine batch size
    batch_size, grad_accum = get_optimal_batch_size()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        effective_batch = batch_size * grad_accum * num_gpus
        logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum × {num_gpus} GPUs = {effective_batch} effective")
    else:
        effective_batch = batch_size * grad_accum
        logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum (CPU) = {effective_batch} effective")
    
    # Configure LoRA
    logging.info("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"] if EMBEDDINGS_RESIZED else None
    )
    
    # Configure SFT training
    logging.info("Configuring SFT training...")
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        packing=PACKING,
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        bf16=USE_BF16,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=WEIGHT_DECAY,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        disable_tqdm=False,
    )
    
    logging.info("Creating SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Register GPU monitoring callback
    logging.info("Registering GPU monitoring callback...")
    gpu_callback = GPUUsageCallback(logging_steps=GPU_LOGGING_STEPS)
    trainer.add_callback(gpu_callback)

    # ========================================================================
    # PRE-TRAINING VALIDATION
    # ========================================================================
    logging.info("=" * 70)
    logging.info("STEP 2: VALIDATING SAVE AND EVAL FUNCTIONALITY")
    logging.info("=" * 70)
    
    validation_passed = True
    
    # Test checkpoint saving
    if not validate_checkpoint_save(trainer, tokenizer):
        validation_passed = False
        logging.error("Checkpoint save validation FAILED!")
    
    # Test evaluation
    if not validate_evaluation(trainer, eval_dataset):
        validation_passed = False
        logging.error("Evaluation validation FAILED!")
    
    if not validation_passed:
        logging.error("=" * 70)
        logging.error("PRE-TRAINING VALIDATION FAILED!")
        logging.error("Please fix the issues above before starting training.")
        logging.error("=" * 70)
        raise RuntimeError("Pre-training validation failed. Training aborted.")
    
    logging.info("=" * 70)
    logging.info("✓ ALL PRE-TRAINING VALIDATIONS PASSED!")
    logging.info("=" * 70)
    
    # ========================================================================
    # START TRAINING
    # ========================================================================
    
    # Setup signal handler
    handler = SignalHandler(trainer, OUTPUT_DIR)
    
    # Check for existing checkpoints
    should_resume = False
    checkpoint_path = None
    
    if RESUME_FROM_CHECKPOINT == 'auto':
        checkpoints = sorted(OUTPUT_DIR.glob('checkpoint-*'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            logging.info(f"Found checkpoint: {checkpoint_path}")
            should_resume = True
        else:
            logging.info("No checkpoints found. Starting fresh training...")
    elif RESUME_FROM_CHECKPOINT and RESUME_FROM_CHECKPOINT != 'false':
        checkpoint_path = Path(RESUME_FROM_CHECKPOINT)
        if checkpoint_path.exists():
            should_resume = True
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            logging.warning(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info("=" * 70)
    logging.info("STEP 3: STARTING TRAINING")
    logging.info("=" * 70)
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
    
    # Final evaluation
    logging.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    with open(OUTPUT_DIR / 'final_eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logging.info(f"Model saved to {final_model_path}")
    logging.info(f"Final evaluation results: {eval_results}")
    logging.info("=" * 70)
    logging.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
    logging.info("=" * 70)
    
except ValueError as e:
    logging.error(str(e))
    raise
    
except torch.cuda.OutOfMemoryError:
    logging.error("GPU out of memory!")
    logging.error("Try:")
    logging.error("  1. Reduce BATCH_SIZE")
    logging.error("  2. Increase GRAD_ACCUM_STEPS")
    logging.error("  3. Reduce MAX_SEQ_LENGTH")
    logging.error("  4. Disable PACKING")
    
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