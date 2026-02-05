#!/bin/bash
#SBATCH -J "GRPO-Online"         # İşin adı
#SBATCH -A idm001                # Proje adı
#SBATCH -p gpu2dq                # Kuyruk adı
#SBATCH -n 64                    # İşlemci sayısı
#SBATCH -N 1                     # Bilgisayar sayısı
#SBATCH --gres=gpu:1             # 1 GPU istiyoruz

# Miniconda ve Ortam
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate turkish-gpt2-training

# ============================================================================
# CONFIGURATION (AYARLAR)
# ============================================================================

export TRAINING_WORKSPACE="$HOME/Cosmos-Proje"
export JOB_ID="${SLURM_JOB_ID}"

# TODO: SFT Modelinin Yolu. Model yolunu güncelle
export MODEL_NAME="$HOME/Cosmos-Proje/jobs/SFT_IS_ID/output/final_model" 

# TODO: Dataset. Dataset yolu güncelle
export DATASET_NAME="online_grpo_dataset.jsonl" 

# GRPO Üretim Ayarları 
export NUM_GENERATIONS="8"       # Her soru için kaç farklı cevap denesin? (VRAM'e göre 4'e düşürülebilir)
export MAX_PROMPT_LENGTH="512"
export MAX_COMPLETION_LENGTH="512" # Modelin cevabı ne kadar uzun olabilir?

# Eğitim Ayarları
export LEARNING_RATE="1e-5"      
export BETA="0.04"               # KL Penalty katsayısı 
export BATCH_SIZE="1"            # Online'da genelde 1 olur, generation hafıza yer
export GRAD_ACCUM_STEPS="8"
export WEIGHT_DECAY="0.1"

# Donanım
export USE_BF16="true"

# Loglama ve Kayıt
export LOGGING_STEPS="5"
export SAVE_STEPS="50"
export SAVE_TOTAL_LIMIT="2"
export GPU_LOGGING_STEPS="20"

# LoRA Ayarları
export LORA_R="64"
export LORA_ALPHA="128"
export LORA_DROPOUT="0.05"

export SEED="42"

# ============================================================================
# RUN (ÇALIŞTIRMA)
# ============================================================================

echo "=========================================="
echo "GRPO Online Job ${SLURM_JOB_ID}"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Generations per step: ${NUM_GENERATIONS}"
echo "Output: ${TRAINING_WORKSPACE}/jobs/${SLURM_JOB_ID}/output"
echo "=========================================="

python train_grpo_online.py

echo "=========================================="
echo "Training job completed at $(date)"
echo "=========================================="