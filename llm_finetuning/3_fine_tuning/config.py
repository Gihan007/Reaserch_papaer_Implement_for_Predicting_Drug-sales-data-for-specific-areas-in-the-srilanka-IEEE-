"""
Step 3: LLM Fine-Tuning Configuration
Edit these settings to match your hardware
"""

# Model Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Smaller 3.8B model, faster download
OUTPUT_DIR = "../output/fine_tuned_model"

# Data Paths
TRAINING_DATA_PATH = "../output/training_data/sales_explanations.jsonl"
MEDICAL_DOCS_PATH = "../output/medical_docs/processed_documents.jsonl"

# QLoRA Configuration (4-bit quantization)
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

# LoRA Configuration
LORA_R = 16                      # Rank (8, 16, 32, 64) - Higher = more params
LORA_ALPHA = 32                  # Scaling factor (usually 2*r)
LORA_DROPOUT = 0.05              # Dropout for regularization
LORA_TARGET_MODULES = [          # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Hyperparameters (Optimized for RTX 4060 8GB)
BATCH_SIZE = 1                   # RTX 4060 8GB - Start with 1
GRADIENT_ACCUMULATION = 16       # Effective batch = 16 (1 * 16)
LEARNING_RATE = 2e-4             # AdamW learning rate
EPOCHS = 3                       # Number of training epochs
MAX_LENGTH = 1024                # Reduced for 8GB VRAM
WARMUP_STEPS = 100               # Learning rate warmup

# Optimization
GRADIENT_CHECKPOINTING = True    # Saves VRAM at cost of speed
OPTIM = "paged_adamw_32bit"     # Optimizer (memory efficient)
LR_SCHEDULER = "cosine"          # Learning rate schedule

# Logging & Saving
LOGGING_STEPS = 10               # Log every N steps
SAVE_STEPS = 100                 # Save checkpoint every N steps
SAVE_TOTAL_LIMIT = 2             # Keep only last 2 checkpoints

# Experiment Tracking (Optional - requires: pip install wandb)
USE_WANDB = False                # Set True to track with Weights & Biases
WANDB_PROJECT = "pharmaceutical-llm"
WANDB_RUN_NAME = "llama3.1-8b-pharma-finetuning"

# Hardware-Specific Adjustments
"""
RTX 3090 (24GB):
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 8
    MAX_LENGTH = 2048
    
RTX 4090 (24GB):
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    MAX_LENGTH = 2048
    
A100 (40GB):
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 2
    MAX_LENGTH = 2048
    
RTX 3060 (12GB):
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    MAX_LENGTH = 1024
"""
