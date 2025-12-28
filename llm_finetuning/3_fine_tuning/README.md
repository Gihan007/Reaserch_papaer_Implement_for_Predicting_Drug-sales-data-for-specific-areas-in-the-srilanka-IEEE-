# Step 3: Fine-Tuning LLM

## 🎯 Purpose
Train Llama 3.1 8B on pharmaceutical data to create custom AI explainer.

## ⚠️ IMPORTANT: GPU Required

**Minimum Requirements:**
- GPU: RTX 3090, RTX 4090, or A100 (16GB+ VRAM)
- RAM: 32GB system memory
- Storage: 40GB free space
- Time: 2-4 hours training

**No GPU?** Skip to Step 4 for alternatives (GPT-4, pre-trained models)

## ⚡ Quick Run

```bash
python train_llm.py
```

**What Happens:**
1. Loads Llama 3.1 8B Instruct from HuggingFace (15GB download)
2. Applies QLoRA (4-bit quantization + LoRA adapters)
3. Trains on 400+ pharmaceutical examples + 40+ medical docs
4. Saves fine-tuned model to `../output/fine_tuned_model/`

## 📊 Training Configuration

### Default Settings (config.py)
```python
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "../output/fine_tuned_model"

# QLoRA Configuration
LOAD_IN_4BIT = True              # 4-bit quantization (saves 75% VRAM)
LORA_R = 16                      # LoRA rank
LORA_ALPHA = 32                  # LoRA scaling
LORA_DROPOUT = 0.05              # Regularization

# Training Hyperparameters
BATCH_SIZE = 4                   # Per device
GRADIENT_ACCUMULATION = 4        # Effective batch = 16
LEARNING_RATE = 2e-4             # AdamW
EPOCHS = 3                       # Usually sufficient
MAX_LENGTH = 2048                # Token context window
WARMUP_STEPS = 100               # LR warmup
```

### Adjust for Your GPU

| GPU | VRAM | Batch Size | Gradient Accum | Training Time |
|-----|------|------------|----------------|---------------|
| RTX 3090 | 24GB | 2 | 8 | 4 hours |
| RTX 4090 | 24GB | 4 | 4 | 2-3 hours |
| A100 | 40GB | 8 | 2 | 1-2 hours |
| RTX 3060 | 12GB | 1 | 16 | 6+ hours |

Edit `config.py` line 25-26:
```python
BATCH_SIZE = 1                   # Lower for smaller GPUs
GRADIENT_ACCUMULATION = 16        # Increase to compensate
```

## 🔍 Training Process

### Step-by-Step Breakdown

**Phase 1: Model Loading (5-10 min)**
```
Downloading Meta-Llama-3.1-8B-Instruct...
✓ Downloaded 15.2 GB
✓ Loading in 4-bit precision (3.8 GB VRAM)
✓ Applying LoRA adapters (r=16, alpha=32)
```

**Phase 2: Data Preparation (1 min)**
```
Loading training data...
✓ 400 sales examples from ../output/training_data/
✓ 40 medical documents from ../output/medical_docs/
✓ Total: 440 training examples
✓ Tokenizing (max_length=2048)...
```

**Phase 3: Training (2-4 hours)**
```
Epoch 1/3:
  Step 50/330  | Loss: 1.432 | LR: 1.5e-4 | 12 min
  Step 100/330 | Loss: 0.876 | LR: 2.0e-4 | 24 min
  Step 150/330 | Loss: 0.654 | LR: 2.0e-4 | 36 min
  ...

Epoch 3/3:
  Step 300/330 | Loss: 0.234 | LR: 3.2e-5 | 98 min
  Step 330/330 | Loss: 0.198 | LR: 0.0    | 102 min

✅ Training complete! Loss dropped 87% (1.43 → 0.19)
```

**Phase 4: Saving (2 min)**
```
Saving fine-tuned model...
✓ LoRA adapters → ../output/fine_tuned_model/adapter_model.bin
✓ Configuration → ../output/fine_tuned_model/adapter_config.json
✓ Tokenizer → ../output/fine_tuned_model/tokenizer_config.json
✓ Total size: 62 MB (adapters only!)
```

## 📁 Output Structure

```
../output/fine_tuned_model/
├── adapter_model.bin           # LoRA weights (62 MB)
├── adapter_config.json         # LoRA configuration
├── tokenizer_config.json       # Tokenizer settings
├── special_tokens_map.json     # Token mappings
├── training_args.bin           # Training configuration
└── training_log.json           # Loss curves, metrics
```

**Note:** Base Llama model not saved (still needs 15GB). Fine-tuned model merges adapters with base at inference time.

## ✅ Verify Training Success

### Check Loss Curve
```python
import json
with open('../output/fine_tuned_model/training_log.json') as f:
    log = json.load(f)
    print(f"Final loss: {log['train_loss'][-1]:.3f}")
    print(f"Best loss: {min(log['train_loss']):.3f}")
```

**Good Training Signs:**
- ✅ Loss decreases steadily
- ✅ Final loss < 0.3
- ✅ No sudden spikes (instability)

### Quick Test Inference
```bash
python test_model.py
```

Input: `"Why is M01AB (C1) selling 50 units in Week 12 of 2018 in Colombo?"`

Expected Output:
```
M01AB (Anti-inflammatory Acetic Acid) shows increased demand due to:

1. SEASONAL FACTORS: Week 12 coincides with monsoon season transition, 
   causing humidity-related musculoskeletal complaints.

2. DEMOGRAPHIC: Colombo's aging population (65+ years) experiences higher 
   arthritis prevalence, driving NSAID utilization.

3. PUBLIC HEALTH: Local dengue outbreak in early 2018 increased awareness 
   of pain management, though aspirin must be avoided in suspected cases.

RECOMMENDATIONS:
- Healthcare: Monitor for NSAID-related GI bleeding, ensure gastroprotection
- Government: Stock management for seasonal surges
- Public: Educate on proper use, contraindications
```

## 🐛 Troubleshooting

### "CUDA out of memory"
**Solution 1:** Reduce batch size
```python
# config.py
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
```

**Solution 2:** Use gradient checkpointing
```python
# train_llm.py line 80
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Add this
    ...
)
```

**Solution 3:** Reduce max length
```python
# config.py
MAX_LENGTH = 1024  # Instead of 2048
```

### "Loss not decreasing"
**Causes:**
- Learning rate too high/low
- Not enough training data
- Data quality issues

**Solution:**
```python
# config.py - Try lower learning rate
LEARNING_RATE = 1e-4  # Instead of 2e-4
EPOCHS = 5            # Train longer
```

### "Training too slow"
**Solution:** Enable Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

```python
# train_llm.py line 45
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Add this
    ...
)
```

### "No HuggingFace access token"
Llama 3.1 requires approval:

1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Request Access"
3. Get token: https://huggingface.co/settings/tokens
4. Login:
```bash
huggingface-cli login
# Paste your token
```

## 🔧 Advanced Configuration

### Enable Experiment Tracking (Weights & Biases)
```bash
pip install wandb
wandb login
```

```python
# config.py
USE_WANDB = True
WANDB_PROJECT = "pharmaceutical-llm"
```

### Multi-GPU Training
```bash
# Use 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_llm.py
```

### Resume from Checkpoint
```python
# train_llm.py line 90
trainer = Trainer(
    resume_from_checkpoint="../output/fine_tuned_model/checkpoint-200"
)
```

## 💡 Optimization Tips

### Faster Training
1. **Flash Attention 2** → 2x speed
2. **BF16 precision** (A100 only) → 1.5x speed
3. **Gradient checkpointing** → Fit larger batches

### Better Results
1. **More data** → Add local hospital reports, WHO updates
2. **Longer training** → 5 epochs instead of 3
3. **Larger LoRA rank** → r=32 instead of r=16 (needs more VRAM)

### Smaller Model Size
Current: 62 MB adapters  
Alternative: Use **QLoRA rank 8** → 31 MB adapters
```python
# config.py
LORA_R = 8
LORA_ALPHA = 16
```

## ✅ Success Criteria

Training complete when:
- ✅ 3 epochs finished without OOM errors
- ✅ Training loss < 0.3
- ✅ Model file saved (62 MB)
- ✅ Test inference produces coherent pharmaceutical explanations

## ✅ Next Step

Once training completes successfully:
```bash
cd ../4_inference
python test_explainer.py
```

Read `../4_inference/README.md` for testing your fine-tuned model!

---

**Model trained? ✅ Move to Step 4: Test Inference**

## 🚫 Skip Fine-Tuning (Alternative Approaches)

### Option A: Use Pre-trained Llama 3.1 (No fine-tuning)
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
# Use training examples as few-shot prompts
```

### Option B: Use OpenAI GPT-4
```python
import openai
# Use ../output/training_data/ as few-shot examples
# Cost: ~$0.03 per explanation
```

### Option C: Use Smaller Model (Phi-3 Mini)
```python
# Only 3.8B parameters, runs on 8GB GPU
model = "microsoft/Phi-3-mini-4k-instruct"
```

See `../4_inference/README.md` for implementation details!
