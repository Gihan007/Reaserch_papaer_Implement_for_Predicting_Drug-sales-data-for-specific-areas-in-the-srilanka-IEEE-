"""
Step 3: Fine-Tune LLM on Pharmaceutical Data
Run from project root: python llm_finetuning/3_fine_tuning/train_llm.py

IMPORTANT: Requires GPU with 16GB+ VRAM!
"""

import os
import sys
import json

# Disable TensorFlow to avoid import conflicts
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import config

class PharmaceuticalLLMTrainer:
    """Fine-tune Llama 3.1 8B for pharmaceutical explanations"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        if self.device == "cpu":
            print("⚠️  WARNING: No GPU detected! Fine-tuning will be very slow.")
            print("Consider using cloud GPU (Colab, Vast.ai, RunPod)")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    def load_model_and_tokenizer(self):
        """Load Llama 3.1 8B with QLoRA"""
        print("\n📥 Loading Llama 3.1 8B Instruct...")
        print(f"   Model: {config.MODEL_NAME}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")  # HuggingFace token
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print(f"✅ Model loaded in 4-bit ({self.model.get_memory_footprint() / 1e9:.2f} GB)")
    
    def configure_lora(self):
        """Apply LoRA adapters"""
        print("\n🔧 Configuring LoRA...")
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ LoRA configured:")
        print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"   Total params: {total_params:,}")
    
    def load_training_data(self):
        """Load and prepare training dataset"""
        print("\n📚 Loading training data...")
        
        # Load datasets
        datasets_to_load = []
        
        if os.path.exists(config.TRAINING_DATA_PATH):
            print(f"   ✓ Sales data: {config.TRAINING_DATA_PATH}")
            datasets_to_load.append(config.TRAINING_DATA_PATH)
        else:
            print(f"   ⚠️  Sales data not found: {config.TRAINING_DATA_PATH}")
        
        if os.path.exists(config.MEDICAL_DOCS_PATH):
            print(f"   ✓ Medical docs: {config.MEDICAL_DOCS_PATH}")
            datasets_to_load.append(config.MEDICAL_DOCS_PATH)
        else:
            print(f"   ⚠️  Medical docs not found: {config.MEDICAL_DOCS_PATH}")
        
        if not datasets_to_load:
            raise FileNotFoundError("No training data found! Run Steps 1 and 2 first.")
        
        # Combine datasets
        combined_data = []
        for path in datasets_to_load:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    combined_data.append(json.loads(line))
        
        print(f"✅ Loaded {len(combined_data)} training examples")
        
        # Format for Llama 3.1 Instruct
        formatted_data = []
        for example in combined_data:
            text = self.format_instruction(
                example['instruction'],
                example['input'],
                example['output']
            )
            formatted_data.append({"text": text})
        
        # Convert to HuggingFace Dataset
        self.dataset = load_dataset("json", data_files={"train": datasets_to_load})["train"]
        self.dataset = self.dataset.map(
            self.tokenize_function,
            remove_columns=self.dataset.column_names
        )
        
        print(f"✅ Dataset tokenized (max_length={config.MAX_LENGTH})")
    
    def format_instruction(self, instruction: str, input_text: str, output: str) -> str:
        """Format as Llama 3.1 Instruct prompt"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
    
    def tokenize_function(self, examples):
        """Tokenize examples"""
        # Format each example
        texts = []
        for i in range(len(examples['instruction'])):
            text = self.format_instruction(
                examples['instruction'][i],
                examples['input'][i],
                examples['output'][i]
            )
            texts.append(text)
        
        # Tokenize
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length"
        )
    
    def train(self):
        """Train the model"""
        print("\n🚀 Starting training...")
        print(f"   Epochs: {config.EPOCHS}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Gradient accumulation: {config.GRADIENT_ACCUMULATION}")
        print(f"   Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
        print(f"   Learning rate: {config.LEARNING_RATE}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
            learning_rate=config.LEARNING_RATE,
            fp16=True,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            save_total_limit=config.SAVE_TOTAL_LIMIT,
            warmup_steps=config.WARMUP_STEPS,
            optim=config.OPTIM,
            lr_scheduler_type=config.LR_SCHEDULER,
            gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
            report_to="wandb" if config.USE_WANDB else "none",
            run_name=config.WANDB_RUN_NAME if config.USE_WANDB else None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer
        )
        
        # Train!
        print("\n⏱️  Training started (this will take 2-4 hours)...\n")
        trainer.train()
        
        print("\n✅ Training complete!")
    
    def save_model(self):
        """Save fine-tuned model"""
        print(f"\n💾 Saving model to {config.OUTPUT_DIR}...")
        
        self.model.save_pretrained(config.OUTPUT_DIR)
        self.tokenizer.save_pretrained(config.OUTPUT_DIR)
        
        # Save training log
        log_path = os.path.join(config.OUTPUT_DIR, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                "model": config.MODEL_NAME,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "lora_r": config.LORA_R,
                "status": "completed"
            }, f, indent=2)
        
        print(f"✅ Model saved!")
        print(f"   Adapter size: {sum(os.path.getsize(os.path.join(config.OUTPUT_DIR, f)) for f in os.listdir(config.OUTPUT_DIR)) / 1e6:.1f} MB")


def main():
    print("="*80)
    print("🚀 STEP 3: FINE-TUNING LLM")
    print("="*80)
    
    # Check for HuggingFace token
    if not os.getenv("HF_TOKEN"):
        print("\n⚠️  HuggingFace token not found!")
        print("Llama 3.1 requires authentication.")
        print("\nSteps:")
        print("1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("2. Click 'Request Access' and wait for approval")
        print("3. Get token: https://huggingface.co/settings/tokens")
        print("4. Run: $env:HF_TOKEN='your_token_here'")
        print("\nOr run: huggingface-cli login")
        sys.exit(1)
    
    # Initialize trainer
    trainer = PharmaceuticalLLMTrainer()
    
    # Load model
    trainer.load_model_and_tokenizer()
    
    # Configure LoRA
    trainer.configure_lora()
    
    # Load data
    trainer.load_training_data()
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*80)
    print("✅ STEP 3 COMPLETE!")
    print("="*80)
    print(f"\n✅ Fine-tuned model saved to: {config.OUTPUT_DIR}")
    print("\nNext step:")
    print("  cd ../4_inference")
    print("  python test_explainer.py")
    print("\nOr read: ../4_inference/README.md")


if __name__ == "__main__":
    main()
