"""
LLM Fine-Tuning Pipeline for Pharmaceutical Sales Explanation
Fine-tunes Llama 3.1 8B on pharmaceutical domain data
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from typing import Dict, List

class PharmaceuticalLLMFineTuner:
    """Fine-tune LLaMA for pharmaceutical sales explanation generation"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir: str = "models_llm/pharmaceutical_llama"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.tokenizer = None
        self.model = None
    
    def load_datasets(self) -> Dict[str, Dataset]:
        """Load training datasets from JSON files"""
        print("\n📂 Loading datasets...")
        
        datasets = {}
        
        # Load sales explanation dataset
        sales_data_path = 'llm_training_data/pharmaceutical_sales_explanations.jsonl'
        if os.path.exists(sales_data_path):
            datasets['sales'] = load_dataset('json', data_files=sales_data_path, split='train')
            print(f"   ✓ Sales explanations: {len(datasets['sales'])} examples")
        else:
            print(f"   ⚠️  Sales data not found: {sales_data_path}")
            print(f"   Run: python src/llm/data_preparation.py first")
        
        # Load medical documents
        docs_data_path = 'llm_training_data/medical_documents/processed_documents.jsonl'
        if os.path.exists(docs_data_path):
            datasets['documents'] = load_dataset('json', data_files=docs_data_path, split='train')
            print(f"   ✓ Medical documents: {len(datasets['documents'])} examples")
        else:
            print(f"   ⚠️  Medical documents not found: {docs_data_path}")
            print(f"   Run: python src/llm/medical_document_scraper.py first")
        
        if not datasets:
            raise ValueError("No training data found! Run data preparation scripts first.")
        
        # Combine all datasets
        from datasets import concatenate_datasets
        combined = concatenate_datasets(list(datasets.values()))
        
        # Shuffle and split
        combined = combined.shuffle(seed=42)
        split = combined.train_test_split(test_size=0.1)
        
        print(f"\n📊 Dataset split:")
        print(f"   Training: {len(split['train'])} examples")
        print(f"   Validation: {len(split['test'])} examples")
        
        return split
    
    def load_model(self, use_4bit: bool = True):
        """Load model with quantization for efficient fine-tuning"""
        print(f"\n🤖 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for 4-bit training (QLoRA)
        if use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            print("   ✓ Loaded with 4-bit quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            print("   ✓ Loaded in full precision")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def format_prompt(self, example: Dict) -> str:
        """Format example into prompt template"""
        
        # Llama 3.1 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        
        return prompt
    
    def preprocess_function(self, examples):
        """Tokenize examples for training"""
        
        # Format prompts
        prompts = [self.format_prompt(ex) for ex in examples]
        
        # Tokenize
        model_inputs = self.tokenizer(
            prompts,
            max_length=2048,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    def train(self, dataset: Dict, num_epochs: int = 3, batch_size: int = 4):
        """Fine-tune the model"""
        print(f"\n🚀 Starting fine-tuning...")
        
        # Preprocess datasets
        print("   Tokenizing datasets...")
        tokenized_train = dataset['train'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        tokenized_val = dataset['test'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['test'].column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=self.device == "cuda",
            optim="paged_adamw_8bit" if self.device == "cuda" else "adamw_torch",
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
            run_name="pharmaceutical-llama-ft"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator
        )
        
        # Train
        print("\n   Training started...")
        trainer.train()
        
        # Save final model
        print(f"\n💾 Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n✅ Fine-tuning complete!")
    
    def test_inference(self, prompt: str):
        """Test the fine-tuned model"""
        print("\n🧪 Testing inference...")
        print(f"\n📝 Prompt:\n{prompt}\n")
        
        # Format prompt
        formatted = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        print(f"🤖 Response:\n{response}")
        
        return response


def main():
    """Main execution pipeline"""
    
    print("🏥 Pharmaceutical LLM Fine-Tuning Pipeline")
    print("="*80)
    
    # Check if training data exists
    sales_data = 'llm_training_data/pharmaceutical_sales_explanations.jsonl'
    docs_data = 'llm_training_data/medical_documents/processed_documents.jsonl'
    
    if not os.path.exists(sales_data):
        print("\n⚠️  Sales explanation data not found!")
        print("Run: python src/llm/data_preparation.py")
        return
    
    if not os.path.exists(docs_data):
        print("\n⚠️  Medical documents not found!")
        print("Run: python src/llm/medical_document_scraper.py")
        return
    
    # Initialize fine-tuner
    fine_tuner = PharmaceuticalLLMFineTuner(
        model_name="meta-llama/Llama-3.1-8B-Instruct",  # Or use "meta-llama/Llama-2-7b-chat-hf"
        output_dir="models_llm/pharmaceutical_llama"
    )
    
    # Load datasets
    dataset = fine_tuner.load_datasets()
    
    # Load model
    fine_tuner.load_model(use_4bit=True)
    
    # Train
    fine_tuner.train(
        dataset=dataset,
        num_epochs=3,
        batch_size=4
    )
    
    # Test inference
    test_prompt = """Category C1 (Anti-inflammatory) sales increased 15% in Colombo during December. 
What are the likely reasons and what should stakeholders be aware of?"""
    
    fine_tuner.test_inference(test_prompt)
    
    print("\n✅ Fine-tuning pipeline complete!")
    print(f"\nModel saved to: models_llm/pharmaceutical_llama")
    print("\nNext steps:")
    print("1. Integrate with Flask API: python src/llm/llm_api.py")
    print("2. Test with web interface")


if __name__ == "__main__":
    main()
