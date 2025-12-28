"""
Step 4: Test Fine-Tuned Pharmaceutical Explainer
Run from project root: python llm_finetuning/4_inference/test_explainer.py
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class PharmaceuticalLLMExplainer:
    """Test pharmaceutical explanation generation"""
    
    def __init__(self, model_path: str = "../output/fine_tuned_model"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned model"""
        print(f"\n📥 Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                "Run Step 3 (fine-tuning) first!"
            )
        
        # Load base model
        base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"   Loading base model: {base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )
        
        # Load LoRA adapters
        print("   Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()  # Merge for faster inference
        
        print("✅ Model loaded successfully!")
    
    def generate_explanation(self, prediction_data: dict) -> str:
        """Generate pharmaceutical explanation"""
        
        # Build prompt
        prompt = self.build_prompt(prediction_data)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        return response.strip()
    
    def build_prompt(self, data: dict) -> str:
        """Build prompt from prediction data"""
        
        category_map = {
            'C1': 'M01AB (Anti-inflammatory Acetic acid derivatives)',
            'C2': 'M01AE (Anti-inflammatory Propionic acid derivatives)',
            'C3': 'N02BA (Analgesics, Salicylic acid)',
            'C4': 'N02BE (Analgesics, Pyrazolones)',
            'C5': 'N05B (Anxiolytics)',
            'C6': 'N05C (Hypnotics and sedatives)',
            'C7': 'R03 (Drugs for obstructive airway diseases)',
            'C8': 'R06 (Antihistamines for systemic use)'
        }
        
        category = data.get('category', 'C1')
        atc_code = category_map.get(category, category)
        prediction = data.get('prediction', 0)
        week = data.get('week', 1)
        year = data.get('year', 2018)
        location = data.get('location', 'Colombo')
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert pharmaceutical analyst specializing in drug utilization patterns in Sri Lanka. Provide detailed, evidence-based explanations for medication sales predictions, considering clinical, environmental, and public health factors.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze this pharmaceutical sales prediction:

Drug Category: {atc_code}
Predicted Sales: {prediction:.2f} units
Time Period: Week {week}, {year}
Location: {location}

Provide a comprehensive explanation covering:
1. Seasonal and environmental factors
2. Demographic and socioeconomic patterns
3. Public health context relevant to Sri Lanka
4. Clinical considerations and safety
5. Recommendations for healthcare providers, government, and public awareness

Focus on pharmaceutical expertise and Sri Lankan healthcare context.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt


def run_test_scenarios():
    """Test with various scenarios"""
    print("\n" + "="*80)
    print("🧪 TESTING PHARMACEUTICAL EXPLAINER")
    print("="*80)
    
    # Initialize explainer
    explainer = PharmaceuticalLLMExplainer()
    
    # Test scenarios
    scenarios = [
        {
            "name": "High NSAID Demand (Monsoon Season)",
            "data": {
                "category": "C1",
                "prediction": 50.38,
                "week": 12,
                "year": 2018,
                "location": "Colombo"
            }
        },
        {
            "name": "Aspirin Sales (Dengue Season)",
            "data": {
                "category": "C3",
                "prediction": 78.62,
                "week": 24,
                "year": 2018,
                "location": "Galle"
            }
        },
        {
            "name": "Anxiolytic Usage (Urban Area)",
            "data": {
                "category": "C5",
                "prediction": 32.15,
                "week": 8,
                "year": 2019,
                "location": "Colombo"
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"📊 SCENARIO {i}: {scenario['name']}")
        print(f"{'='*80}")
        print(f"Input: {json.dumps(scenario['data'], indent=2)}")
        print("\n🤖 Generating explanation...\n")
        
        explanation = explainer.generate_explanation(scenario['data'])
        
        print(f"{'─'*80}")
        print(explanation)
        print(f"{'─'*80}")
        
        results.append({
            "scenario": scenario['name'],
            "input": scenario['data'],
            "explanation": explanation
        })
    
    return results


def save_results(results: list, output_dir: str = "../output/inference_tests"):
    """Save test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "test_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save markdown
    md_path = os.path.join(output_dir, "sample_explanations.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Pharmaceutical Explanation Examples\n\n")
        f.write(f"Generated from fine-tuned Llama 3.1 8B model\n\n")
        f.write("---\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"## Scenario {i}: {result['scenario']}\n\n")
            f.write(f"**Input:**\n```json\n{json.dumps(result['input'], indent=2)}\n```\n\n")
            f.write(f"**Explanation:**\n\n{result['explanation']}\n\n")
            f.write("---\n\n")
    
    print(f"\n✅ Results saved:")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")


def main():
    print("🚀 Step 4: Testing Fine-Tuned Pharmaceutical Explainer")
    
    # Run tests
    results = run_test_scenarios()
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("✅ STEP 4 COMPLETE!")
    print("="*80)
    print("\n✅ Generated explanations for 3 test scenarios!")
    print("\nNext step:")
    print("  cd ../5_integration")
    print("  Read README.md for Flask API integration")


if __name__ == "__main__":
    main()
