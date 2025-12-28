"""
Pharmaceutical LLM API - Integration Module
Copy this file to your src/ directory and import in app.py
"""

import os
import json
import torch
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class PharmaceuticalLLMAPI:
    """
    API wrapper for pharmaceutical explanation generation
    
    Usage in Flask app:
        from llm_api import PharmaceuticalLLMAPI
        
        # Initialize once at startup
        explainer = PharmaceuticalLLMAPI()
        
        # Use in endpoint
        @app.route('/api/explain', methods=['POST'])
        def explain():
            data = request.json
            explanation = explainer.explain(data)
            return jsonify({'explanation': explanation})
    """
    
    def __init__(self, model_path: str = 'llm_finetuning/output/fine_tuned_model'):
        """
        Initialize LLM explainer
        
        Args:
            model_path: Path to fine-tuned model directory
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        print(f"[LLM API] Device: {self.device}")
        
        # Lazy loading - model loaded on first use
        self._model_loaded = False
    
    def _load_model(self):
        """Load model (lazy loading)"""
        if self._model_loaded:
            return
        
        print(f"[LLM API] Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            print(f"[LLM API] WARNING: Fine-tuned model not found at {self.model_path}")
            print("[LLM API] Using fallback explanations")
            self._model_loaded = True
            return
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load base model
            base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN")
            )
            
            if self.device == "cpu":
                base_model = base_model.to(self.device)
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
            
            print("[LLM API] Model loaded successfully!")
            self._model_loaded = True
            
        except Exception as e:
            print(f"[LLM API] ERROR loading model: {e}")
            print("[LLM API] Falling back to template explanations")
            self._model_loaded = True
    
    def explain(self, prediction_data: Dict) -> str:
        """
        Generate pharmaceutical explanation
        
        Args:
            prediction_data: Dict with keys:
                - category (str): Drug category (C1-C8)
                - prediction (float): Predicted sales value
                - week (int, optional): Week number
                - year (int, optional): Year
                - location (str, optional): Location name
        
        Returns:
            str: Detailed pharmaceutical explanation
        """
        # Load model if not already loaded
        self._load_model()
        
        # If model failed to load, use fallback
        if self.model is None or self.tokenizer is None:
            return self._fallback_explanation(prediction_data)
        
        try:
            # Generate with model
            prompt = self._build_prompt(prediction_data)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            
            return response.strip()
        
        except Exception as e:
            print(f"[LLM API] Error generating explanation: {e}")
            return self._fallback_explanation(prediction_data)
    
    def _build_prompt(self, data: Dict) -> str:
        """Build Llama 3.1 Instruct prompt"""
        
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
        year = data.get('year', 2024)
        location = data.get('location', 'Colombo')
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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
5. Recommendations for healthcare providers, government, and public awareness<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _fallback_explanation(self, data: Dict) -> str:
        """Fallback explanation when model not available"""
        
        category = data.get('category', 'Unknown')
        prediction = data.get('prediction', 0)
        week = data.get('week', 1)
        year = data.get('year', 2024)
        location = data.get('location', 'Colombo')
        
        fallback_templates = {
            'C1': f"""M01AB (Anti-inflammatory Acetic Acid Derivatives) - {prediction:.2f} units forecast

ANALYSIS:
This category includes NSAIDs like Diclofenac and Indomethacin. The {prediction:.2f} unit prediction for Week {week}, {year} in {location} reflects:

SEASONAL FACTORS:
• Monsoon season transitions increase humidity-related musculoskeletal complaints
• Temperature variations stress joints, increasing arthritis symptoms
• Historical patterns show NSAID demand spikes during rainy seasons

DEMOGRAPHIC PATTERNS:
• Aging urban population with higher arthritis prevalence
• Professional workers seeking rapid pain relief
• Healthcare accessibility in {location}

CLINICAL CONSIDERATIONS:
⚠️ Monitor for GI bleeding, cardiovascular events, renal impairment

RECOMMENDATIONS:
Healthcare: Ensure gastroprotection for high-risk patients
Government: Maintain adequate stock levels for seasonal surges
Public: Seek medical advice for chronic pain, report side effects""",
            
            'C3': f"""N02BA (Salicylic Acid Analgesics - Aspirin) - {prediction:.2f} units forecast

ANALYSIS:
Week {week}, {year} prediction for {location} considers:

CRITICAL CONTEXT:
⚠️ DENGUE SEASON: Aspirin contraindicated in suspected dengue due to bleeding risk
• Patients may substitute with other analgesics (paracetamol)
• Public health awareness affects utilization patterns

CARDIOVASCULAR USE:
• Low-dose aspirin for CV prevention in aging population
• Chronic therapy patients maintain stable demand

RECOMMENDATIONS:
Healthcare: Screen for dengue before prescribing aspirin
Public: Avoid aspirin if fever + rash/bleeding symptoms
Government: Ensure alternative analgesics available during outbreaks""",
            
            'C5': f"""N05B (Anxiolytics - Benzodiazepines) - {prediction:.2f} units forecast

ANALYSIS:
Mental health medication demand for Week {week}, {year} in {location}:

SOCIAL FACTORS:
• Reduced stigma around mental health treatment
• Urban stress and work pressure in {location}
• COVID-19 aftermath: increased anxiety prevalence

CLINICAL CONSIDERATIONS:
⚠️ Controlled substances - prescription required
• Risk of dependence, tolerance, withdrawal
• Short-term use recommended (<4 weeks)

RECOMMENDATIONS:
Healthcare: Assess for dependency, consider CBT alternatives
Regulators: Monitor prescription patterns for misuse
Public: Seek psychological support, not just medication"""
        }
        
        template = fallback_templates.get(category, f"""
Pharmaceutical Sales Prediction Analysis

Category: {category}
Predicted Sales: {prediction:.2f} units
Period: Week {week}, {year}
Location: {location}

This prediction reflects standard pharmaceutical utilization patterns for this drug category. 
Factors include seasonal variations, demographic trends, and local healthcare access.

For detailed analysis, please ensure the fine-tuned LLM model is properly loaded.
""")
        
        return template


def test_api():
    """Test the API"""
    print("Testing Pharmaceutical LLM API...")
    
    api = PharmaceuticalLLMAPI()
    
    test_data = {
        'category': 'C1',
        'prediction': 50.38,
        'week': 12,
        'year': 2018,
        'location': 'Colombo'
    }
    
    explanation = api.explain(test_data)
    print("\n" + "="*80)
    print("EXPLANATION:")
    print("="*80)
    print(explanation)


if __name__ == "__main__":
    test_api()
