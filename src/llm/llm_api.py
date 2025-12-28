"""
LLM API Integration for Pharmaceutical Sales Explanation
Connects fine-tuned LLaMA model with Flask backend
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
from typing import Dict, Optional
import numpy as np

class PharmaceuticalLLMExplainer:
    """Generate explanations for drug sales predictions using fine-tuned LLM"""
    
    def __init__(self, model_path: str = "models_llm/pharmaceutical_llama"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        print(f"🤖 LLM Explainer initialized (device: {self.device})")
    
    def load_model(self):
        """Lazy load the fine-tuned model"""
        if self.loaded:
            return
        
        print(f"📥 Loading fine-tuned model from {self.model_path}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.loaded = True
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Could not load fine-tuned model: {e}")
            print("   Using fallback rule-based explanations")
            self.loaded = False
    
    def build_context(
        self,
        category: str,
        predicted_value: float,
        date: str,
        model_used: str,
        historical_data: Optional[Dict] = None,
        causal_data: Optional[Dict] = None
    ) -> str:
        """Build rich context from prediction data"""
        
        category_names = {
            'C1': 'M01AB - Anti-inflammatory Acetic acid derivatives',
            'C2': 'M01AE - Anti-inflammatory Propionic acid derivatives',
            'C3': 'N02BA - Analgesics Salicylic acid',
            'C4': 'N02BE - Analgesics Pyrazolones',
            'C5': 'N05B - Anxiolytics',
            'C6': 'N05C - Hypnotics and sedatives',
            'C7': 'R03 - Drugs for obstructive airway diseases',
            'C8': 'R06 - Antihistamines'
        }
        
        context = f"""Drug Sales Prediction Analysis:

Category: {category} ({category_names.get(category, 'Unknown')})
Predicted Sales: {predicted_value:.2f} units
Date: {date}
Model Used: {model_used}
"""
        
        # Add historical context if available
        if historical_data:
            context += f"""
Historical Context:
- Previous week: {historical_data.get('previous_value', 'N/A')} units
- Change: {historical_data.get('change_pct', 0):+.1f}%
- Trend: {historical_data.get('trend', 'stable')}
- Season: {historical_data.get('season', 'unknown')}
"""
        
        # Add causal insights if available
        if causal_data and 'causal_factors' in causal_data:
            context += f"""
Causal Analysis:
"""
            for factor, data in list(causal_data['causal_factors'].items())[:3]:
                if isinstance(data, dict) and 'effect' in data:
                    context += f"- {factor}: effect={data['effect']:.3f}, p-value={data.get('significance', 'N/A')}\n"
        
        context += "\nQuestion: Why is this drug category showing this sales pattern in this area and week? What should people and government be aware of?"
        
        return context
    
    def generate_explanation(
        self,
        category: str,
        predicted_value: float,
        date: str,
        model_used: str,
        historical_data: Optional[Dict] = None,
        causal_data: Optional[Dict] = None,
        max_length: int = 1024
    ) -> str:
        """Generate explanation using fine-tuned LLM"""
        
        # Load model if not already loaded
        if not self.loaded:
            self.load_model()
        
        # If model failed to load, use fallback
        if not self.loaded:
            return self._fallback_explanation(category, predicted_value, date, historical_data)
        
        # Build context
        context = self.build_context(
            category, predicted_value, date, model_used,
            historical_data, causal_data
        )
        
        # Format prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a pharmaceutical analytics expert specializing in Sri Lankan healthcare. Analyze the drug sales prediction data and explain the reasons for the sales pattern. Provide insights for pharmacists, government officials, and public health awareness.<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"⚠️  LLM generation error: {e}")
            return self._fallback_explanation(category, predicted_value, date, historical_data)
    
    def _fallback_explanation(
        self,
        category: str,
        predicted_value: float,
        date: str,
        historical_data: Optional[Dict] = None
    ) -> str:
        """Fallback rule-based explanation when LLM is unavailable"""
        
        explanations = {
            'C1': f"""**Analysis of C1 (Anti-inflammatory) Sales Pattern**

**Primary Factors:**
1. **Seasonal Musculoskeletal Impact**: High humidity during this period exacerbates arthritis and joint pain in Sri Lanka's aging population (15% over 60 years).

2. **Weather-Related Inflammation**: Monsoon conditions trigger increased musculoskeletal complaints, particularly in urban areas like Colombo with high pollution.

3. **Healthcare Access**: Improved access to pharmacies and government health campaigns promoting self-care for minor ailments.

**🏥 Public Health Implications:**
- Monitor for potential overuse or misuse of anti-inflammatory medications
- Ensure adequate supply chain to prevent shortages during peak demand
- Track adverse events, especially GI bleeding in elderly populations

**🏛️ Government Recommendations:**
- Maintain 20% buffer stock in national pharmacies
- Launch public awareness on proper NSAID usage and risks
- Deploy mobile health units to underserved rural areas

**💊 Pharmacy Management:**
- Adjust inventory: Stock {predicted_value * 1.2:.0f} units (20% safety margin)
- Prepare related products (topical pain relief, gastro-protective agents)
- Train staff on patient consultation for OTC anti-inflammatory selection

**⚠️ Risk Awareness:**
- Side effects: GI bleeding, renal impairment, cardiovascular events
- Contraindications: Active peptic ulcer, severe heart failure, third trimester pregnancy
- Drug interactions: Anticoagulants, antihypertensives, other NSAIDs
""",
            'C2': f"""**Analysis of C2 (Ibuprofen-class) Sales Pattern**

**Primary Factors:**
1. **High OTC Accessibility**: Most commonly used pain reliever globally, readily available without prescription
2. **Fever Management**: Preferred antipyretic during flu season and viral infections
3. **Self-Medication Culture**: Strong preference for self-treatment of minor ailments in Sri Lankan population

**Public Health Insights:**
- Generally safer GI profile than C1 class
- Education needed on proper dosing and duration
- Risk of overuse due to easy accessibility

**Government Actions:**
- Ensure quality control of imported generics
- Public education campaigns on responsible use
- Monitor adverse event reports

**Pharmacy Recommendations:**
- Stock: {predicted_value * 1.15:.0f} units
- Cross-sell: Paracetamol as alternative
- Counsel patients on maximum daily dose
""",
            'C5': f"""**Analysis of C5 (Anxiolytics) Sales Pattern**

**Primary Factors:**
1. **Mental Health Awareness Growth**: Reduced stigma and increased treatment-seeking behavior
2. **Post-Pandemic Impact**: COVID-19 aftermath continues to affect mental health
3. **Economic Stressors**: Financial pressures in current economic climate

**Public Health Implications:**
- Rising mental health crisis requires systemic response
- Balance between access and preventing misuse
- Need for non-pharmacological interventions (therapy, support groups)

**Government Priorities:**
- Strengthen mental health services
- Ensure controlled substance monitoring
- Launch mental health awareness campaigns

**Pharmacy Requirements:**
- Controlled substance documentation mandatory
- Patient counseling on dependence risk
- Coordinate with prescribing physicians
""",
            'C7': f"""**Analysis of C7 (Respiratory Drugs) Sales Pattern**

**Primary Factors:**
1. **Air Quality Issues**: Urban pollution and seasonal dust affect respiratory health
2. **Asthma Prevalence**: 5-8% of population, higher in urban areas
3. **Seasonal Allergens**: Pollen and mold spores increase during specific periods

**Public Health Context:**
- Growing respiratory disease burden
- Climate change impact on air quality
- Pediatric asthma cases increasing

**Government Actions:**
- Air quality monitoring and public alerts
- Ensure availability of emergency inhalers
- School-based asthma management programs

**Pharmacy Guidance:**
- Proper inhaler technique counseling
- Emergency action plans for patients
- Stock: {predicted_value * 1.25:.0f} units (higher safety margin)
"""
        }
        
        default = f"""**Sales Pattern Analysis for {category}**

Predicted sales of {predicted_value:.2f} units on {date} reflect multiple factors including seasonal patterns, disease prevalence, and healthcare access improvements in Sri Lanka.

**Key Considerations:**
- Regular monitoring of stock levels
- Patient education on proper medication use
- Coordination with healthcare providers
- Awareness of potential side effects and interactions

**Recommendations:**
- Maintain adequate inventory
- Ensure quality control
- Provide patient counseling
- Monitor for adverse events
"""
        
        return explanations.get(category, default)


# Global instance for API
_llm_explainer = None

def get_llm_explainer() -> PharmaceuticalLLMExplainer:
    """Get or create global LLM explainer instance"""
    global _llm_explainer
    if _llm_explainer is None:
        _llm_explainer = PharmaceuticalLLMExplainer()
    return _llm_explainer


# Flask API helper function
def explain_prediction(
    category: str,
    forecast_value: float,
    date: str,
    model_used: str,
    historical_context: Optional[Dict] = None,
    causal_context: Optional[Dict] = None
) -> Dict:
    """
    Generate explanation for a prediction
    Returns dict with explanation text and metadata
    """
    
    explainer = get_llm_explainer()
    
    explanation = explainer.generate_explanation(
        category=category,
        predicted_value=forecast_value,
        date=date,
        model_used=model_used,
        historical_data=historical_context,
        causal_data=causal_context
    )
    
    return {
        'explanation': explanation,
        'category': category,
        'predicted_value': forecast_value,
        'date': date,
        'model_used': model_used,
        'llm_status': 'fine-tuned' if explainer.loaded else 'fallback'
    }


if __name__ == "__main__":
    # Test the explainer
    print("🧪 Testing LLM Explainer...")
    
    explainer = PharmaceuticalLLMExplainer()
    
    # Test prediction
    result = explain_prediction(
        category='C1',
        forecast_value=50.38,
        date='2025-12-15',
        model_used='Ensemble',
        historical_context={
            'previous_value': 43.76,
            'change_pct': 15.1,
            'trend': 'increasing',
            'season': 'monsoon'
        }
    )
    
    print("\n" + "="*80)
    print("EXPLANATION:")
    print("="*80)
    print(result['explanation'])
    print("\n" + "="*80)
    print(f"Status: {result['llm_status']}")
