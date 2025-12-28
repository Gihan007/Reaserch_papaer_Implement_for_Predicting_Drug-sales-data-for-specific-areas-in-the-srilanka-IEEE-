"""
LLM Fine-Tuning Quick Start Guide
Complete pipeline to train pharmaceutical sales explanation model
"""

# 🚀 PHARMACEUTICAL LLM FINE-TUNING PIPELINE
# ============================================

## STEP 0: Prerequisites

### Hardware Requirements:
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100)
- RAM: 32GB+ system memory
- Storage: 50GB+ free space

### Software Requirements:
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git


## STEP 1: Install Dependencies

```bash
# Install LLM-specific requirements
pip install -r requirements-llm.txt

# If on Windows, install PyTorch with CUDA separately:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## STEP 2: Prepare Training Data from Your CSV Files

```bash
# This script converts your C1.csv - C8.csv into LLM training format
python src/llm/data_preparation.py
```

**What this does:**
- Loads all 8 category CSV files (C1-C8)
- Analyzes sales patterns, trends, seasonal variations
- Generates 400+ training examples (50 per category)
- Creates instruction-input-output format for fine-tuning
- Saves to: `llm_training_data/pharmaceutical_sales_explanations.json`

**Output example:**
```json
{
  "instruction": "You are a pharmaceutical expert...",
  "input": "Category C1 sales: 50.38 units, +15% increase, monsoon season",
  "output": "Analysis: Anti-inflammatory sales rising due to humidity..."
}
```


## STEP 3: Collect Medical Domain Documents

```bash
# This script gathers pharmaceutical knowledge base
python src/llm/medical_document_scraper.py
```

**What this does:**
- WHO ATC classification guidelines for all 8 categories
- FDA drug labeling information
- PubMed article templates
- Sri Lankan healthcare context
- Saves to: `llm_training_data/medical_documents/`

**Total documents: 40-50** covering:
- Drug mechanisms, side effects, contraindications
- Seasonal patterns in Sri Lanka
- Disease epidemiology
- Healthcare system context


## STEP 4: Fine-Tune Llama 3.1 8B Model

```bash
# This is the main fine-tuning script
python src/llm/fine_tune_llm.py
```

**What this does:**
- Downloads Llama 3.1 8B Instruct from Hugging Face
- Applies 4-bit quantization (QLoRA) for efficient training
- Fine-tunes on your pharmaceutical data
- Trains for 3 epochs (~2-4 hours on RTX 3090)
- Saves fine-tuned model to: `models_llm/pharmaceutical_llama/`

**Training configuration:**
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Batch size: 4 (adjustable based on GPU memory)
- Learning rate: 2e-4
- LoRA rank: 16
- Target modules: q_proj, k_proj, v_proj, o_proj

**Expected GPU memory:** 12-16GB


## STEP 5: Test the Fine-Tuned Model

```bash
# Test inference with fine-tuned model
python src/llm/llm_api.py
```

**Example interaction:**
```
Input: "C1 sales increased 15% in December. Why?"

Output: "Anti-inflammatory drug (C1) sales are rising due to:
1. Monsoon season humidity (85-90%) triggering arthritis flare-ups
2. Aging population (15% over 60) in urban areas
3. Government health campaigns promoting self-medication
Government should ensure 20% buffer stock and monitor for shortages..."
```


## STEP 6: Integrate with Flask API

Add to `app.py`:

```python
from src.llm.llm_api import explain_prediction

@app.route('/api/explain', methods=['POST'])
def api_explain():
    """Generate LLM explanation for prediction"""
    try:
        data = request.get_json()
        
        result = explain_prediction(
            category=data['category'],
            forecast_value=data['forecast_value'],
            date=data['date'],
            model_used=data.get('model_used', 'Ensemble'),
            historical_context=data.get('historical_context'),
            causal_context=data.get('causal_context')
        )
        
        return jsonify({
            'success': True,
            'explanation': result['explanation'],
            'llm_status': result['llm_status']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```


## ALTERNATIVE: Use Pre-trained Models (No Fine-Tuning)

If you don't have GPU or want faster setup:

### Option A: Use OpenAI GPT-4 API
```python
# src/llm/openai_explainer.py
import openai

openai.api_key = "your-api-key"

def explain_with_gpt4(category, value, date):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a pharmaceutical expert..."},
            {"role": "user", "content": f"Explain why {category} sales are {value}..."}
        ]
    )
    return response.choices[0].message.content
```

### Option B: Use Smaller Open-Source Models
```python
# Use Llama 3.1 8B without fine-tuning (less accurate but works)
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
explanation = generator(prompt, max_length=512)
```


## TROUBLESHOOTING

### Error: "CUDA out of memory"
**Solution:** Reduce batch size in `fine_tune_llm.py`:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Reduce from 4 to 2
    gradient_accumulation_steps=8    # Increase from 4 to 8
)
```

### Error: "Model not found on Hugging Face"
**Solution:** Login to Hugging Face and accept Llama license:
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### Error: "Training takes too long"
**Solution:** Reduce epochs or use smaller dataset:
```python
# In data_preparation.py
dataset = prep.generate_dataset(samples_per_category=20)  # Reduce from 50
```

### Fallback Mode Activated
If fine-tuned model fails to load, system uses rule-based explanations automatically.


## EXPECTED RESULTS

### Before Fine-Tuning (Baseline):
- Generic explanations
- Limited pharmaceutical knowledge
- No Sri Lankan context

### After Fine-Tuning:
- Pharmaceutical domain expertise
- Sri Lankan healthcare context
- Seasonal pattern awareness
- Stakeholder-specific recommendations
- Causal reasoning

### Evaluation Metrics:
- Human evaluation by pharmacists/health officials
- ROUGE score for text similarity
- Factual accuracy on drug information
- Stakeholder usefulness ratings


## PRODUCTION DEPLOYMENT

### Model Size:
- Full model: ~16GB
- LoRA adapters only: ~100MB

### Inference Speed:
- GPU: ~2-3 seconds per explanation
- CPU: ~10-15 seconds per explanation

### Deployment Options:
1. **Local:** Load model in Flask app (requires 16GB GPU)
2. **API Service:** Deploy on separate ML server with GPU
3. **Cloud:** Use AWS SageMaker, Google Vertex AI, or Azure ML
4. **Hybrid:** Fine-tuned for complex cases, rules for simple ones


## COST ESTIMATION

### Fine-Tuning Costs:
- Cloud GPU (A100): ~$1-2/hour × 3 hours = $3-6 total
- Local GPU: Electricity cost only

### Inference Costs (if using API):
- OpenAI GPT-4: ~$0.03 per explanation
- Self-hosted: Infrastructure cost only
- Hybrid: $0.01 per explanation average


## NEXT STEPS AFTER FINE-TUNING

1. **Evaluation:** Test with real pharmacists and health officials
2. **Iteration:** Collect feedback and retrain with better examples
3. **Multilingual:** Fine-tune for Sinhala and Tamil languages
4. **Integration:** Add to forecast UI with "Explain" button
5. **Monitoring:** Track explanation quality and user satisfaction


## SUPPORT

For issues or questions:
1. Check logs in `models_llm/pharmaceutical_llama/`
2. Review Hugging Face transformers documentation
3. Test with smaller model first (Llama 7B)
4. Use fallback mode for development


---

🎯 **QUICK START COMMAND SEQUENCE:**

```bash
# Complete pipeline in 4 commands
pip install -r requirements-llm.txt
python src/llm/data_preparation.py
python src/llm/medical_document_scraper.py
python src/llm/fine_tune_llm.py

# Then test
python src/llm/llm_api.py
```

**Estimated time:** 3-5 hours (mostly GPU training)
**Result:** Fine-tuned LLM generating pharmaceutical sales explanations!

---

✅ Your CSV data + Medical documents = Smart pharmaceutical AI explainer!
