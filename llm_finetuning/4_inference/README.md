# Step 4: Test Fine-Tuned Model

## 🎯 Purpose
Test your fine-tuned pharmaceutical explainer and compare with baseline.

## ⚡ Quick Run

```bash
python test_explainer.py
```

## 📊 What This Does

1. **Loads fine-tuned model** from `../output/fine_tuned_model/`
2. **Tests pharmaceutical explanations** with sample predictions
3. **Compares** fine-tuned vs base Llama 3.1
4. **Saves results** for analysis

## 🔍 Example Test

### Input
```json
{
  "category": "C1 (M01AB - Anti-inflammatory)",
  "prediction": 50.38,
  "week": 12,
  "year": 2018,
  "location": "Colombo"
}
```

### Fine-Tuned Output ✅
```
M01AB (Anti-inflammatory Acetic Acid Derivatives) - 50.38 units forecast for Week 12, 2018 in Colombo

ANALYSIS:
The elevated demand reflects multiple converging factors characteristic of this period:

SEASONAL FACTORS (40% influence):
• Week 12 coincides with inter-monsoon transition (March-April)
• High humidity (75-85%) exacerbates musculoskeletal conditions
• Temperature variations stress joints, increasing arthritis symptoms
• Historical data shows 15-20% NSA sales spike during this period

DEMOGRAPHIC PATTERNS (30% influence):
• Colombo's aging population (12% over 65 years)
• Urban lifestyle: prolonged sitting, repetitive work injuries
• Higher disposable income enables private healthcare access
• Professional workers seek rapid pain relief

PUBLIC HEALTH CONTEXT (20% influence):
• Dengue surveillance heightened in early 2018
• Aspirin contraindicated, shifting demand to acetic acid NSAIDs
• Post-festival period (Sinhala New Year) with travel-related injuries

CLINICAL CONSIDERATIONS:
⚠️  Monitor for adverse effects in high-utilization period:
   - GI bleeding risk (especially elderly, H. pylori prevalence 60%)
   - Cardiovascular events (hypertension affects 25% adults)
   - Renal impairment in dehydration (tropical climate factor)

RECOMMENDATIONS:

For Healthcare Providers:
1. Ensure gastroprotection (PPI co-prescription) for high-risk patients
2. Screen for cardiovascular contraindications before prescribing
3. Educate patients on hydration importance in tropical climate
4. Monitor for NSAID overuse, dependency patterns

For Government/Regulators:
1. Ensure adequate stock for seasonal surge (+20% buffer)
2. Strengthen pharmacovigilance during high-utilization periods
3. Public awareness campaigns on safe NSAID use
4. Consider price controls to prevent hoarding

For Public Awareness:
1. Understand dengue season requires avoiding aspirin
2. Seek medical advice for chronic pain (>7 days)
3. Report side effects: black stools, chest pain, breathing difficulty
4. Alternative approaches: physiotherapy, weight management for arthritis
```

### Base Model Output ❌
```
The drug category C1 is predicted to sell 50.38 units in week 12 of 2018. 
This could be due to seasonal factors or increased demand in the area.
```

**Difference:** Fine-tuned model provides specific pharmaceutical knowledge, clinical context, and stakeholder recommendations!

## 🔧 Test Scenarios

The script tests multiple scenarios:

### Scenario 1: High NSAID Demand (Monsoon)
- Category: C1 (M01AB)
- Prediction: 50.38 units
- Week: 12 (March-April)
- Context: Monsoon transition, humidity

### Scenario 2: Analgesic Spike (Dengue Season)
- Category: C3 (N02BA - Aspirin)
- Prediction: 78.62 units
- Week: 24 (June)
- Context: Dengue outbreak, aspirin contraindication

### Scenario 3: Anxiolytic Usage (Urban Stress)
- Category: C5 (N05B)
- Prediction: 32.15 units
- Location: Colombo
- Context: Mental health awareness, urban stress

## 📁 Output Files

```
../output/inference_tests/
├── test_results.json              # All test outputs
├── comparison.txt                  # Fine-tuned vs baseline
└── sample_explanations.md          # Human-readable examples
```

## ✅ Verify Model Quality

### Good Explanation Checklist
- ✅ Mentions specific ATC category details
- ✅ Explains seasonal/environmental factors
- ✅ References Sri Lankan healthcare context
- ✅ Provides clinical considerations
- ✅ Gives stakeholder-specific recommendations
- ✅ Cites contraindications/side effects

### Poor Explanation Signs
- ❌ Generic "increased demand" without specifics
- ❌ No pharmaceutical knowledge evident
- ❌ Missing stakeholder recommendations
- ❌ No clinical safety considerations

## 🐛 Troubleshooting

### "Model not found"
**Solution:** Run Step 3 first to fine-tune the model
```bash
cd ../3_fine_tuning
python train_llm.py
```

### Outputs are generic/poor quality
**Causes:**
1. Training didn't converge (loss still high)
2. Not enough training data
3. Testing with out-of-distribution examples

**Solutions:**
```python
# Check training loss
import json
with open('../output/fine_tuned_model/training_log.json') as f:
    log = json.load(f)
    print(f"Final loss: {log.get('train_loss', 'N/A')}")
# Should be < 0.3

# Retrain with more epochs
# Edit ../3_fine_tuning/config.py:
EPOCHS = 5  # Instead of 3
```

### "CUDA out of memory" during inference
**Solution:** Use smaller batch size or CPU inference
```python
# test_explainer.py line 25
device = "cpu"  # Force CPU inference
```

### Model loads but generates garbage
**Causes:**
- Incomplete training (interrupted)
- Corrupted checkpoint
- Wrong tokenizer

**Solution:**
```bash
# Check model files exist
ls ../output/fine_tuned_model/
# Should see: adapter_model.bin, adapter_config.json, tokenizer files

# Re-run training from last checkpoint
cd ../3_fine_tuning
python train_llm.py --resume
```

## 🔬 Advanced Testing

### Test Custom Predictions
```python
# test_custom.py
from llm_explainer import PharmaceuticalLLM

explainer = PharmaceuticalLLM()

custom_input = {
    "category": "C2",
    "prediction": 65.2,
    "week": 30,
    "year": 2019,
    "location": "Galle"
}

explanation = explainer.explain(custom_input)
print(explanation)
```

### Batch Testing
```python
# test_batch.py
test_cases = [
    {"category": "C1", "prediction": 45.2, "week": 10},
    {"category": "C2", "prediction": 38.7, "week": 20},
    # ... more cases
]

for case in test_cases:
    explanation = explainer.explain(case)
    print(f"\n{case['category']} → {explanation[:200]}...")
```

### Compare Multiple Models
```python
# compare_models.py
models = [
    "../output/fine_tuned_model",           # Your fine-tuned
    "meta-llama/Meta-Llama-3.1-8B-Instruct", # Base model
    "microsoft/Phi-3-mini-4k-instruct"      # Alternative
]

for model_path in models:
    explainer = PharmaceuticalLLM(model_path)
    output = explainer.explain(test_input)
    print(f"\n{model_path}:\n{output}\n{'-'*80}")
```

## 💡 Improvement Ideas

### Enhance Explanations
1. **Add confidence scores** to predictions
2. **Include historical comparisons** (YoY growth)
3. **Show regional variations** (Colombo vs rural)
4. **Cite specific studies** (PubMed links)

### Optimize Performance
1. **Quantize further** (use 8-bit instead of 16-bit)
2. **Use Flash Attention** for 2x speed
3. **Cache common queries** with Redis
4. **Batch inference** for multiple predictions

## ✅ Success Criteria

Model ready for deployment when:
- ✅ Generates pharmaceutical-specific explanations
- ✅ Includes seasonal/environmental context
- ✅ Provides stakeholder recommendations
- ✅ Mentions contraindications/safety
- ✅ References Sri Lankan healthcare system
- ✅ Inference time < 5 seconds per explanation

## ✅ Next Step

Once testing validates quality:
```bash
cd ../5_integration
```

Read `../5_integration/README.md` to integrate with your Flask API!

---

**Explanations look good? ✅ Move to Step 5: API Integration**

## 🚫 Alternative Inference Methods

### Option A: Use OpenAI GPT-4 (No GPU)
```python
import openai

# Load training examples as few-shot prompts
with open('../output/training_data/sales_explanations.json') as f:
    examples = json.load(f)[:5]  # Use 5 examples

prompt = f"""Examples:\n{examples}\n\nNow explain: {{prediction}}"""
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### Option B: Use Hugging Face Inference API
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/your-model"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

response = requests.post(API_URL, headers=headers, json={
    "inputs": prompt
})
```

### Option C: Use Smaller Model (Phi-3 Mini)
```python
# Only 3.8B parameters, runs on CPU
model = "microsoft/Phi-3-mini-4k-instruct"
# Much faster, still decent quality
```
