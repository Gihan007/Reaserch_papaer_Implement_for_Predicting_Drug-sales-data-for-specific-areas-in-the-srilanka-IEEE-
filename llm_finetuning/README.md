# 🚀 LLM Fine-Tuning for Pharmaceutical Sales Explanation

## 📁 Folder Structure

```
llm_finetuning/
├── 1_data_preparation/          # Step 1: Prepare training data from CSV
│   ├── prepare_data.py
│   └── README.md
├── 2_medical_documents/          # Step 2: Collect medical knowledge
│   ├── scrape_documents.py
│   └── README.md
├── 3_fine_tuning/                # Step 3: Fine-tune the model
│   ├── train_llm.py
│   ├── config.py
│   └── README.md
├── 4_inference/                  # Step 4: Use the fine-tuned model
│   ├── llm_explainer.py
│   └── README.md
├── 5_integration/                # Step 5: Integrate with Flask
│   ├── api_endpoints.py
│   └── README.md
├── output/                       # Generated files
│   ├── training_data/
│   ├── medical_docs/
│   └── fine_tuned_model/
└── README.md                     # Main guide (this file)
```

## 🎯 What This Does

Transforms your pharmaceutical forecasting system from:
- **Before**: "C1 sales will be 50.38 units" ❌
- **After**: "C1 sales rising due to monsoon humidity (85%) triggering arthritis in aging population. Government should maintain 20% buffer stock..." ✅

## ⚡ Quick Start (5 Steps)

### **Step 1: Prepare Training Data** (5 minutes)
```bash
cd llm_finetuning/1_data_preparation
python prepare_data.py
```
**Output**: 400+ training examples from your C1.csv - C8.csv files

### **Step 2: Collect Medical Documents** (2 minutes)
```bash
cd ../2_medical_documents
python scrape_documents.py
```
**Output**: 40-50 pharmaceutical documents (WHO, FDA, Sri Lankan context)

### **Step 3: Fine-Tune LLM** (2-4 hours on GPU)
```bash
cd ../3_fine_tuning
python train_llm.py
```
**Output**: Fine-tuned Llama 3.1 8B model

### **Step 4: Test Inference** (30 seconds)
```bash
cd ../4_inference
python llm_explainer.py
```
**Output**: Test the model with sample predictions

### **Step 5: Integrate with Flask** (5 minutes)
```bash
cd ../5_integration
# Copy api_endpoints.py code to your app.py
```
**Output**: New `/api/explain` endpoint in your Flask app

## 📊 Expected Results

### Training Data Format
```json
{
  "instruction": "You are a pharmaceutical expert...",
  "input": "Category C1 sales: 50.38, +15% increase, monsoon season",
  "output": "Anti-inflammatory sales rising due to: 1) Humidity..."
}
```

### LLM Output Example
```
Why is C1 increasing?

**Analysis of M01AB (Anti-inflammatory) Sales Pattern**

1. **Monsoon Season Impact** (Primary Factor)
   High humidity (85-90%) triggers arthritis flare-ups in 
   aging population (25% over 60 in urban Colombo)

2. **Government Health Campaign**
   Recent 'Self-Care Week' promoted OTC medications

🏥 Public Health Implications:
- Monitor for overuse in elderly populations
- Ensure adequate supply chain

🏛️ Government Actions:
- Maintain 20% buffer stock
- Launch awareness campaigns

💊 Pharmacy Recommendations:
- Stock 60 units (20% above forecast)
- Prepare gastro-protective agents
```

## 💻 Hardware Requirements

### Minimum (CPU Only)
- RAM: 16GB+
- Storage: 50GB
- Time: ~8 hours training

### Recommended (GPU)
- GPU: NVIDIA RTX 3090 / 4090 (16GB+ VRAM)
- RAM: 32GB
- Storage: 50GB
- Time: 2-4 hours training

## 🔧 Installation

```bash
# Install dependencies
pip install -r llm_finetuning/requirements.txt

# Or install manually
pip install torch transformers accelerate peft bitsandbytes datasets
```

## 📈 Progress Tracking

Each step creates output files you can inspect:
- ✅ Step 1: `output/training_data/sales_explanations.json`
- ✅ Step 2: `output/medical_docs/documents.json`
- ✅ Step 3: `output/fine_tuned_model/` (model files)
- ✅ Step 4: Console output with test explanation
- ✅ Step 5: Flask API endpoint working

## 🎓 For Your IEEE Paper

This adds:
- **Novel Contribution**: First LLM-based pharmaceutical forecasting explainer
- **Domain Expertise**: Fine-tuned on pharmaceutical + Sri Lankan data
- **Multi-Stakeholder**: Pharmacies, government, public health insights
- **Explainable AI**: Not just predictions, but causal reasoning

## 🆘 Troubleshooting

### "CUDA out of memory"
**Solution**: Edit `3_fine_tuning/config.py`:
```python
BATCH_SIZE = 2  # Reduce from 4
GRADIENT_ACCUMULATION = 8  # Increase from 4
```

### "Model not found"
**Solution**: Login to Hugging Face:
```bash
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
```

### "Training too slow"
**Solution**: Reduce dataset size in `1_data_preparation/prepare_data.py`:
```python
samples_per_category = 20  # Reduce from 50
```

## 🚀 Alternative: Skip Fine-Tuning

If you don't have GPU, use OpenAI GPT-4 instead:
```python
# In 4_inference/llm_explainer.py
import openai
explanation = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

## 📞 Support

Each folder has its own README.md with:
- Detailed step-by-step instructions
- Code explanation
- Expected output
- Troubleshooting tips

Start with `llm_finetuning/1_data_preparation/README.md`

---

**Ready to transform your predictions into insights? Start with Step 1! 🎯**
