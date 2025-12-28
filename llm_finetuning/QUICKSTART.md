# 🚀 QUICK START GUIDE

## One-Command Setup

```bash
# Install dependencies
pip install -r llm_finetuning/requirements.txt

# Generate training data (5 min)
python llm_finetuning/1_data_preparation/prepare_data.py

# Collect medical documents (2 min)
python llm_finetuning/2_medical_documents/scrape_documents.py
```

## What You Get

After running these 2 scripts:
- ✅ **400+ training examples** from your CSV data
- ✅ **40+ medical documents** for pharmaceutical knowledge
- ✅ Ready for fine-tuning (Step 3) or direct use with GPT-4

## File Output

```
llm_finetuning/output/
├── training_data/
│   ├── sales_explanations.json       # 400+ examples
│   ├── sales_explanations.jsonl      # HuggingFace format
│   └── preview.txt                    # See sample examples
└── medical_docs/
    ├── raw_documents.json             # 40+ documents
    ├── processed_documents.jsonl       # Training format
    └── document_index.txt              # Quick reference
```

## Check Your Output

```powershell
# View training examples preview
type llm_finetuning\output\training_data\preview.txt

# Count examples
(Get-Content llm_finetuning\output\training_data\sales_explanations.jsonl).Count

# View document index
type llm_finetuning\output\medical_docs\document_index.txt
```

## Next Steps

### Option A: Fine-Tune Your Own Model (GPU Required)
```bash
# See llm_finetuning/3_fine_tuning/README.md
cd llm_finetuning/3_fine_tuning
python train_llm.py
```
**Time**: 2-4 hours on GPU  
**Result**: Custom pharmaceutical AI explainer

### Option B: Use OpenAI GPT-4 (No GPU Needed)
```python
# See llm_finetuning/4_inference/README.md
import openai
# Use generated training data as few-shot examples
```
**Cost**: ~$0.03 per explanation  
**Result**: Instant pharmaceutical explanations

## 📖 Full Documentation

Read `llm_finetuning/README.md` for complete guide!

---

**Generated training data? ✅ You're ready for Step 3!**
