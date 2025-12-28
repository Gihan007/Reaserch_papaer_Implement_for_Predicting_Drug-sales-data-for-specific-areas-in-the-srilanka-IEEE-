# LLM Fine-Tuning System - Output Directory Structure

This directory contains all generated files from the LLM fine-tuning pipeline.

## 📁 Directory Structure

```
output/
├── training_data/              # Step 1: Generated training examples
│   ├── sales_explanations.json      # 400+ training examples (JSON)
│   ├── sales_explanations.jsonl     # HuggingFace format
│   └── preview.txt                   # Human-readable preview
│
├── medical_docs/               # Step 2: Collected medical documents
│   ├── raw_documents.json            # 40+ source documents
│   ├── processed_documents.jsonl     # Training format
│   └── document_index.txt            # Quick reference
│
├── fine_tuned_model/           # Step 3: Fine-tuned LLM
│   ├── adapter_model.bin             # LoRA weights (62 MB)
│   ├── adapter_config.json           # LoRA configuration
│   ├── tokenizer_config.json         # Tokenizer settings
│   ├── special_tokens_map.json       # Token mappings
│   └── training_log.json             # Training metrics
│
└── inference_tests/            # Step 4: Test results
    ├── test_results.json             # All test outputs
    ├── comparison.txt                # Fine-tuned vs baseline
    └── sample_explanations.md        # Human-readable examples
```

## 🚀 Generation Commands

To populate this directory, run:

```bash
# Step 1: Generate training data
python llm_finetuning/1_data_preparation/prepare_data.py

# Step 2: Collect medical documents
python llm_finetuning/2_medical_documents/scrape_documents.py

# Step 3: Fine-tune model (GPU required, 2-4 hours)
python llm_finetuning/3_fine_tuning/train_llm.py

# Step 4: Test inference
python llm_finetuning/4_inference/test_explainer.py
```

## 📊 Expected File Sizes

| File | Size | Description |
|------|------|-------------|
| training_data/sales_explanations.json | ~800 KB | 400+ training examples |
| medical_docs/raw_documents.json | ~200 KB | 40+ medical documents |
| fine_tuned_model/adapter_model.bin | ~62 MB | LoRA adapter weights |
| inference_tests/test_results.json | ~50 KB | Test outputs |

**Total**: ~65 MB (just the LoRA adapters, not the full 15GB base model)

## ✅ Verify Files

Check if all files generated successfully:

```powershell
# Check training data
Get-ChildItem llm_finetuning\output\training_data

# Check medical docs
Get-ChildItem llm_finetuning\output\medical_docs

# Check model files
Get-ChildItem llm_finetuning\output\fine_tuned_model

# Check test results
Get-ChildItem llm_finetuning\output\inference_tests
```

## 🔒 .gitignore

Add to your `.gitignore`:

```
# LLM output files (large)
llm_finetuning/output/fine_tuned_model/*.bin
llm_finetuning/output/fine_tuned_model/*.safetensors

# Keep config files (small)
!llm_finetuning/output/fine_tuned_model/*.json
```

## 📦 Backup Important Files

Before deploying, backup:
- `fine_tuned_model/` - Your trained model
- `training_data/` - Can regenerate training data
- `inference_tests/` - For quality validation

```bash
# Compress model for backup
tar -czf fine_tuned_model_backup.tar.gz llm_finetuning/output/fine_tuned_model/
```

## 🗑️ Clean Output

To start fresh:

```powershell
# WARNING: Deletes all generated files!
Remove-Item llm_finetuning\output\* -Recurse -Force

# Then regenerate from Step 1
```

---

**Output files generated? ✅ Your LLM system is ready!**
