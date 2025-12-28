# Step 1: Data Preparation

## 🎯 Purpose
Convert your CSV sales data (C1.csv - C8.csv) into training examples for fine-tuning the LLM.

## 📝 What This Does

1. **Loads** all 8 category CSV files
2. **Analyzes** sales patterns, trends, seasonal variations
3. **Generates** 400+ training examples (50 per category)
4. **Formats** into instruction-input-output structure
5. **Saves** to `../output/training_data/`

## ⚡ Quick Run

```bash
python prepare_data.py
```

**Time**: ~2-3 minutes  
**Output**: `../output/training_data/sales_explanations.json` (400+ examples)

## 📊 Example Output

### Input (What the LLM sees):
```
Category: C1 (M01AB - Anti-inflammatory)
Current Sales: 50.38 units
Date: 2025-12-15
Trend: increasing (+15.1%)
Season: monsoon

Question: Why is this drug category showing this sales pattern?
```

### Output (What the LLM learns to generate):
```
**Analysis of M01AB Sales Pattern**

1. **Monsoon Season Impact**
   High humidity (85-90%) triggers arthritis flare-ups in aging 
   population, particularly affecting urban areas like Colombo.

2. **Weather-Related Inflammation**
   Monsoon conditions exacerbate musculoskeletal complaints...

🏥 Public Health Implications:
- Monitor for potential overuse
- Ensure adequate supply chain

🏛️ Government Recommendations:
- Maintain 20% buffer stock
- Launch public awareness campaigns
...
```

## 🔧 Configuration

Edit the script to customize:

```python
# Number of examples per category (line ~380)
samples_per_category = 50  # Reduce to 20 for faster training

# Categories to process (line ~10)
categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
```

## 📁 Output Files

| File | Description | Size |
|------|-------------|------|
| `sales_explanations.json` | Full dataset (formatted) | ~2-3 MB |
| `sales_explanations.jsonl` | Line-delimited (for HF) | ~2-3 MB |
| `preview.txt` | Sample examples (2) | ~5 KB |

## 🔍 Verify Output

Check the generated file:

```bash
# View first few lines
head -n 20 ../output/training_data/sales_explanations.jsonl

# Count examples
type ..\output\training_data\sales_explanations.jsonl | Measure-Object -Line
# Should show ~400 lines
```

## 📈 What Gets Analyzed

For each time point in your CSV:
- ✅ **Current sales value**
- ✅ **Historical trend** (4-week window)
- ✅ **Percentage change**
- ✅ **Season** (monsoon/dry/inter-monsoon)
- ✅ **Week of year**
- ✅ **Month**
- ✅ **Direction** (increasing/decreasing/stable)

## 🎓 Training Example Structure

```json
{
  "instruction": "You are a pharmaceutical analytics expert...",
  "input": "Drug Sales Analysis Request:\n\nCategory: C1...",
  "output": "**Analysis of M01AB Sales Pattern**\n\n1. ..."
}
```

This format works with:
- ✅ Hugging Face `transformers`
- ✅ Llama fine-tuning
- ✅ OpenAI fine-tuning API
- ✅ Most LLM training frameworks

## 🐛 Troubleshooting

### Error: "FileNotFoundError: C1.csv"
**Solution**: Run from project root or set base_path:
```python
prep = LLMDataPreparation(base_path='../..')  # Adjust path
```

### Error: "Empty dataset generated"
**Solution**: Check CSV files have data:
```bash
head C1.csv
# Should show: datum,C1
```

### Warning: "Only generated 50 examples"
**Solution**: This is normal if CSV has <54 rows. Most categories have 500+ rows and will generate full 50 examples.

## ✅ Next Step

Once you see output files in `../output/training_data/`, proceed to:
```bash
cd ../2_medical_documents
```

Read `../2_medical_documents/README.md` for Step 2!

---

**Output Verified? ✅ Move to Step 2: Medical Documents**
