# Step 2: Medical Document Collection

## 🎯 Purpose
Collect pharmaceutical domain knowledge to enhance LLM's medical expertise.

## 📝 What This Does

1. **WHO ATC Guidelines** - All 8 drug categories (M01AB, M01AE, N02BA, etc.)
2. **Drug Information** - Mechanisms, side effects, contraindications
3. **Sri Lankan Context** - Healthcare system, climate, diseases
4. **Epidemiology** - Disease patterns, seasonal variations

## ⚡ Quick Run

```bash
python scrape_documents.py
```

**Time**: ~1-2 minutes  
**Output**: `../output/medical_docs/` (40-50 documents)

## 📊 Document Types Generated

| Type | Count | Content |
|------|-------|---------|
| ATC Guidelines | 8 | WHO classification, therapeutic uses |
| Drug Labels | 8 | Common medications in each category |
| Sri Lankan Context | 4 | Healthcare system, climate, epidemiology |
| PubMed Topics | 8-20 | Research articles (templates) |
| **Total** | **28-40** | Comprehensive pharmaceutical knowledge |

## 📁 Output Structure

```
../output/medical_docs/
├── raw_documents.json          # All documents with metadata
├── processed_documents.jsonl   # Formatted for training
└── document_index.txt          # Quick reference list
```

## 🔍 Example Document Content

### WHO ATC Guideline (C1 - M01AB)
```
ATC Code M01AB - Anti-inflammatory Acetic Acid Derivatives

Common Medications:
- Diclofenac: Widely used NSAID for pain and inflammation
- Indomethacin: Potent NSAID for arthritis

Therapeutic Indications:
- Rheumatoid arthritis
- Osteoarthritis
- Post-operative pain

Contraindications:
- Active peptic ulcer disease
- Severe heart failure
- Third trimester pregnancy

Side Effects:
- GI bleeding (most common)
- Cardiovascular events
- Renal impairment

Epidemiology:
High utilization in tropical climates due to musculoskeletal 
conditions exacerbated by humidity...
```

### Sri Lankan Healthcare Context
```
Healthcare System in Sri Lanka:
- Public sector: Free universal healthcare
- Private sector: Growing rapidly in urban areas
- Pharmacy network: Extensive, both chain and independent
- Health indicators: Life expectancy 77 years, literacy >90%
- Challenges: Aging population, rising NCDs, import dependency

Disease Epidemiology:
- Dengue: Seasonal outbreaks during monsoon
- Diabetes: 10-15% prevalence
- Hypertension: 25-30% of adults
- Asthma: 5-8%, higher in urban areas
```

## 🔧 Configuration

Edit `scrape_documents.py` to customize:

```python
# Adjust number of PubMed articles (line ~150)
max_articles = 20  # Increase for more medical knowledge

# Add custom documents (line ~200)
custom_docs = [
    {'title': 'Your Document', 'content': '...'}
]
```

## 🌐 Real Data Collection (Advanced)

The script uses template data. For real scraping:

### PubMed API
```python
from Bio import Entrez
Entrez.email = "your@email.com"
handle = Entrez.esearch(db="pubmed", term="drug utilization Sri Lanka", retmax=20)
```

### WHO API
```python
import requests
url = "https://www.whocc.no/atc_ddd_index/?code=M01AB"
response = requests.get(url)
```

## ✅ Verify Output

Check generated documents:

```powershell
# Count documents
(Get-Content ..\output\medical_docs\processed_documents.jsonl).Count

# View first document
Get-Content ..\output\medical_docs\document_index.txt
```

## 📋 Document Format

Each document is structured as:

```json
{
  "source": "WHO ATC Guidelines",
  "title": "ATC Code M01AB",
  "content": "Detailed pharmaceutical information...",
  "category": "guidelines",
  "language": "en"
}
```

Training format:
```json
{
  "instruction": "Provide pharmaceutical information...",
  "input": "Document: ATC Code M01AB\nQuestion: Summarize key info",
  "output": "Anti-inflammatory medications used for arthritis..."
}
```

## 🐛 Troubleshooting

### "No documents generated"
**Solution**: Check output directory exists:
```bash
mkdir -p ../output/medical_docs
```

### Want more medical knowledge?
**Solution**: 
1. Add your own documents to `custom_docs` list
2. Increase `max_articles` parameter
3. Integrate real PubMed/WHO APIs

### Documents too generic?
**Solution**: This is intentional for demo. In production:
- Use real PubMed API with your institution credentials
- Scrape WHO/FDA websites (check robots.txt)
- Add local Sri Lankan health ministry reports

## 💡 Enhancement Ideas

### Add Real Medical Papers
```python
# Use Semantic Scholar API
import requests
query = "pharmaceutical utilization Sri Lanka"
papers = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}")
```

### Include Local Guidelines
- Sri Lankan Ministry of Health circulars
- National Formulary guidelines
- NMRA (National Medicines Regulatory Authority) documents

### Multi-Language Support
- Add Sinhala translations of drug information
- Tamil medical terminology
- English-Sinhala-Tamil parallel corpus

## ✅ Next Step

Once you see documents in `../output/medical_docs/`, proceed to:
```bash
cd ../3_fine_tuning
```

Read `../3_fine_tuning/README.md` for Step 3!

---

**40+ Documents Generated? ✅ Move to Step 3: Fine-Tuning**
