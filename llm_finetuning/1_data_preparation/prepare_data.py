"""
Step 1: Data Preparation - Convert CSV sales data to LLM training format
Run from project root: python llm_finetuning/1_data_preparation/prepare_data.py
"""

import pandas as pd
import json
import os
import sys
import numpy as np
import random
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class PharmaceuticalDataPreparation:
    """Prepare training data for fine-tuning LLM on pharmaceutical sales explanations"""
    
    def __init__(self):
        self.categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        
        # ATC Category descriptions
        self.category_info = {
            'C1': {
                'code': 'M01AB',
                'name': 'Anti-inflammatory and antirheumatic products, Acetic acid derivatives',
                'common_drugs': ['Diclofenac', 'Indomethacin'],
                'conditions': ['Arthritis', 'Joint pain', 'Inflammation'],
                'risk_factors': ['Humid weather', 'Aging population', 'Monsoon season']
            },
            'C2': {
                'code': 'M01AE',
                'name': 'Anti-inflammatory products, Propionic acid derivatives',
                'common_drugs': ['Ibuprofen', 'Naproxen'],
                'conditions': ['Pain relief', 'Fever', 'Headaches'],
                'risk_factors': ['Flu season', 'Weather changes', 'Urban lifestyle']
            },
            'C3': {
                'code': 'N02BA',
                'name': 'Analgesics and antipyretics, Salicylic acid derivatives',
                'common_drugs': ['Aspirin'],
                'conditions': ['Fever', 'Pain', 'Cardiovascular prevention'],
                'risk_factors': ['Dengue outbreaks', 'Viral infections']
            },
            'C4': {
                'code': 'N02BE',
                'name': 'Analgesics and antipyretics, Pyrazolones',
                'common_drugs': ['Metamizole'],
                'conditions': ['Severe pain', 'High fever'],
                'risk_factors': ['Hospital procedures', 'Emergency care']
            },
            'C5': {
                'code': 'N05B',
                'name': 'Anxiolytics',
                'common_drugs': ['Diazepam', 'Lorazepam'],
                'conditions': ['Anxiety', 'Stress', 'Sleep disorders'],
                'risk_factors': ['Economic stress', 'Urban living', 'Mental health awareness']
            },
            'C6': {
                'code': 'N05C',
                'name': 'Hypnotics and sedatives',
                'common_drugs': ['Zolpidem', 'Zopiclone'],
                'conditions': ['Insomnia', 'Sleep disturbances'],
                'risk_factors': ['Stress', 'Lifestyle changes', 'Mental health']
            },
            'C7': {
                'code': 'R03',
                'name': 'Drugs for obstructive airway diseases',
                'common_drugs': ['Salbutamol', 'Beclometasone'],
                'conditions': ['Asthma', 'COPD', 'Bronchitis'],
                'risk_factors': ['Air pollution', 'Seasonal allergies', 'Dust']
            },
            'C8': {
                'code': 'R06',
                'name': 'Antihistamines for systemic use',
                'common_drugs': ['Cetirizine', 'Loratadine'],
                'conditions': ['Allergies', 'Hay fever', 'Urticaria'],
                'risk_factors': ['Pollen season', 'Weather changes', 'Air quality']
            }
        }
        
        # Sri Lankan seasonal patterns
        self.seasons = {
            'monsoon': {'months': [5, 6, 10, 11], 'description': 'Monsoon season with high humidity'},
            'dry': {'months': [1, 2, 3, 7, 8], 'description': 'Dry season with dust and heat'},
            'inter_monsoon': {'months': [4, 9, 12], 'description': 'Inter-monsoon transition'}
        }
        
        self.dataset = []
    
    def load_csv(self, category: str) -> pd.DataFrame:
        """Load sales data for a category"""
        try:
            df = pd.read_csv(f'{category}.csv')
            df['datum'] = pd.to_datetime(df['datum'])
            df = df.sort_values('datum')
            print(f"✓ Loaded {category}: {len(df)} records")
            return df
        except Exception as e:
            print(f"✗ Error loading {category}: {e}")
            return None
    
    def get_season(self, month: int) -> str:
        """Get season based on month"""
        for season, info in self.seasons.items():
            if month in info['months']:
                return season
        return 'inter_monsoon'
    
    def calculate_stats(self, df: pd.DataFrame, category: str, idx: int) -> Dict:
        """Calculate statistics for context"""
        current_value = df[category].iloc[idx]
        
        # Historical context (4-week window)
        if idx >= 4:
            previous_values = df[category].iloc[idx-4:idx].values
            avg_previous = np.mean(previous_values)
            change_pct = ((current_value - avg_previous) / avg_previous) * 100
            trend = 'increasing' if change_pct > 5 else ('decreasing' if change_pct < -5 else 'stable')
        else:
            avg_previous = current_value
            change_pct = 0
            trend = 'stable'
        
        date = df['datum'].iloc[idx]
        
        return {
            'current_value': float(current_value),
            'previous_avg': float(avg_previous),
            'change_pct': float(change_pct),
            'trend': trend,
            'date': date.strftime('%Y-%m-%d'),
            'month': date.month,
            'season': self.get_season(date.month),
            'week_of_year': date.isocalendar()[1]
        }
    
    def generate_explanation(self, category: str, stats: Dict) -> str:
        """Generate detailed explanation based on category and statistics"""
        
        cat_info = self.category_info[category]
        factors = []
        
        # Factor 1: Seasonal influence
        if stats['season'] == 'monsoon':
            if category in ['C1', 'C2']:
                factors.append({
                    'title': 'Monsoon Season Impact',
                    'description': f"High humidity during monsoon season (85-90%) triggers increased {', '.join(cat_info['conditions'][:2])}. The damp weather exacerbates musculoskeletal conditions in Sri Lanka's aging population."
                })
            elif category in ['C7', 'C8']:
                factors.append({
                    'title': 'Respiratory Challenges',
                    'description': f"Monsoon brings increased air moisture and mold growth, triggering {', '.join(cat_info['conditions'][:2])} in susceptible populations."
                })
        
        # Factor 2: Trend-based reasoning
        if stats['change_pct'] > 10:
            factors.append({
                'title': 'Significant Demand Increase',
                'description': f"Rising demand (+{stats['change_pct']:.1f}%) suggests increased prevalence of {cat_info['conditions'][0]} requiring {cat_info['common_drugs'][0]} treatment."
            })
        
        # Factor 3: Healthcare access
        factors.append({
            'title': 'Healthcare Access and Demographics',
            'description': f"Sri Lanka's aging population (15% over 60) and improved healthcare access increase sustained demand for {', '.join(cat_info['common_drugs'][:2])}."
        })
        
        # Build explanation
        explanation = f"""**Analysis of {cat_info['code']} ({cat_info['name']}) Sales Pattern**

"""
        
        for i, factor in enumerate(factors[:3], 1):
            explanation += f"""**{i}. {factor['title']}**
{factor['description']}

"""
        
        explanation += f"""**🏥 Public Health Implications:**
- Monitor for potential overuse or misuse of {', '.join(cat_info['common_drugs'][:2])}
- Ensure adequate supply chain to prevent shortages
- Track adverse events and drug interactions

**🏛️ Government Action Items:**
- Maintain {abs(stats['change_pct']) + 15:.0f}% buffer stock in national pharmacies
- Deploy mobile health units to underserved areas
- Launch public awareness on proper medication use

**💊 Pharmacy Management:**
- Adjust inventory: Stock {stats['current_value'] * 1.2:.0f} units (20% safety margin)
- Prepare related categories for cross-sell opportunities
- Train staff on patient consultation and OTC guidance

**⚠️ Risk Awareness:**
- Monitor for side effects in elderly patients
- Check contraindications: Pregnancy, renal/hepatic impairment
- Watch for drug interactions with anticoagulants and antihypertensives
"""
        
        return explanation.strip()
    
    def generate_training_example(self, category: str, stats: Dict) -> Dict:
        """Generate a single training example"""
        
        cat_info = self.category_info[category]
        
        instruction = (
            "You are a pharmaceutical analytics expert specializing in Sri Lankan healthcare. "
            "Analyze the drug sales prediction data and explain the reasons for the sales pattern. "
            "Provide insights for pharmacists, government officials, and public health awareness."
        )
        
        input_text = f"""Drug Sales Analysis Request:

Category: {category} ({cat_info['code']} - {cat_info['name']})
Current Sales: {stats['current_value']:.2f} units
Date: {stats['date']}
Trend: {stats['trend']} ({stats['change_pct']:+.1f}% change from previous weeks)
Season: {stats['season']}

Question: Why is this drug category showing this sales pattern? What should stakeholders be aware of?"""

        output_text = self.generate_explanation(category, stats)
        
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        }
    
    def process_category(self, category: str, samples: int = 50):
        """Process one category and generate training examples"""
        
        df = self.load_csv(category)
        if df is None or len(df) < 5:
            return
        
        # Sample random time points
        total_rows = len(df)
        if total_rows < samples + 4:
            sample_indices = range(4, total_rows)
        else:
            sample_indices = random.sample(range(4, total_rows), samples)
        
        for idx in sample_indices:
            stats = self.calculate_stats(df, category, idx)
            example = self.generate_training_example(category, stats)
            self.dataset.append(example)
    
    def save_dataset(self, output_dir: str = '../output/training_data'):
        """Save dataset in multiple formats"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'sales_explanations.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        # Save as JSONL (for Hugging Face)
        jsonl_path = os.path.join(output_dir, 'sales_explanations.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save preview
        preview_path = os.path.join(output_dir, 'preview.txt')
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING DATA PREVIEW (First 2 Examples)\n")
            f.write("="*80 + "\n\n")
            
            for i, example in enumerate(self.dataset[:2], 1):
                f.write(f"\n{'─'*80}\n")
                f.write(f"EXAMPLE {i}\n")
                f.write(f"{'─'*80}\n")
                f.write(f"\n📋 INSTRUCTION:\n{example['instruction']}\n")
                f.write(f"\n📊 INPUT:\n{example['input']}\n")
                f.write(f"\n💡 OUTPUT:\n{example['output']}\n")
        
        print(f"\n✅ Dataset saved:")
        print(f"   JSON:    {json_path}")
        print(f"   JSONL:   {jsonl_path}")
        print(f"   Preview: {preview_path}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total examples: {len(self.dataset)}")
        print(f"   Avg input length: {np.mean([len(ex['input']) for ex in self.dataset]):.0f} chars")
        print(f"   Avg output length: {np.mean([len(ex['output']) for ex in self.dataset]):.0f} chars")


def main():
    print("🚀 Step 1: Data Preparation - CSV to LLM Training Format")
    print("="*80)
    
    prep = PharmaceuticalDataPreparation()
    
    print("\n📂 Loading CSV files and generating training examples...")
    print("   (Processing 8 categories × 50 samples = 400 examples)")
    print()
    
    for category in prep.categories:
        prep.process_category(category, samples=50)
    
    print(f"\n✅ Generated {len(prep.dataset)} training examples!")
    
    prep.save_dataset()
    
    print("\n" + "="*80)
    print("✅ STEP 1 COMPLETE!")
    print("="*80)
    print("\nNext step:")
    print("  cd ../2_medical_documents")
    print("  python scrape_documents.py")
    print("\nOr read: ../2_medical_documents/README.md")


if __name__ == "__main__":
    main()
