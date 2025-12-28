"""
LLM Fine-Tuning Data Preparation Pipeline
Converts CSV sales data + medical documents into training format for Llama 3.1
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple
import random

class LLMDataPreparation:
    """Prepare training data for fine-tuning LLM on pharmaceutical sales explanations"""
    
    def __init__(self, base_path=''):
        self.base_path = base_path
        self.categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        
        # ATC Category descriptions
        self.category_info = {
            'C1': {
                'code': 'M01AB',
                'name': 'Anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives',
                'common_drugs': ['Diclofenac', 'Indomethacin', 'Sulindac'],
                'conditions': ['Arthritis', 'Joint pain', 'Inflammation', 'Musculoskeletal disorders'],
                'risk_factors': ['Humid weather', 'Aging population', 'Physical labor', 'Monsoon season']
            },
            'C2': {
                'code': 'M01AE',
                'name': 'Anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives',
                'common_drugs': ['Ibuprofen', 'Naproxen', 'Ketoprofen'],
                'conditions': ['Pain relief', 'Fever', 'Headaches', 'Dental pain'],
                'risk_factors': ['Flu season', 'Weather changes', 'Stress', 'Urban lifestyle']
            },
            'C3': {
                'code': 'N02BA',
                'name': 'Analgesics and antipyretics, Salicylic acid and derivatives',
                'common_drugs': ['Aspirin', 'Acetylsalicylic acid'],
                'conditions': ['Fever', 'Pain', 'Cardiovascular prevention', 'Headaches'],
                'risk_factors': ['Dengue outbreaks', 'Viral infections', 'Seasonal diseases']
            },
            'C4': {
                'code': 'N02BE',
                'name': 'Analgesics and antipyretics, Pyrazolones',
                'common_drugs': ['Metamizole', 'Phenazone'],
                'conditions': ['Severe pain', 'High fever', 'Post-operative pain'],
                'risk_factors': ['Hospital procedures', 'Emergency care demand', 'Acute conditions']
            },
            'C5': {
                'code': 'N05B',
                'name': 'Anxiolytics',
                'common_drugs': ['Diazepam', 'Lorazepam', 'Alprazolam'],
                'conditions': ['Anxiety', 'Stress', 'Sleep disorders', 'Mental health'],
                'risk_factors': ['Economic stress', 'Urban living', 'Work pressure', 'Social issues']
            },
            'C6': {
                'code': 'N05C',
                'name': 'Hypnotics and sedatives',
                'common_drugs': ['Zolpidem', 'Zopiclone', 'Nitrazepam'],
                'conditions': ['Insomnia', 'Sleep disturbances', 'Anxiety-related sleep issues'],
                'risk_factors': ['Stress', 'Lifestyle changes', 'Mental health awareness', 'Aging']
            },
            'C7': {
                'code': 'R03',
                'name': 'Drugs for obstructive airway diseases',
                'common_drugs': ['Salbutamol', 'Beclometasone', 'Budesonide'],
                'conditions': ['Asthma', 'COPD', 'Bronchitis', 'Respiratory infections'],
                'risk_factors': ['Air pollution', 'Seasonal allergies', 'Monsoon', 'Dust', 'Pollen']
            },
            'C8': {
                'code': 'R06',
                'name': 'Antihistamines for systemic use',
                'common_drugs': ['Cetirizine', 'Loratadine', 'Chlorpheniramine'],
                'conditions': ['Allergies', 'Hay fever', 'Urticaria', 'Allergic rhinitis'],
                'risk_factors': ['Pollen season', 'Weather changes', 'Dust', 'Air quality', 'Climate']
            }
        }
        
        # Sri Lankan seasonal patterns
        self.seasons = {
            'monsoon': {'months': [5, 6, 10, 11], 'description': 'Monsoon season with high humidity'},
            'dry': {'months': [1, 2, 3, 7, 8], 'description': 'Dry season with dust and heat'},
            'inter_monsoon': {'months': [4, 9, 12], 'description': 'Inter-monsoon transition'}
        }
    
    def load_sales_data(self, category: str) -> pd.DataFrame:
        """Load sales data for a category"""
        file_path = os.path.join(self.base_path, f'{category}.csv')
        df = pd.read_csv(file_path)
        df['datum'] = pd.to_datetime(df['datum'])
        df = df.sort_values('datum')
        return df
    
    def calculate_statistics(self, df: pd.DataFrame, category: str, idx: int, window: int = 4) -> Dict:
        """Calculate statistics for context"""
        current_value = df[category].iloc[idx]
        
        # Historical context
        if idx >= window:
            previous_values = df[category].iloc[idx-window:idx].values
            avg_previous = np.mean(previous_values)
            change_pct = ((current_value - avg_previous) / avg_previous) * 100
            trend = 'increasing' if change_pct > 5 else ('decreasing' if change_pct < -5 else 'stable')
        else:
            avg_previous = current_value
            change_pct = 0
            trend = 'stable'
        
        # Date context
        date = df['datum'].iloc[idx]
        month = date.month
        season = self._get_season(month)
        
        return {
            'current_value': float(current_value),
            'previous_avg': float(avg_previous),
            'change_pct': float(change_pct),
            'trend': trend,
            'date': date.strftime('%Y-%m-%d'),
            'month': month,
            'season': season,
            'week_of_year': date.isocalendar()[1]
        }
    
    def _get_season(self, month: int) -> str:
        """Get season based on month"""
        for season, info in self.seasons.items():
            if month in info['months']:
                return season
        return 'inter_monsoon'
    
    def generate_training_example(self, category: str, stats: Dict) -> Dict:
        """Generate a single training example with instruction, input, and output"""
        
        cat_info = self.category_info[category]
        
        # Create instruction (consistent across all examples)
        instruction = (
            "You are a pharmaceutical analytics expert specializing in Sri Lankan healthcare. "
            "Analyze the drug sales prediction data and explain the reasons for the sales pattern. "
            "Provide insights for pharmacists, government officials, and public health awareness."
        )
        
        # Create input (the context/question)
        input_text = f"""Drug Sales Analysis Request:

Category: {category} ({cat_info['code']} - {cat_info['name']})
Current Sales: {stats['current_value']:.2f} units
Date: {stats['date']}
Trend: {stats['trend']} ({stats['change_pct']:+.1f}% change from previous weeks)
Season: {stats['season']}

Question: Why is this drug category showing this sales pattern? What should stakeholders be aware of?"""

        # Create output (the explanation) - this is what the model learns to generate
        output_text = self._generate_explanation(category, stats)
        
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        }
    
    def _generate_explanation(self, category: str, stats: Dict) -> str:
        """Generate detailed explanation based on category and statistics"""
        
        cat_info = self.category_info[category]
        
        # Determine primary factors based on trend and season
        factors = []
        
        # Factor 1: Seasonal influence
        if stats['season'] == 'monsoon':
            if category in ['C1', 'C2']:
                factors.append({
                    'title': 'Monsoon Season Impact',
                    'description': f"High humidity during monsoon season (85-90%) triggers increased {', '.join(cat_info['conditions'][:2])}. The damp weather exacerbates musculoskeletal conditions in Sri Lanka's aging population, particularly affecting urban areas like Colombo."
                })
            elif category in ['C7', 'C8']:
                factors.append({
                    'title': 'Respiratory Challenges',
                    'description': f"Monsoon brings increased air moisture and mold growth, triggering {', '.join(cat_info['conditions'][:2])} in susceptible populations. Pediatric cases rise by 30-40% during this period."
                })
            elif category in ['C3', 'C4']:
                factors.append({
                    'title': 'Viral Disease Prevalence',
                    'description': f"Monsoon season correlates with dengue and flu outbreaks in Sri Lanka, increasing demand for {', '.join(cat_info['common_drugs'][:2])} to manage fever and pain symptoms."
                })
        
        elif stats['season'] == 'dry':
            if category in ['C7', 'C8']:
                factors.append({
                    'title': 'Air Quality and Allergens',
                    'description': f"Dry season increases dust particles and pollen concentration, triggering {', '.join(cat_info['conditions'][:2])}. Agricultural burning in rural areas worsens air quality."
                })
        
        # Factor 2: Trend-based reasoning
        if stats['change_pct'] > 10:
            if category in ['C5', 'C6']:
                factors.append({
                    'title': 'Mental Health Awareness Growth',
                    'description': f"Rising demand (+{stats['change_pct']:.1f}%) reflects increased mental health awareness and reduced stigma. Recent government campaigns and COVID-19 aftermath have normalized seeking treatment for {', '.join(cat_info['conditions'][:2])}."
                })
            else:
                factors.append({
                    'title': 'Disease Outbreak Pattern',
                    'description': f"Significant increase (+{stats['change_pct']:.1f}%) suggests possible disease outbreak or health event requiring {cat_info['name']} medications. Requires immediate monitoring."
                })
        elif stats['change_pct'] < -10:
            factors.append({
                'title': 'Supply Chain or Policy Impact',
                'description': f"Decline (-{stats['change_pct']:.1f}%) may indicate supply chain disruptions, policy changes affecting availability, or shift to generic alternatives."
            })
        
        # Factor 3: Demographic and healthcare access
        factors.append({
            'title': 'Healthcare Access and Demographics',
            'description': f"Sri Lanka's aging population (15% over 60) and improved healthcare access through government clinics increase sustained demand for {', '.join(cat_info['common_drugs'][:2])}. Urban-rural disparities affect distribution patterns."
        })
        
        # Build the complete explanation
        explanation = f"""**Analysis of {cat_info['code']} ({cat_info['name']}) Sales Pattern**

"""
        
        # Add factors
        for i, factor in enumerate(factors[:3], 1):
            explanation += f"""**{i}. {factor['title']}**
{factor['description']}

"""
        
        # Public health implications
        explanation += f"""**🏥 Public Health Implications:**
- Monitor for potential overuse or misuse of {', '.join(cat_info['common_drugs'][:2])}
- Ensure adequate supply chain to prevent shortages
- Track adverse events and drug interactions
- Consider preventive healthcare campaigns targeting root causes

"""
        
        # Government recommendations
        explanation += f"""**🏛️ Government Action Items:**
- Maintain {abs(stats['change_pct']) + 15:.0f}% buffer stock in national pharmacies
- Deploy mobile health units to underserved areas
- Launch public awareness on proper medication use
- Coordinate with WHO for disease surveillance if outbreak suspected

"""
        
        # Pharmacy recommendations
        explanation += f"""**💊 Pharmacy Management:**
- Adjust inventory: Stock {stats['current_value'] * 1.2:.0f} units (20% safety margin)
- Prepare related categories (cross-sell opportunities)
- Train staff on patient consultation and OTC guidance
- Monitor expiry dates given {'increased' if stats['trend'] == 'increasing' else 'decreased'} turnover

"""
        
        # Risk awareness
        explanation += f"""**⚠️ Risk Awareness:**
- Patient side effects: Monitor for {self._get_side_effects(category)}
- Contraindications: Elderly patients, pregnancy, renal/hepatic impairment
- Drug interactions: Anticoagulants, other NSAIDs, antihypertensives
- Regulatory compliance: Ensure proper documentation for controlled substances
"""
        
        return explanation.strip()
    
    def _get_side_effects(self, category: str) -> str:
        """Get common side effects for category"""
        side_effects = {
            'C1': 'GI bleeding, renal impairment, cardiovascular events',
            'C2': 'GI upset, headache, dizziness',
            'C3': 'Reye syndrome (children), GI bleeding',
            'C4': 'Agranulocytosis, hypersensitivity',
            'C5': 'Drowsiness, dependence, respiratory depression',
            'C6': 'Daytime drowsiness, tolerance, dependence',
            'C7': 'Tachycardia, tremor, paradoxical bronchospasm',
            'C8': 'Drowsiness, dry mouth, dizziness'
        }
        return side_effects.get(category, 'various adverse effects')
    
    def generate_dataset(self, samples_per_category: int = 50) -> List[Dict]:
        """Generate complete training dataset from all categories"""
        
        dataset = []
        
        for category in self.categories:
            print(f"Processing {category}...")
            
            df = self.load_sales_data(category)
            
            # Sample random time points
            total_rows = len(df)
            if total_rows < samples_per_category:
                sample_indices = range(4, total_rows)  # Skip first 4 for context
            else:
                sample_indices = random.sample(range(4, total_rows), samples_per_category)
            
            for idx in sample_indices:
                stats = self.calculate_statistics(df, category, idx)
                example = self.generate_training_example(category, stats)
                dataset.append(example)
        
        print(f"\nGenerated {len(dataset)} training examples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset in formats for fine-tuning"""
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save as JSONL (for Hugging Face, OpenAI format)
        jsonl_path = output_path.replace('.json', '.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved dataset to:")
        print(f"   - {output_path}")
        print(f"   - {jsonl_path}")
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"Total examples: {len(dataset)}")
        print(f"Avg input length: {np.mean([len(ex['input']) for ex in dataset]):.0f} chars")
        print(f"Avg output length: {np.mean([len(ex['output']) for ex in dataset]):.0f} chars")
    
    def preview_examples(self, dataset: List[Dict], n: int = 2):
        """Preview sample training examples"""
        print("\n" + "="*80)
        print("SAMPLE TRAINING EXAMPLES")
        print("="*80)
        
        for i, example in enumerate(dataset[:n], 1):
            print(f"\n{'─'*80}")
            print(f"EXAMPLE {i}")
            print(f"{'─'*80}")
            print(f"\n📋 INSTRUCTION:\n{example['instruction']}")
            print(f"\n📊 INPUT:\n{example['input']}")
            print(f"\n💡 OUTPUT:\n{example['output']}")
            print(f"\n{'─'*80}")


def main():
    """Main execution function"""
    
    print("🚀 LLM Fine-Tuning Data Preparation Pipeline")
    print("="*80)
    
    # Initialize
    prep = LLMDataPreparation(base_path='')
    
    # Generate dataset (50 examples per category = 400 total)
    print("\n📊 Generating training dataset from CSV files...")
    dataset = prep.generate_dataset(samples_per_category=50)
    
    # Preview examples
    prep.preview_examples(dataset, n=2)
    
    # Save dataset
    output_path = 'llm_training_data/pharmaceutical_sales_explanations.json'
    prep.save_dataset(dataset, output_path)
    
    print("\n✅ Data preparation complete!")
    print("\nNext steps:")
    print("1. Run: python src/llm/medical_document_scraper.py")
    print("2. Run: python src/llm/fine_tune_llm.py")


if __name__ == "__main__":
    main()
