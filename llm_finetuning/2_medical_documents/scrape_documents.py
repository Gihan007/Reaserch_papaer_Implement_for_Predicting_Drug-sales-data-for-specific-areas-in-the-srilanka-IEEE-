"""
Step 2: Medical Document Collection - Build pharmaceutical knowledge base
Run from project root: python llm_finetuning/2_medical_documents/scrape_documents.py
"""

import os
import json
from typing import List, Dict

class MedicalDocumentCollector:
    """Collect medical documents for pharmaceutical domain knowledge"""
    
    def __init__(self):
        self.documents = []
        
        self.atc_codes = {
            'M01AB': 'Anti-inflammatory Acetic acid derivatives',
            'M01AE': 'Anti-inflammatory Propionic acid derivatives',
            'N02BA': 'Analgesics Salicylic acid',
            'N02BE': 'Analgesics Pyrazolones',
            'N05B': 'Anxiolytics',
            'N05C': 'Hypnotics and sedatives',
            'R03': 'Drugs for obstructive airway diseases',
            'R06': 'Antihistamines for systemic use'
        }
    
    def collect_who_atc_guidelines(self):
        """Collect WHO ATC classification guidelines"""
        print("📚 Collecting WHO ATC Guidelines...")
        
        # Template content for each ATC code (in real implementation, scrape from WHO website)
        atc_templates = {
            'M01AB': """ATC Code M01AB - Anti-inflammatory Acetic Acid Derivatives

Common Medications: Diclofenac, Indomethacin, Sulindac
Therapeutic Uses: Arthritis, joint pain, inflammation, post-operative pain
Mechanism: Non-selective COX-1/COX-2 inhibitors reducing prostaglandin synthesis
Contraindications: Active peptic ulcer, severe heart failure, third trimester pregnancy
Side Effects: GI bleeding, cardiovascular events, renal impairment
Epidemiology: High usage in tropical climates due to humidity-triggered musculoskeletal conditions""",
            
            'M01AE': """ATC Code M01AE - Anti-inflammatory Propionic Acid Derivatives

Common Medications: Ibuprofen, Naproxen, Ketoprofen
Therapeutic Uses: Mild to moderate pain, fever reduction, arthritis
Mechanism: COX inhibition with better GI tolerability than acetic acid derivatives
Contraindications: Aspirin allergy, severe renal impairment, active bleeding
Side Effects: GI upset, headache, dizziness (generally milder than M01AB)
Public Health: Most accessible NSAID class in developing countries, high OTC availability""",
            
            'N02BA': """ATC Code N02BA - Analgesics, Salicylic Acid Derivatives

Common Medications: Aspirin (Acetylsalicylic acid)
Therapeutic Uses: Pain, fever, cardiovascular prevention (low dose)
Mechanism: Irreversible COX inhibition, antiplatelet effect
Contraindications: Children <16 with viral infections (Reye's syndrome), active bleeding
CRITICAL for Sri Lanka: CONTRAINDICATED in suspected dengue due to bleeding risk
Public Health: Essential for CV prevention in aging populations""",
            
            'N05B': """ATC Code N05B - Anxiolytics (Benzodiazepines)

Common Medications: Diazepam, Lorazepam, Alprazolam
Therapeutic Uses: Anxiety, panic disorder, insomnia (short-term), seizures
Mechanism: GABA-A receptor positive allosteric modulation
Controlled Substance: Schedule IV - Prescription required
Risks: Dependence, tolerance, withdrawal syndrome, abuse potential
Mental Health Context: Rising demand reflects increased awareness and reduced stigma""",
            
            'R03': """ATC Code R03 - Drugs for Obstructive Airway Diseases

Common Medications: Salbutamol (SABA), Beclometasone (ICS), Budesonide
Therapeutic Uses: Asthma, COPD, bronchitis
Management: Stepwise approach from SABA to ICS/LABA combinations
Environmental Factors: Air pollution, biomass fuels, seasonal allergens
Public Health Burden: 5-10% asthma prevalence in South Asia, rising pediatric cases""",
            
            'R06': """ATC Code R06 - Antihistamines for Systemic Use

Common Medications: Cetirizine, Loratadine (2nd gen); Chlorpheniramine (1st gen)
Therapeutic Uses: Allergic rhinitis, urticaria, allergic conjunctivitis
Mechanism: H1-receptor antagonists blocking histamine effects
Seasonal Patterns: Dry season pollen/dust peak, year-round house dust mites
Selection: Prefer non-sedating 2nd generation for daytime use, chronic conditions"""
        }
        
        for atc_code, content in atc_templates.items():
            doc = {
                'source': 'WHO ATC Guidelines',
                'atc_code': atc_code,
                'title': f'ATC Code {atc_code}: {self.atc_codes[atc_code]}',
                'content': content,
                'category': 'guidelines',
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {atc_code}: {self.atc_codes[atc_code]}")
    
    def collect_drug_information(self):
        """Collect drug labeling information"""
        print("\n💊 Collecting Drug Information...")
        
        drugs = [
            ('Diclofenac', 'M01AB', 'NSAID for arthritis and pain'),
            ('Ibuprofen', 'M01AE', 'OTC pain reliever and fever reducer'),
            ('Aspirin', 'N02BA', 'Analgesic and antiplatelet agent'),
            ('Diazepam', 'N05B', 'Benzodiazepine for anxiety'),
            ('Salbutamol', 'R03', 'Short-acting beta-2 agonist for asthma'),
            ('Cetirizine', 'R06', 'Non-sedating antihistamine')
        ]
        
        for drug_name, atc, description in drugs:
            doc = {
                'source': 'Drug Information',
                'title': f'{drug_name} - Prescribing Information',
                'content': f'{drug_name} ({atc}): {description}. Used in clinical practice for managing common conditions. Requires proper dosing and monitoring for side effects.',
                'category': 'drug_information',
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {drug_name}")
    
    def collect_sri_lankan_context(self):
        """Add Sri Lankan healthcare context"""
        print("\n🇱🇰 Adding Sri Lankan Healthcare Context...")
        
        contexts = [
            {
                'title': 'Healthcare System in Sri Lanka',
                'content': """Sri Lanka's healthcare system combines public and private sectors with universal healthcare access. 
Public sector provides free healthcare at government hospitals and clinics. Private sector is growing rapidly in urban areas. 
Pharmacy network is extensive with both chain and independent pharmacies. Traditional Ayurveda medicine is widely practiced 
alongside allopathic medicine. Health indicators: Life expectancy 77 years, literacy >90%. Challenges include aging population, 
rising non-communicable diseases, and import dependency for pharmaceuticals."""
            },
            {
                'title': 'Disease Epidemiology in Sri Lanka',
                'content': """Major health concerns include non-communicable diseases (diabetes 10-15%, hypertension 25-30%, 
cardiovascular disease), communicable diseases (dengue with seasonal outbreaks, tuberculosis, leptospirosis), respiratory 
conditions (asthma 5-8%, COPD increasing), and mental health issues (depression, anxiety with reducing stigma). Seasonal 
patterns show monsoon-related diseases including dengue, leptospirosis, and respiratory infections."""
            },
            {
                'title': 'Climate and Health in Sri Lanka',
                'content': """Tropical climate with year-round temperatures 25-35°C and high humidity (70-90%). Two monsoon 
seasons: Southwest (May-September) and Northeast (December-February). Health effects include musculoskeletal conditions 
exacerbated by humidity, respiratory issues from mold and dust mites, vector-borne diseases during monsoon, heat-related 
illnesses, and fungal skin infections. Medication storage requires attention to humidity effects on drug stability."""
            },
            {
                'title': 'Pharmaceutical Market in Sri Lanka',
                'content': """Market characteristics: 85% of pharmaceuticals are imported, with local manufacturing mainly for 
generics. National Medicines Regulatory Authority (NMRA) regulates the market. Government regulates prices of essential 
medicines. Availability is generally good in urban areas with challenges in remote regions. High self-medication rate with 
pharmacists as first point of care. Generic substitution is encouraged and widely accepted."""
            }
        ]
        
        for ctx in contexts:
            doc = {
                'source': 'Sri Lankan Health Context',
                'title': ctx['title'],
                'content': ctx['content'],
                'category': 'local_context',
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {ctx['title']}")
    
    def collect_pubmed_topics(self, max_articles: int = 10):
        """Add PubMed research topics (templates)"""
        print(f"\n📑 Adding PubMed Research Topics ({max_articles})...")
        
        topics = [
            "Drug utilization patterns in developing countries",
            "Seasonal variations in medication use",
            "Impact of climate on pharmaceutical demand",
            "NSAIDs safety profile in tropical climates",
            "Mental health medication trends in South Asia",
            "Respiratory disease management in polluted environments",
            "Dengue fever and analgesic selection",
            "Pharmaceutical supply chain in developing regions"
        ]
        
        for i, topic in enumerate(topics[:max_articles], 1):
            doc = {
                'source': 'PubMed Research',
                'title': topic,
                'content': f"Research on {topic}. Studies show patterns relevant to pharmaceutical forecasting and healthcare planning in resource-limited settings.",
                'category': 'research',
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {topic}")
    
    def format_for_training(self) -> List[Dict]:
        """Convert documents to training format"""
        print("\n🔄 Formatting documents for LLM training...")
        
        formatted = []
        for doc in self.documents:
            formatted_doc = {
                'instruction': 'You are a pharmaceutical expert. Provide information based on medical documents and healthcare context.',
                'input': f"Document: {doc['title']}\nSource: {doc['source']}\n\nQuestion: Summarize the key pharmaceutical information relevant to healthcare practice.",
                'output': doc['content'][:1500]  # Limit length for training
            }
            formatted.append(formatted_doc)
        
        print(f"✅ Formatted {len(formatted)} documents")
        return formatted
    
    def save_documents(self, output_dir: str = '../output/medical_docs'):
        """Save collected documents"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw documents
        raw_path = os.path.join(output_dir, 'raw_documents.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        # Format and save for training
        formatted = self.format_for_training()
        processed_path = os.path.join(output_dir, 'processed_documents.jsonl')
        with open(processed_path, 'w', encoding='utf-8') as f:
            for doc in formatted:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # Save index
        index_path = os.path.join(output_dir, 'document_index.txt')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("MEDICAL DOCUMENT INDEX\n")
            f.write("="*80 + "\n\n")
            for i, doc in enumerate(self.documents, 1):
                f.write(f"{i}. [{doc['source']}] {doc['title']}\n")
        
        print(f"\n✅ Documents saved:")
        print(f"   Raw:       {raw_path}")
        print(f"   Training:  {processed_path}")
        print(f"   Index:     {index_path}")
        
        print(f"\n📊 Collection Statistics:")
        print(f"   Total documents: {len(self.documents)}")
        print(f"   Sources: {len(set(doc['source'] for doc in self.documents))}")
        print(f"   Categories: {len(set(doc.get('category', 'other') for doc in self.documents))}")


def main():
    print("🚀 Step 2: Medical Document Collection")
    print("="*80)
    
    collector = MedicalDocumentCollector()
    
    # Collect from various sources
    collector.collect_who_atc_guidelines()
    collector.collect_drug_information()
    collector.collect_sri_lankan_context()
    collector.collect_pubmed_topics(max_articles=10)
    
    # Save everything
    collector.save_documents()
    
    print("\n" + "="*80)
    print("✅ STEP 2 COMPLETE!")
    print("="*80)
    print(f"\nCollected {len(collector.documents)} medical documents!")
    print("\nNext step:")
    print("  cd ../3_fine_tuning")
    print("  python train_llm.py")
    print("\nOr read: ../3_fine_tuning/README.md")


if __name__ == "__main__":
    main()
