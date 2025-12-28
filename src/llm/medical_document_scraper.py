"""
Medical Document Collection Pipeline
Scrapes and processes medical documents for LLM fine-tuning knowledge base
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict
import re
from urllib.parse import urljoin
import hashlib

class MedicalDocumentScraper:
    """Collect medical documents from various sources for pharmaceutical domain knowledge"""
    
    def __init__(self, output_dir='llm_training_data/medical_documents'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ATC categories we're interested in
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
        
        self.documents = []
    
    def scrape_who_atc_guidelines(self):
        """Scrape WHO ATC classification guidelines"""
        print("📚 Scraping WHO ATC Guidelines...")
        
        # WHO ATC classification system documentation
        who_base_url = "https://www.whocc.no/atc_ddd_index/"
        
        for atc_code, description in self.atc_codes.items():
            try:
                # Simulate WHO document (in real implementation, use actual WHO API/website)
                doc = {
                    'source': 'WHO ATC Guidelines',
                    'atc_code': atc_code,
                    'title': f'ATC Code {atc_code}: {description}',
                    'content': self._generate_atc_content(atc_code, description),
                    'url': f'{who_base_url}?code={atc_code}',
                    'category': 'guidelines',
                    'language': 'en'
                }
                self.documents.append(doc)
                print(f"  ✓ {atc_code}: {description}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  ✗ Error fetching {atc_code}: {e}")
    
    def _generate_atc_content(self, atc_code: str, description: str) -> str:
        """Generate ATC classification content (placeholder - replace with actual scraping)"""
        
        content_templates = {
            'M01AB': """
ATC Code M01AB - Anti-inflammatory and Antirheumatic Products, Non-steroids, Acetic Acid Derivatives

Classification: Musculo-skeletal system (M) > Anti-inflammatory and antirheumatic products (M01) > 
Non-steroids (M01A) > Acetic acid derivatives and related substances (M01AB)

Common Medications:
- Diclofenac (M01AB05): Widely used NSAID for pain and inflammation
- Indomethacin (M01AB01): Potent NSAID for arthritis and gout
- Sulindac (M01AB02): Used for arthritis and acute musculoskeletal disorders

Therapeutic Indications:
- Rheumatoid arthritis
- Osteoarthritis
- Ankylosing spondylitis
- Acute gout
- Post-operative pain
- Musculoskeletal injuries

Mechanism of Action:
Non-selective COX-1 and COX-2 inhibitors that reduce prostaglandin synthesis, thereby decreasing 
inflammation, pain, and fever.

Contraindications:
- Active peptic ulcer disease
- Severe heart failure
- History of GI bleeding
- Third trimester of pregnancy
- Hypersensitivity to NSAIDs

Side Effects:
- Gastrointestinal: Dyspepsia, ulceration, bleeding (most common)
- Cardiovascular: Increased risk of MI and stroke with long-term use
- Renal: Fluid retention, acute kidney injury
- Hepatic: Elevated liver enzymes
- Hematologic: Increased bleeding time

Special Populations:
- Elderly: Increased risk of GI and renal adverse effects
- Pregnancy: Avoid in third trimester (risk of premature closure of ductus arteriosus)
- Renal impairment: Dose adjustment required

Drug Interactions:
- Anticoagulants: Increased bleeding risk
- Antihypertensives: Reduced efficacy
- Methotrexate: Increased toxicity
- Lithium: Increased plasma levels

Epidemiology:
High utilization in tropical climates due to musculoskeletal conditions exacerbated by humidity 
and physical labor. Seasonal peaks observed during monsoon seasons.
""",
            'M01AE': """
ATC Code M01AE - Anti-inflammatory and Antirheumatic Products, Propionic Acid Derivatives

Classification: Musculo-skeletal system (M) > Anti-inflammatory and antirheumatic products (M01) > 
Non-steroids (M01A) > Propionic acid derivatives (M01AE)

Common Medications:
- Ibuprofen (M01AE01): Most commonly used OTC NSAID globally
- Naproxen (M01AE02): Longer half-life, twice-daily dosing
- Ketoprofen (M01AE03): Potent analgesic and anti-inflammatory

Therapeutic Indications:
- Mild to moderate pain (headache, dental, dysmenorrhea)
- Fever reduction
- Rheumatoid arthritis
- Osteoarthritis
- Soft tissue injuries

Pharmacokinetics:
- Ibuprofen: Rapid absorption, peak plasma 1-2 hours, t½ 2 hours
- Naproxen: Slower absorption, peak 2-4 hours, t½ 12-17 hours
- Well absorbed orally, highly protein-bound (>99%)

Dosing:
- Ibuprofen: 200-400mg every 4-6 hours (max 1200mg/day OTC)
- Naproxen: 250-500mg twice daily
- Ketoprofen: 50-100mg twice daily

Safety Profile:
Generally better GI tolerability than acetic acid derivatives. Lower CV risk compared to COX-2 
selective inhibitors. Suitable for short-term use in general population.

Public Health Considerations:
Most accessible NSAID class in developing countries. High OTC availability contributes to 
self-medication practices. Education needed on proper dosing and duration.
""",
            'N02BA': """
ATC Code N02BA - Analgesics, Salicylic Acid and Derivatives

Classification: Nervous system (N) > Analgesics (N02) > Other analgesics and antipyretics (N02B) > 
Salicylic acid and derivatives (N02BA)

Primary Medication:
- Aspirin/Acetylsalicylic acid (N02BA01): One of the oldest and most widely used drugs

Multiple Therapeutic Roles:
1. Analgesic: Mild to moderate pain
2. Antipyretic: Fever reduction
3. Anti-inflammatory: High doses for rheumatic conditions
4. Antiplatelet: Low dose (75-150mg) for cardiovascular prevention

Mechanism:
Irreversible inhibition of COX enzymes, particularly COX-1, reducing thromboxane A2 production 
(basis for antiplatelet effect).

Special Warnings:
- Reye's Syndrome: Contraindicated in children <16 years with viral infections
- Bleeding risk: Due to antiplatelet effect lasting 7-10 days
- Asthma exacerbation: In aspirin-sensitive individuals

Dengue Fever Context (Critical for Sri Lanka):
CONTRAINDICATED in suspected dengue due to:
- Increased bleeding risk in thrombocytopenic patients
- Potential hepatotoxicity
- Acetaminophen recommended instead

Public Health Impact:
Essential medicine for cardiovascular prevention in aging populations. Requires pharmacovigilance 
for appropriate use, especially during dengue seasons in tropical regions.
""",
            'N05B': """
ATC Code N05B - Anxiolytics (Psycholeptics)

Classification: Nervous system (N) > Psycholeptics (N05) > Anxiolytics (N05B)

Benzodiazepines (Primary Class):
- Diazepam (N05BA01): Long-acting, broad spectrum
- Lorazepam (N05BA06): Intermediate-acting, less active metabolites
- Alprazolam (N05BA12): Short-acting, higher potency

Therapeutic Uses:
- Generalized anxiety disorder
- Panic disorder
- Acute stress reactions
- Insomnia (short-term)
- Alcohol withdrawal
- Seizure disorders (diazepam)

Mechanism: 
GABA-A receptor positive allosteric modulation, enhancing inhibitory neurotransmission.

Controlled Substance Status:
Schedule IV (most countries) - Requires prescription and monitoring due to:
- Dependence potential (physical and psychological)
- Tolerance development
- Withdrawal syndrome
- Abuse liability

Treatment Guidelines:
- Short-term use preferred (2-4 weeks for anxiety, 7-10 days for insomnia)
- Gradual tapering required for discontinuation
- Cognitive-behavioral therapy as first-line for anxiety disorders

Mental Health Context:
Rising demand in developing countries reflects:
- Increased mental health awareness and reduced stigma
- Economic stressors and urbanization
- Post-pandemic mental health crisis
- Improved access to psychiatric services

Regulatory Concerns:
Balance between access for legitimate medical need and prevention of misuse. Pharmacist counseling 
essential for safe use.
""",
            'R03': """
ATC Code R03 - Drugs for Obstructive Airway Diseases

Classification: Respiratory system (R) > Drugs for obstructive airway diseases (R03)

Major Subclasses:
R03A - Adrenergics, inhalants (Beta-2 agonists)
R03B - Other drugs for obstructive airway diseases, inhalants (Corticosteroids)
R03C - Adrenergics for systemic use
R03D - Other systemic drugs for obstructive airway diseases

Common Medications:
- Salbutamol/Albuterol (R03AC02): Short-acting beta-2 agonist (SABA)
- Beclometasone (R03BA01): Inhaled corticosteroid (ICS)
- Budesonide (R03BA02): ICS with high pulmonary deposition

Asthma Management Stepwise Approach:
Step 1: As-needed SABA
Step 2: Low-dose ICS + as-needed SABA
Step 3: Low-dose ICS/LABA combination
Step 4: Medium/high-dose ICS/LABA
Step 5: Add-on therapy (LAMA, biologics)

COPD Management:
- GOLD A: Bronchodilator (SABA or LAMA)
- GOLD B: Long-acting bronchodilator (LABA or LAMA)
- GOLD C: LAMA or ICS/LABA
- GOLD D: LAMA + LABA ± ICS

Environmental Factors in Tropics:
- Air pollution: Biomass fuel combustion, vehicle emissions
- Allergens: Mold, dust mites thrive in humid conditions
- Seasonal variations: Monsoon affects air quality and triggers
- Occupational exposures: Agriculture, textiles

Public Health Burden:
Asthma prevalence: 5-10% in South Asia
COPD underdiagnosis common
Rising pediatric asthma rates
Climate change impact on respiratory diseases
""",
            'R06': """
ATC Code R06 - Antihistamines for Systemic Use

Classification: Respiratory system (R) > Antihistamines for systemic use (R06)

First-Generation (Sedating):
- Chlorpheniramine (R06AB04): Widely available, low cost
- Diphenhydramine (R06AA02): Strong sedative effect
- Promethazine (R06AD02): Antiemetic properties

Second-Generation (Non-sedating):
- Cetirizine (R06AE07): Once-daily, minimal CNS penetration
- Loratadine (R06AX13): Non-sedating, OTC available
- Fexofenadine (R06AX26): No cardiac effects, truly non-sedating

Mechanism:
H1-receptor antagonists that block histamine effects:
- Reduced vasodilation and vascular permeability
- Decreased pruritus
- Reduced mucus secretion

Indications:
- Allergic rhinitis (seasonal and perennial)
- Urticaria and angioedema
- Allergic conjunctivitis
- Pruritus
- Anaphylaxis adjunct

Seasonal Pattern in Sri Lanka:
- Dry season (January-March): Pollen and dust peak
- Monsoon transitions: Mold spore increase
- Year-round: House dust mites due to humidity
- Urban air pollution: Persistent allergen exposure

First vs. Second Generation Selection:
First-generation: 
  Pros - Lower cost, effective for acute urticaria
  Cons - Sedation, anticholinergic effects, cognitive impairment
  
Second-generation:
  Pros - No sedation, once-daily dosing, suitable for chronic use
  Cons - Higher cost
  
Preferred: Second-generation for daytime use, chronic conditions, occupation requiring alertness

Safety Considerations:
- Driving impairment with first-generation
- Elderly: Avoid first-generation (fall risk, delirium)
- Pregnancy: Loratadine, cetirizine considered safer
- Children: Dose adjustment required, avoid <2 years for first-generation
"""
        }
        
        return content_templates.get(atc_code, f"Content for {atc_code} - {description}")
    
    def scrape_pubmed_articles(self, max_articles: int = 20):
        """Scrape relevant PubMed articles (simplified version)"""
        print(f"\n📑 Collecting PubMed articles (max {max_articles})...")
        
        # In real implementation, use NCBI E-utilities API
        # For now, create template articles based on common topics
        
        topics = [
            ("Drug utilization patterns in developing countries", "pharmaceutical_epidemiology"),
            ("Seasonal variations in medication use", "seasonal_patterns"),
            ("Impact of climate on pharmaceutical demand", "environmental_factors"),
            ("NSAIDs safety profile in tropical climates", "drug_safety"),
            ("Mental health medication trends in South Asia", "mental_health"),
            ("Respiratory disease management in polluted environments", "respiratory_health"),
            ("Antibiotic resistance and self-medication practices", "antimicrobial_stewardship"),
            ("Dengue fever and analgesic selection", "tropical_diseases")
        ]
        
        for title, topic in topics[:max_articles]:
            doc = {
                'source': 'PubMed',
                'title': title,
                'content': f"Research article on {title}. [Placeholder for actual PubMed content]",
                'category': topic,
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {title}")
    
    def scrape_fda_drug_labels(self):
        """Scrape FDA drug labeling information"""
        print("\n💊 Collecting FDA Drug Labels...")
        
        # Common drugs in our categories
        drugs = [
            'Diclofenac', 'Ibuprofen', 'Naproxen', 'Aspirin',
            'Diazepam', 'Lorazepam', 'Salbutamol', 'Cetirizine'
        ]
        
        for drug in drugs:
            doc = {
                'source': 'FDA Drug Labels',
                'title': f'{drug} - Prescribing Information',
                'content': f"FDA-approved labeling information for {drug}. [Placeholder for actual FDA content]",
                'category': 'drug_information',
                'language': 'en'
            }
            self.documents.append(doc)
            print(f"  ✓ {drug}")
    
    def add_sri_lankan_health_context(self):
        """Add Sri Lankan specific healthcare context"""
        print("\n🇱🇰 Adding Sri Lankan Healthcare Context...")
        
        contexts = [
            {
                'title': 'Healthcare System in Sri Lanka',
                'content': """
Sri Lanka's healthcare system combines public and private sectors with universal healthcare access.
- Public sector: Free healthcare at government hospitals and clinics
- Private sector: Growing rapidly, especially in urban areas
- Pharmacy network: Extensive, both chain and independent pharmacies
- Traditional medicine: Ayurveda widely practiced alongside allopathic medicine
- Health indicators: Life expectancy 77 years, literacy rate >90%
- Challenges: Aging population, rising NCDs, import dependency for drugs
"""
            },
            {
                'title': 'Disease Epidemiology in Sri Lanka',
                'content': """
Major health concerns:
- Non-communicable diseases: Diabetes (10-15%), hypertension (25-30%), cardiovascular disease
- Communicable diseases: Dengue (seasonal outbreaks), tuberculosis, leptospirosis
- Respiratory: Asthma (5-8%), COPD (increasing with air pollution)
- Mental health: Depression, anxiety (stigma reducing, treatment-seeking increasing)
- Seasonal patterns: Monsoon-related diseases (dengue, leptospirosis, respiratory infections)
"""
            },
            {
                'title': 'Pharmaceutical Market in Sri Lanka',
                'content': """
Market characteristics:
- Import-dependent: 85% of pharmaceuticals imported
- Local manufacturing: Generic drugs, limited API production
- Regulatory: National Medicines Regulatory Authority (NMRA)
- Pricing: Government regulates prices of essential medicines
- Availability: Generally good in urban areas, challenges in remote regions
- OTC culture: High self-medication rate, pharmacists as first point of care
- Generic substitution: Encouraged by government, high acceptance
"""
            },
            {
                'title': 'Climate and Health in Sri Lanka',
                'content': """
Tropical climate impacts:
- Temperature: 25-35°C year-round, high humidity (70-90%)
- Monsoons: Southwest (May-September), Northeast (December-February)
- Health effects:
  * Musculoskeletal: Humidity exacerbates arthritis, joint pain
  * Respiratory: Mold, dust mites thrive; air quality varies
  * Vector-borne: Mosquito breeding in monsoon, dengue peaks
  * Heat-related: Dehydration, heat exhaustion during dry season
  * Skin conditions: Fungal infections common in humid conditions
- Medication storage: Humidity affects drug stability, cold chain challenges
"""
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
    
    def process_documents_for_training(self) -> List[Dict]:
        """Convert documents into training format for fine-tuning"""
        print("\n🔄 Processing documents for LLM training...")
        
        processed = []
        
        for doc in self.documents:
            # Create instruction-input-output format
            processed_doc = {
                'instruction': 'You are a pharmaceutical expert. Provide information based on the following medical document.',
                'input': f"Document: {doc['title']}\nSource: {doc['source']}\n\nQuestion: Summarize the key information from this document relevant to pharmaceutical practice.",
                'output': doc['content'][:2000]  # Limit length
            }
            processed.append(processed_doc)
        
        print(f"✅ Processed {len(processed)} documents")
        return processed
    
    def save_documents(self):
        """Save collected documents"""
        
        # Save raw documents
        raw_path = os.path.join(self.output_dir, 'raw_documents.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        # Save processed for training
        processed = self.process_documents_for_training()
        processed_path = os.path.join(self.output_dir, 'processed_documents.jsonl')
        with open(processed_path, 'w', encoding='utf-8') as f:
            for doc in processed:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Saved documents:")
        print(f"   - Raw: {raw_path}")
        print(f"   - Processed: {processed_path}")
        
        return processed_path


def main():
    """Main execution"""
    print("🌐 Medical Document Collection Pipeline")
    print("="*80)
    
    scraper = MedicalDocumentScraper()
    
    # Collect from various sources
    scraper.scrape_who_atc_guidelines()
    scraper.scrape_pubmed_articles(max_articles=10)
    scraper.scrape_fda_drug_labels()
    scraper.add_sri_lankan_health_context()
    
    # Save all documents
    processed_path = scraper.save_documents()
    
    print(f"\n✅ Document collection complete!")
    print(f"Total documents: {len(scraper.documents)}")
    print(f"\nNext step: Combine with sales data and fine-tune LLM")
    print(f"Run: python src/llm/fine_tune_llm.py")


if __name__ == "__main__":
    main()
