import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from forecast_utils import forecast_sales
import os
import json

app = Flask(__name__)

# Helper function to convert non-serializable types to JSON-serializable types
def make_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# ============================================
# HTML Page Routes
# ============================================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/forecast')
def forecast_page():
    """Forecasting page"""
    return render_template('forecast.html')

@app.route('/meta-learning')
def meta_learning_page():
    """Meta-learning page"""
    return render_template('meta-learning.html')

@app.route('/advanced')
def advanced_page():
    """Advanced AI (NAS & Federated) page"""
    return render_template('advanced.html')

@app.route('/causal')
def causal_page():
    """Causal analysis page"""
    return render_template('causal.html')

@app.route('/analytics')
def analytics_page():
    """Analytics dashboard page"""
    return render_template('analytics.html')

# ============================================
# API Endpoints
# ============================================

# Lazy loading for meta-learning system
meta_system = None

def get_meta_system():
    """Lazy load meta-learning system"""
    global meta_system
    if meta_system is None:
        try:
            print("Attempting to import MetaLearningSystem...")
            from src.models.meta_learning import MetaLearningSystem
            print("MetaLearningSystem imported successfully, initializing...")
            meta_system = MetaLearningSystem()
            print("MetaLearningSystem initialized successfully")
        except ImportError as e:
            print(f"ERROR: Could not import meta-learning module: {e}")
            import traceback
            traceback.print_exc()
            meta_system = None
        except Exception as e:
            print(f"ERROR: Could not initialize meta-learning system: {e}")
            import traceback
            traceback.print_exc()
            meta_system = None
    return meta_system

# API endpoint for forecast (JSON-based for frontend)
@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint for forecast that accepts JSON data"""
    try:
        data = request.get_json()
        category = data.get('category')
        date = data.get('date')
        model_type = data.get('model_type', 'ensemble')

        if not category or not date:
            return jsonify({
                'success': False,
                'error': 'Category and date are required'
            }), 400

        # Generate forecast value and plot based on inputs
        forecast_value, closest_prediction_date, plot_file, model_used = forecast_sales(category, date, model_type)

        # Ensure all values are JSON serializable
        return jsonify({
            'success': True,
            'forecast_value': float(forecast_value) if forecast_value is not None else 0.0,
            'closest_prediction_date': closest_prediction_date.strftime('%Y-%m-%d') if hasattr(closest_prediction_date, 'strftime') else str(closest_prediction_date),
            'plot_url': str(plot_file),
            'model_used': str(model_used),
            'category': str(category),
            'input_date': str(date)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Meta-learning endpoints
meta_system = None

@app.route('/api/meta-learning/train', methods=['POST'])
def train_meta_model():
    """Train meta-learning model across categories"""
    try:
        data = request.get_json()
        categories = data.get('categories', ['C1', 'C2', 'C3', 'C4'])
        
        print(f"Training MAML with categories: {categories}")

        meta_sys = get_meta_system()
        if meta_sys is None:
            print("ERROR: Meta-learning system failed to initialize")
            return jsonify({
                'status': 'error',
                'message': 'Meta-learning system not available. Check that src/models/meta_learning.py exists and dependencies are installed.'
            }), 500

        print("Meta-learning system initialized, starting training...")
        maml_model = meta_sys.train_maml(categories, n_epochs=5)
        print("MAML training completed successfully")

        return jsonify({
            'status': 'success',
            'message': 'Meta-learning model trained successfully',
            'categories_used': categories
        })

    except Exception as e:
        print(f"ERROR in train_meta_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/meta-learning/few-shot', methods=['POST'])
def few_shot_adaptation():
    """Perform few-shot adaptation to new category"""
    try:
        data = request.get_json()
        target_category = data.get('target_category')
        support_samples = data.get('support_samples', 10)
        adaptation_steps = data.get('adaptation_steps', 20)

        meta_sys = get_meta_system()
        if meta_sys is None:
            return jsonify({
                'status': 'error',
                'message': 'Meta-learning system not available'
            }), 500

        adapted_model, scaler = meta_sys.few_shot_adaptation(
            target_category, support_samples, adaptation_steps
        )

        return jsonify({
            'status': 'success',
            'message': f'Few-shot adaptation completed for {target_category}',
            'target_category': target_category,
            'support_samples': support_samples,
            'adaptation_steps': adaptation_steps
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/meta-learning/transfer', methods=['POST'])
def transfer_learning():
    """Perform transfer learning from source to target category"""
    try:
        data = request.get_json()
        source_category = data.get('source_category')
        target_category = data.get('target_category')
        fine_tune_steps = data.get('fine_tune_steps', 50)

        meta_sys = get_meta_system()
        if meta_sys is None:
            return jsonify({
                'status': 'error',
                'message': 'Meta-learning system not available'
            }), 500

        transfer_model, scaler = meta_sys.transfer_learning(
            source_category, target_category, fine_tune_steps
        )

        return jsonify({
            'status': 'success',
            'message': f'Transfer learning completed from {source_category} to {target_category}',
            'source_category': source_category,
            'target_category': target_category,
            'fine_tune_steps': fine_tune_steps
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/meta-learning/predict', methods=['POST'])
def meta_predict():
    """Make predictions using meta-learned models"""
    try:
        data = request.get_json()
        category = data.get('category')
        model_type = data.get('model_type', 'maml')  # 'maml', 'few_shot', 'transfer'

        meta_sys = get_meta_system()
        if meta_sys is None:
            return jsonify({
                'status': 'error',
                'message': 'Meta-learning system not available'
            }), 500

        forecast_value = meta_sys.predict_with_meta_model(category, model_type)

        return jsonify({
            'status': 'success',
            'forecast_value': forecast_value,
            'category': category,
            'model_type': model_type
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/meta-learning/status')
def meta_status():
    """Get meta-learning system status"""
    try:
        meta_sys = get_meta_system()
        status = {
            'initialized': meta_sys is not None,
            'maml_trained': meta_sys.maml_model is not None if meta_sys else False,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'initialized': False,
            'maml_trained': False,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
            'error': str(e)
        })

# Neural Architecture Search Routes
@app.route('/api/nas/search', methods=['POST'])
def nas_search():
    """Run Neural Architecture Search for a category"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        generations = data.get('generations', 3)

        from src.models.advanced.nas_drug_prediction import DrugPredictionNAS
        nas = DrugPredictionNAS()
        result = nas.search_optimal_architecture(category, generations)

        return jsonify({
            'success': True,
            'result': make_json_serializable(result),
            'message': f'NAS completed for {category}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/nas/batch_search', methods=['POST'])
def nas_batch_search():
    """Run NAS for multiple categories"""
    try:
        data = request.get_json()
        categories = data.get('categories', ['C1', 'C2', 'C3'])
        generations = data.get('generations', 2)

        from src.models.advanced.nas_drug_prediction import run_nas_for_all_categories
        results = run_nas_for_all_categories(categories, generations)

        return jsonify({
            'success': True,
            'results': make_json_serializable(results),
            'message': f'NAS completed for {len(results)} categories'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Federated Learning Routes
@app.route('/api/federated/train', methods=['POST'])
def federated_train():
    """Run federated learning training"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        num_clients = data.get('num_clients', 5)
        num_rounds = data.get('num_rounds', 8)
        distribution_type = data.get('distribution_type', 'iid')

        from src.models.advanced.federated_learning import run_federated_drug_prediction
        results = run_federated_drug_prediction(
            category=category,
            num_clients=num_clients,
            num_rounds=num_rounds,
            distribution_type=distribution_type
        )

        return jsonify({
            'success': True,
            'results': make_json_serializable(results),
            'message': f'Federated learning completed for {category}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/federated/compare', methods=['POST'])
def federated_compare():
    """Compare federated vs centralized learning"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')

        from src.models.advanced.federated_learning import compare_federated_vs_centralized
        comparison = compare_federated_vs_centralized(category)

        return jsonify({
            'success': True,
            'comparison': make_json_serializable(comparison),
            'message': f'Comparison completed for {category}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Advanced Features Status
@app.route('/api/advanced/status')
def advanced_status():
    """Get status of advanced features"""
    try:
        nas_available = False
        federated_available = False

        try:
            from src.models.advanced.nas_drug_prediction import DrugPredictionNAS
            nas_available = True
        except ImportError:
            pass

        try:
            from src.models.advanced.federated_learning import FederatedLearningSystem
            federated_available = True
        except ImportError:
            pass

        status = {
            'neural_architecture_search': nas_available,
            'federated_learning': federated_available,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        }

        return jsonify(status)
    except Exception as e:
        return jsonify({
            'neural_architecture_search': False,
            'federated_learning': False,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
            'error': str(e)
        })

# Causal Inference Routes
@app.route('/api/causal/discovery', methods=['POST'])
def causal_discovery():
    """Run causal discovery analysis"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        max_lags = data.get('max_lags', 5)

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.discover_causal_relationships(category, max_lags=max_lags)

        # Convert results to JSON serializable format
        results = make_json_serializable(results)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/causal/effects', methods=['POST'])
def causal_effects():
    """Estimate causal effects"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        treatment = data.get('treatment', 'sales_lag1')

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.estimate_causal_effects(category, treatment)

        # Convert results to JSON serializable format
        results = make_json_serializable(results)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/causal/counterfactual', methods=['POST'])
def causal_counterfactual():
    """Run counterfactual analysis"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        variable = data.get('variable', 'sales_lag1')
        change_percent = data.get('change_percent', 20)

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.counterfactual_analysis(category, variable, change_percent)

        # Convert results to JSON serializable format
        results = make_json_serializable(results)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/causal/complete', methods=['POST'])
def causal_complete():
    """Run complete causal analysis"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.complete_causal_analysis(category)

        # Convert results to JSON serializable format
        results = make_json_serializable(results)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ============================================
# LLM Explanation API
# ============================================
@app.route('/api/explain', methods=['POST'])
def api_explain():
    """Generate pharmaceutical explanation for prediction"""
    try:
        data = request.get_json()
        
        # Required fields
        category = data.get('category')
        prediction = data.get('prediction')
        
        if not category or prediction is None:
            return jsonify({
                'success': False,
                'error': 'Missing category or prediction'
            }), 400
        
        # Optional context
        date = data.get('date', '')
        model_type = data.get('model_type', 'Ensemble')
        
        # Parse date to get week/year
        week = 1
        year = 2025
        if date:
            try:
                from datetime import datetime
                dt = datetime.strptime(date, '%Y-%m-%d')
                week = dt.isocalendar()[1]
                year = dt.year
            except:
                pass
        
        # Generate explanation using template (no GPU needed)
        explanation = generate_pharmaceutical_explanation(
            category=category,
            prediction=float(prediction),
            week=week,
            year=year,
            model_type=model_type
        )
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'metadata': {
                'category': category,
                'prediction': prediction,
                'week': week,
                'year': year,
                'model_type': model_type
            }
        })
    
    except Exception as e:
        import traceback
        print(f"Error in /api/explain: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def generate_pharmaceutical_explanation(category, prediction, week, year, model_type='Ensemble'):
    """Generate detailed pharmaceutical explanation (template-based)"""
    
    category_info = {
        'C1': {
            'name': 'M01AB - Anti-inflammatory Acetic Acid Derivatives',
            'drugs': 'Diclofenac, Indomethacin',
            'uses': 'Arthritis, joint pain, inflammation'
        },
        'C2': {
            'name': 'M01AE - Anti-inflammatory Propionic Acid Derivatives',
            'drugs': 'Ibuprofen, Naproxen',
            'uses': 'Pain relief, fever reduction'
        },
        'C3': {
            'name': 'N02BA - Analgesics, Salicylic Acid',
            'drugs': 'Aspirin',
            'uses': 'Pain, fever, cardiovascular prevention'
        },
        'C4': {
            'name': 'N02BE - Analgesics, Pyrazolones',
            'drugs': 'Metamizole',
            'uses': 'Severe pain, fever'
        },
        'C5': {
            'name': 'N05B - Anxiolytics',
            'drugs': 'Diazepam, Lorazepam',
            'uses': 'Anxiety, panic disorder, insomnia'
        },
        'C6': {
            'name': 'N05C - Hypnotics and Sedatives',
            'drugs': 'Zolpidem, Zopiclone',
            'uses': 'Insomnia, sleep disorders'
        },
        'C7': {
            'name': 'R03 - Drugs for Obstructive Airway Diseases',
            'drugs': 'Salbutamol, Beclometasone',
            'uses': 'Asthma, COPD, bronchitis'
        },
        'C8': {
            'name': 'R06 - Antihistamines for Systemic Use',
            'drugs': 'Cetirizine, Loratadine',
            'uses': 'Allergies, rhinitis, urticaria'
        }
    }
    
    info = category_info.get(category, category_info['C1'])
    
    # Seasonal context
    season_context = ""
    if 10 <= week <= 20:
        season_context = "inter-monsoon transition period (March-April) with high humidity"
    elif 21 <= week <= 35:
        season_context = "Southwest monsoon season (May-September) with heavy rainfall"
    elif week <= 9 or week >= 48:
        season_context = "Northeast monsoon season (December-February)"
    else:
        season_context = "dry season with lower humidity"
    
    explanation = f"""**{info['name']} - Forecast Analysis**

**Predicted Sales:** {prediction:.2f} units  
**Time Period:** Week {week}, {year}  
**Model Used:** {model_type}  
**Common Medications:** {info['drugs']}  
**Therapeutic Use:** {info['uses']}

---

### 📊 ANALYSIS

**Seasonal Factors:**
• Week {week} corresponds to {season_context}
• Environmental conditions significantly influence medication demand
• Historical patterns show seasonal variation in pharmaceutical utilization

**Demographic Patterns:**
• Urban population health needs and healthcare accessibility
• Aging population with chronic conditions requiring medication management
• Socioeconomic factors affecting treatment-seeking behavior

**Public Health Context:**
• Healthcare system capacity and medication availability in Sri Lanka
• Awareness campaigns and disease surveillance programs
• Seasonal disease patterns (dengue, respiratory infections)

---

### ⚕️ CLINICAL CONSIDERATIONS

"""
    
    # Category-specific clinical info
    if category in ['C1', 'C2']:
        explanation += """**NSAIDs Safety Profile:**
⚠️ Monitor for gastrointestinal bleeding, especially in elderly patients
⚠️ Cardiovascular risk assessment before long-term use
⚠️ Renal function monitoring in patients with risk factors
⚠️ Avoid in third trimester pregnancy

**Drug Interactions:**
• Increased bleeding risk with anticoagulants
• Reduced effectiveness of antihypertensives
• Lithium toxicity risk
"""
    
    elif category == 'C3':
        explanation += """**Aspirin-Specific Precautions:**
⚠️ CRITICAL: Contraindicated in suspected dengue due to bleeding risk
⚠️ Reye's syndrome risk in children <16 with viral infections
⚠️ Check platelet count before prescribing during fever

**Cardiovascular Prevention:**
• Low-dose aspirin (75-100mg) for secondary prevention
• Assess bleeding vs. cardiovascular risk
"""
    
    elif category in ['C5', 'C6']:
        explanation += """**Controlled Substance Precautions:**
⚠️ Schedule IV drug - prescription required
⚠️ Risk of dependence and tolerance with prolonged use
⚠️ Gradual tapering required to prevent withdrawal
⚠️ Avoid in substance abuse history

**Mental Health Context:**
• Growing awareness reducing stigma
• Consider cognitive behavioral therapy alongside medication
• Monitor for depression and suicidal ideation
"""
    
    elif category == 'C7':
        explanation += """**Asthma/COPD Management:**
⚠️ Proper inhaler technique essential for efficacy
⚠️ Monitor for steroid side effects with long-term ICS use
⚠️ Air quality and pollution levels affect demand

**Stepwise Approach:**
• SABA for acute relief
• ICS for maintenance therapy
• LABA/ICS combination for severe cases
"""
    
    elif category == 'C8':
        explanation += """**Antihistamine Selection:**
• Prefer non-sedating 2nd generation for daytime use
• 1st generation for sleep aid but caution in elderly
• Safe in pregnancy (after first trimester)

**Allergy Management:**
• Seasonal allergens vary throughout year
• House dust mites year-round in tropical climate
• Consider environmental control measures
"""
    
    explanation += f"""
---

### 💡 RECOMMENDATIONS

**For Healthcare Providers:**
1. Ensure appropriate diagnosis before prescribing
2. Educate patients on proper use and potential side effects
3. Monitor for adverse effects during high-utilization periods
4. Consider non-pharmacological alternatives when appropriate
5. Document and report adverse drug reactions

**For Government/Health Authorities:**
1. Maintain adequate stock levels for predicted demand surge (+15-20% buffer)
2. Strengthen pharmacovigilance systems during peak seasons
3. Public awareness campaigns on safe medication use
4. Price monitoring to ensure affordability
5. Quality assurance of pharmaceutical supply chain

**For Public Awareness:**
1. Seek medical consultation for proper diagnosis
2. Follow prescribed dosing and duration
3. Report side effects: GI bleeding, chest pain, breathing difficulty
4. Store medications properly in tropical climate (cool, dry place)
5. Check expiration dates regularly
6. Do not share prescription medications

---

### 📈 Forecast Confidence

**Model Performance:** {model_type} model selected based on historical accuracy
**Prediction Reliability:** High confidence based on established seasonal patterns
**Uncertainty Factors:** Unexpected disease outbreaks, supply chain disruptions, policy changes

---

*This analysis combines pharmaceutical domain knowledge with AI forecasting. For clinical decisions, always consult qualified healthcare professionals.*
"""
    
    return explanation


# ============================================
# SHAP Explainability API
# ============================================
@app.route('/api/explainability', methods=['POST'])
def api_explainability():
    """Get SHAP-based model explainability"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        model_type = data.get('model_type', 'xgboost')
        
        # Import SHAP explainer
        from shap_explainer import get_model_explainability
        
        # Get explainability results
        results = get_model_explainability(category, model_type, base_path='')
        
        if results is None:
            return jsonify({
                'success': False,
                'error': f'Could not generate explainability for {category} using {model_type}'
            }), 500
        
        return jsonify({
            'success': True,
            'results': make_json_serializable(results)
        })
    
    except Exception as e:
        import traceback
        print(f"Error in /api/explainability: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/explainability')
def explainability_page():
    """Model explainability page"""
    return render_template('explainability.html')


if __name__ == '__main__':
    # Run the Flask app
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
