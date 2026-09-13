import os
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT_DIR = Path(__file__).resolve().parents[2]

for import_path in (ROOT_DIR, ROOT_DIR / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from services.forecast_service.app.forecasting import forecast_sales, stacking_response_details
from libs.common.metrics import install_metrics
from libs.common.serialization import make_json_serializable

app = FastAPI()
install_metrics(app, "api-gateway")


@app.get('/health')
async def health():
    return {'status': 'healthy', 'service': 'api-gateway'}


@app.get('/')
async def root():
    return {
        'service': 'api-gateway',
        'status': 'healthy',
        'message': 'Frontend is served by the web-ui service.',
    }


@app.get('/api/branches')
async def api_branches():
    """Return a list of branch locations for Sri Lanka. Replace with DB/csv as needed."""
    # Example sample data; replace or load from a data source as required
    branches = [
        {"id": 1, "name": "Colombo Central Pharmacy", "lat": 6.9271, "lon": 79.8612, "address": "Colombo 01"},
        {"id": 2, "name": "Kandy Health Center", "lat": 7.2906, "lon": 80.6337, "address": "Kandy"},
        {"id": 3, "name": "Galle Medical Hub", "lat": 6.0535, "lon": 80.2210, "address": "Galle"},
        {"id": 4, "name": "Jaffna Pharmacy", "lat": 9.6615, "lon": 80.0255, "address": "Jaffna"},
        {"id": 5, "name": "Trincomalee Health Point", "lat": 8.5879, "lon": 81.2152, "address": "Trincomalee"},
        {"id": 6, "name": "Anuradhapura Care", "lat": 8.3114, "lon": 80.4037, "address": "Anuradhapura"},
        {"id": 7, "name": "Negombo Pharmacy", "lat": 7.2003, "lon": 79.8330, "address": "Negombo"},
        {"id": 8, "name": "Matara Medical Centre", "lat": 5.9481, "lon": 80.5350, "address": "Matara"},
        {"id": 9, "name": "Batticaloa Clinic", "lat": 7.7097, "lon": 81.6924, "address": "Batticaloa"},
        {"id": 10, "name": "Kurunegala Pharmacy", "lat": 7.4863, "lon": 80.3640, "address": "Kurunegala"},
        {"id": 11, "name": "Nuwara Eliya Health", "lat": 6.9707, "lon": 80.7820, "address": "Nuwara Eliya"},
        {"id": 12, "name": "Ratnapura Medical", "lat": 6.6828, "lon": 80.3991, "address": "Ratnapura"}
    ]
    return JSONResponse(content={"success": True, "branches": branches})


# ============================================
# API Endpoints (kept original logic; adapted to FastAPI request/response)
# ============================================

# Lazy loading for meta-learning system
meta_system = None

def get_meta_system():
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


@app.post('/api/forecast')
async def api_forecast(request: Request):
    try:
        data = await request.json()
        category = data.get('category')
        date = data.get('date')
        model_type = data.get('model_type', 'ensemble')

        if not category or not date:
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Category and date are required'})

        forecast_value, closest_prediction_date, plot_file, model_used = forecast_sales(category, date, model_type)

        return JSONResponse(content={
            'success': True,
            'forecast_value': float(forecast_value) if forecast_value is not None else 0.0,
            'closest_prediction_date': closest_prediction_date.strftime('%Y-%m-%d') if hasattr(closest_prediction_date, 'strftime') else str(closest_prediction_date),
            'plot_url': str(plot_file),
            'model_used': str(model_used),
            'category': str(category),
            'input_date': str(date),
            **(stacking_response_details(category, forecast_value, closest_prediction_date) if model_type == 'stacking' else {}),
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


# Meta-learning endpoints


@app.post('/api/meta-learning/train')
async def train_meta_model(request: Request):
    try:
        data = await request.json()
        categories = data.get('categories', ['C1', 'C2', 'C3', 'C4'])

        print(f"Training MAML with categories: {categories}")

        meta_sys = get_meta_system()
        if meta_sys is None:
            return JSONResponse(status_code=500, content={
                'status': 'error',
                'message': 'Meta-learning system not available. Check that src/models/meta_learning.py exists and dependencies are installed.'
            })

        print("Meta-learning system initialized, starting training...")
        maml_model = meta_sys.train_maml(categories, n_epochs=5)
        print("MAML training completed successfully")

        return JSONResponse(content={
            'status': 'success',
            'message': 'Meta-learning model trained successfully',
            'categories_used': categories
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        })


@app.post('/api/meta-learning/few-shot')
async def few_shot_adaptation(request: Request):
    try:
        data = await request.json()
        target_category = data.get('target_category')
        support_samples = data.get('support_samples', 10)
        adaptation_steps = data.get('adaptation_steps', 20)

        meta_sys = get_meta_system()
        if meta_sys is None:
            return JSONResponse(status_code=500, content={'status': 'error', 'message': 'Meta-learning system not available'})

        adapted_model, scaler = meta_sys.few_shot_adaptation(target_category, support_samples, adaptation_steps)

        return JSONResponse(content={
            'status': 'success',
            'message': f'Few-shot adaptation completed for {target_category}',
            'target_category': target_category,
            'support_samples': support_samples,
            'adaptation_steps': adaptation_steps
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post('/api/meta-learning/transfer')
async def transfer_learning(request: Request):
    try:
        data = await request.json()
        source_category = data.get('source_category')
        target_category = data.get('target_category')
        fine_tune_steps = data.get('fine_tune_steps', 50)

        meta_sys = get_meta_system()
        if meta_sys is None:
            return JSONResponse(status_code=500, content={'status': 'error', 'message': 'Meta-learning system not available'})

        transfer_model, scaler = meta_sys.transfer_learning(source_category, target_category, fine_tune_steps)

        return JSONResponse(content={
            'status': 'success',
            'message': f'Transfer learning completed from {source_category} to {target_category}',
            'source_category': source_category,
            'target_category': target_category,
            'fine_tune_steps': fine_tune_steps
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post('/api/meta-learning/predict')
async def meta_predict(request: Request):
    try:
        data = await request.json()
        category = data.get('category')
        model_type = data.get('model_type', 'maml')

        meta_sys = get_meta_system()
        if meta_sys is None:
            return JSONResponse(status_code=500, content={'status': 'error', 'message': 'Meta-learning system not available'})

        forecast_value = meta_sys.predict_with_meta_model(category, model_type)

        return JSONResponse(content={
            'status': 'success',
            'forecast_value': forecast_value,
            'category': category,
            'model_type': model_type
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.get('/api/meta-learning/status')
async def meta_status():
    try:
        meta_sys = get_meta_system()
        status = {
            'initialized': meta_sys is not None,
            'maml_trained': meta_sys.maml_model is not None if meta_sys else False,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        }
        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            'initialized': False,
            'maml_trained': False,
            'available_categories': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
            'error': str(e)
        })


# Neural Architecture Search Routes
@app.post('/api/nas/search')
async def nas_search(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        generations = data.get('generations', 3)

        from src.models.advanced.nas_drug_prediction import DrugPredictionNAS
        nas = DrugPredictionNAS()
        result = nas.search_optimal_architecture(category, generations)

        return JSONResponse(content={'success': True, 'result': make_json_serializable(result), 'message': f'NAS completed for {category}'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.post('/api/nas/batch_search')
async def nas_batch_search(request: Request):
    try:
        data = await request.json()
        categories = data.get('categories', ['C1', 'C2', 'C3'])
        generations = data.get('generations', 2)

        from src.models.advanced.nas_drug_prediction import run_nas_for_all_categories
        results = run_nas_for_all_categories(categories, generations)

        return JSONResponse(content={'success': True, 'results': make_json_serializable(results), 'message': f'NAS completed for {len(results)} categories'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


# Federated Learning Routes
@app.post('/api/federated/train')
async def federated_train(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        num_clients = data.get('num_clients', 5)
        num_rounds = data.get('num_rounds', 8)
        distribution_type = data.get('distribution_type', 'iid')

        from src.models.advanced.federated_learning import run_federated_drug_prediction
        results = run_federated_drug_prediction(category=category, num_clients=num_clients, num_rounds=num_rounds, distribution_type=distribution_type)

        return JSONResponse(content={'success': True, 'results': make_json_serializable(results), 'message': f'Federated learning completed for {category}'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.post('/api/federated/compare')
async def federated_compare(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')

        from src.models.advanced.federated_learning import compare_federated_vs_centralized
        comparison = compare_federated_vs_centralized(category)

        return JSONResponse(content={'success': True, 'comparison': make_json_serializable(comparison), 'message': f'Comparison completed for {category}'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.get('/api/advanced/status')
async def advanced_status():
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

        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(status_code=500, content={'neural_architecture_search': False, 'federated_learning': False, 'available_categories': ['C1','C2','C3','C4','C5','C6','C7','C8'], 'error': str(e)})


# Causal Inference Routes
@app.post('/api/causal/discovery')
async def causal_discovery(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        max_lags = data.get('max_lags', 5)

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.discover_causal_relationships(category, max_lags=max_lags)
        results = make_json_serializable(results)

        return JSONResponse(content={'success': True, 'results': results})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.post('/api/causal/effects')
async def causal_effects(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        treatment = data.get('treatment', 'sales_lag1')

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.estimate_causal_effects(category, treatment)
        results = make_json_serializable(results)

        return JSONResponse(content={'success': True, 'results': results})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.post('/api/causal/counterfactual')
async def causal_counterfactual(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        variable = data.get('variable', 'sales_lag1')
        change_percent = data.get('change_percent', 20)

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.counterfactual_analysis(category, variable, change_percent)
        results = make_json_serializable(results)

        return JSONResponse(content={'success': True, 'results': results})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


@app.post('/api/causal/complete')
async def causal_complete(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.complete_causal_analysis(category)
        results = make_json_serializable(results)

        return JSONResponse(content={'success': True, 'results': results})
    except Exception as e:
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


# LLM Explanation API
@app.post('/api/explain')
async def api_explain(request: Request):
    try:
        data = await request.json()

        category = data.get('category')
        prediction = data.get('prediction')

        if not category or prediction is None:
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Missing category or prediction'})

        date = data.get('date', '')
        model_type = data.get('model_type', 'Ensemble')

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

        explanation = generate_pharmaceutical_explanation(category=category, prediction=float(prediction), week=week, year=year, model_type=model_type)

        return JSONResponse(content={'success': True, 'explanation': explanation, 'metadata': {'category': category, 'prediction': prediction, 'week': week, 'year': year, 'model_type': model_type}})

    except Exception as e:
        import traceback
        print(f"Error in /api/explain: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})


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
3. Monitor for adverse effects during high-utilization periods
4. Consider non-pharmacological alternatives when appropriate
5. Document and report adverse drug reactions

---

### 📈 Forecast Confidence

**Model Performance:** {model_type} model selected based on historical accuracy
**Prediction Reliability:** High confidence based on established seasonal patterns
**Uncertainty Factors:** Unexpected disease outbreaks, supply chain disruptions, policy changes

---

*This analysis combines pharmaceutical domain knowledge with AI forecasting. For clinical decisions, always consult qualified healthcare professionals.*
"""

    return explanation


# PDF report + explanation endpoint
@app.post('/api/explain/report')
async def explain_report(request: Request):
    try:
        # reuse same parsing/validation as api_explain
        data = await request.json()
        category = data.get('category')
        prediction = data.get('prediction')
        if not category or prediction is None:
            return JSONResponse(status_code=400, content={'success': False, 'error': 'Missing category or prediction'})
        date = data.get('date', '')
        model_type = data.get('model_type', 'Ensemble')
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
        explanation = generate_pharmaceutical_explanation(category=category, prediction=float(prediction), week=week, year=year, model_type=model_type)

        # build PDF
        from io import BytesIO
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        text = c.beginText(40, 750)
        for line in explanation.split('\n'):
            text.textLine(line)
        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)
        return StreamingResponse(buffer, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="explanation.pdf"'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})

# SHAP Explainability API
@app.post('/api/explainability')
async def api_explainability(request: Request):
    try:
        data = await request.json()
        category = data.get('category', 'C1')
        model_type = data.get('model_type', 'xgboost')

        from services.explainability_service.app.shap_explainer import get_model_explainability
        results = get_model_explainability(category, model_type, base_path='')

        if results is None:
            return JSONResponse(status_code=500, content={'success': False, 'error': f'Could not generate explainability for {category} using {model_type}'})

        return JSONResponse(content={'success': True, 'results': make_json_serializable(results)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
