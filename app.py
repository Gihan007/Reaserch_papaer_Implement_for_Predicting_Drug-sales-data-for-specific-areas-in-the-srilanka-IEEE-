import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify
from forecast_utils import forecast_sales, generate_plot
import os
import json

app = Flask(__name__)

# Lazy loading for meta-learning system
meta_system = None

def get_meta_system():
    """Lazy load meta-learning system"""
    global meta_system
    if meta_system is None:
        try:
            from src.models.meta_learning import MetaLearningSystem
            meta_system = MetaLearningSystem()
        except ImportError as e:
            print(f"Warning: Could not import meta-learning: {e}")
            meta_system = None
    return meta_system

@app.route('/')
def index():
    return render_template('start.html')

# Home route to display form and results
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        # Get user inputs from the form
        category = request.form['category']
        date = request.form['date']
        model_type = request.form.get('model_type', 'ensemble')  # Default to ensemble

        input_date = pd.to_datetime(date)

        # Generate forecast value and plot based on inputs
        forecast_value, closest_prediction_date, plot_file, model_used = forecast_sales(category, date, model_type)

        # Render results back to the result template
        return render_template('result.html',
                               forecast_value=forecast_value,
                               closest_prediction_date=closest_prediction_date,
                               category=category,
                               user_input_date=input_date,
                               plot_url=f'/plot/{plot_file}',
                               model_used=model_used)

    return render_template('index.html')

# Route to serve plot images from the static folder
@app.route('/plot/<filename>')
def plot_image(filename):
    plot_path = os.path.join('static', filename)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return "Plot not found", 404

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/meta-learning')
def meta_learning():
    return render_template('meta_learning.html')

# Meta-learning endpoints
meta_system = None

@app.route('/api/meta-learning/train', methods=['POST'])
def train_meta_model():
    """Train meta-learning model across categories"""
    try:
        data = request.get_json()
        categories = data.get('categories', ['C1', 'C2', 'C3', 'C4'])

        meta_sys = get_meta_system()
        if meta_sys is None:
            return jsonify({
                'status': 'error',
                'message': 'Meta-learning system not available'
            }), 500

        maml_model = meta_sys.train_maml(categories, n_epochs=5)

        return jsonify({
            'status': 'success',
            'message': 'Meta-learning model trained successfully',
            'categories_used': categories
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
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
@app.route('/nas')
def nas_page():
    """Neural Architecture Search page"""
    return render_template('nas.html')

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
            'result': result,
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
            'results': results,
            'message': f'NAS completed for {len(results)} categories'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Federated Learning Routes
@app.route('/federated')
def federated_page():
    """Federated Learning page"""
    return render_template('federated.html')

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
            'results': results,
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
            'comparison': comparison,
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
@app.route('/causal')
def causal_page():
    """Causal Inference page"""
    return render_template('causal.html')

@app.route('/api/causal/discover', methods=['POST'])
def causal_discovery():
    """Run causal discovery analysis"""
    try:
        data = request.get_json()
        category = data.get('category', 'C1')
        max_lags = data.get('max_lags', 5)

        from src.models.advanced.causal_inference import CausalInferenceEngine
        engine = CausalInferenceEngine()

        results = engine.discover_causal_relationships(category, max_lags)

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

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
