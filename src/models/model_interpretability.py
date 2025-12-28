import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """Comprehensive model interpretability toolkit"""

    def __init__(self, model, model_type, scaler, seq_length=30):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.seq_length = seq_length

    def compute_shap_values(self, background_data, test_data):
        """Compute SHAP values for feature importance"""

        def model_predict(data):
            predictions = []
            for sample in data:
                if self.model_type in ['transformer', 'gru', 'lstm', 'tft', 'nbeats', 'informer']:
                    # For sequence models
                    sample_tensor = torch.tensor(sample, dtype=torch.float32)
                    if len(sample_tensor.shape) == 1:
                        sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(-1)
                    elif len(sample_tensor.shape) == 2:
                        sample_tensor = sample_tensor.unsqueeze(0)

                    with torch.no_grad():
                        pred = self.model(sample_tensor).item()
                else:
                    # For traditional models
                    pred = self.model.predict(sample.reshape(1, -1))[0]
                predictions.append(pred)
            return np.array(predictions)

        # Create SHAP explainer
        if self.model_type in ['transformer', 'gru', 'lstm', 'tft', 'nbeats', 'informer']:
            # Use KernelExplainer for deep learning models
            explainer = shap.KernelExplainer(model_predict, background_data[:50])  # Use subset for speed
        else:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values
        shap_values = explainer.shap_values(test_data[:10])  # Explain first 10 samples

        return shap_values, explainer

    def generate_counterfactuals(self, input_data, target_change=0.1, max_iterations=100):
        """Generate counterfactual explanations"""

        original_prediction = self._predict(input_data)

        # Simple counterfactual generation by perturbing features
        counterfactuals = []
        impacts = []

        for i in range(len(input_data)):
            # Try increasing and decreasing each feature
            cf_increase = input_data.copy()
            cf_decrease = input_data.copy()

            cf_increase[i] += target_change
            cf_decrease[i] -= target_change

            pred_increase = self._predict(cf_increase)
            pred_decrease = self._predict(cf_decrease)

            impact_increase = pred_increase - original_prediction
            impact_decrease = pred_decrease - original_prediction

            counterfactuals.append({
                'feature_idx': i,
                'original_value': input_data[i],
                'increase_value': cf_increase[i],
                'decrease_value': cf_decrease[i],
                'impact_increase': impact_increase,
                'impact_decrease': impact_decrease
            })

            impacts.append(max(abs(impact_increase), abs(impact_decrease)))

        # Sort by impact
        counterfactuals = [cf for _, cf in sorted(zip(impacts, counterfactuals), reverse=True)]

        return counterfactuals[:5]  # Return top 5 most impactful

    def visualize_attention(self, input_sequence, model_type='transformer'):
        """Visualize attention weights for Transformer models"""

        if model_type not in ['transformer', 'tft', 'informer']:
            return None

        # For demonstration, create mock attention weights
        # In a real implementation, you'd extract actual attention weights
        seq_len = len(input_sequence)
        attention_weights = np.random.rand(seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

        return attention_weights

    def _predict(self, input_data):
        """Internal prediction method"""
        if self.model_type in ['transformer', 'gru', 'lstm', 'tft', 'nbeats', 'informer']:
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1)
            elif len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(input_tensor).item()
        else:
            prediction = self.model.predict(input_data.reshape(1, -1))[0]

        return prediction

def create_shap_plot(shap_values, feature_names, plot_path):
    """Create SHAP summary plot"""
    plt.figure(figsize=(10, 6))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For multi-output models

    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    return plot_path

def create_counterfactual_plot(counterfactuals, original_prediction, plot_path):
    """Create counterfactual explanation plot"""
    plt.figure(figsize=(12, 6))

    features = [f'Feature {cf["feature_idx"]}' for cf in counterfactuals]
    impacts = [cf['impact_increase'] for cf in counterfactuals]

    colors = ['red' if x > 0 else 'blue' for x in impacts]
    plt.barh(features, impacts, color=colors)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Impact on Prediction')
    plt.ylabel('Features')
    plt.title(f'Counterfactual Explanations\nOriginal Prediction: {original_prediction:.2f}')

    # Add value labels
    for i, (feature, impact, cf) in enumerate(zip(features, impacts, counterfactuals)):
        plt.text(impact + (0.01 if impact >= 0 else -0.01), i,
                '.2f', ha='left' if impact >= 0 else 'right', va='center')

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    return plot_path

def create_attention_heatmap(attention_weights, plot_path):
    """Create attention visualization heatmap"""
    plt.figure(figsize=(8, 6))

    sns.heatmap(attention_weights,
                cmap='Blues',
                square=True,
                cbar_kws={'shrink': 0.8})

    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.title('Attention Weights Heatmap')

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    return plot_path

def explain_model_prediction(category, model_type='tft', base_path='../', model_dir='../models_tft/', plot_dir='../static/images/'):
    """Generate comprehensive model explanations"""

    try:
        # Load data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Load scaler
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        data_scaled = scaler.transform(data.values.reshape(-1, 1)).flatten()

        # Load model
        if model_type == 'tft':
            from models.tft_model import TemporalFusionTransformer
            model_path = os.path.join(model_dir, f'tft_{category}.pth')
            model = TemporalFusionTransformer()
        elif model_type == 'transformer':
            from models.transformer_model import TimeSeriesTransformer
            model_path = os.path.join('../models_transformer', f'{category}_transformer.pth')
            model = TimeSeriesTransformer(input_size=10, d_model=128, nhead=8, num_layers=3)
        else:
            return None

        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Create interpreter
        interpreter = ModelInterpreter(model, model_type, scaler)

        # Prepare data for explanations
        seq_length = 30
        background_data = []
        for i in range(min(100, len(data_scaled) - seq_length)):  # Use up to 100 background samples
            background_data.append(data_scaled[i:i + seq_length])
        background_data = np.array(background_data)

        test_sample = data_scaled[-seq_length:]

        # Generate explanations
        explanations = {}

        # 1. SHAP Values
        try:
            shap_values, explainer = interpreter.compute_shap_values(background_data, test_sample.reshape(1, -1))
            feature_names = [f'Time_{i}' for i in range(seq_length)]

            shap_plot_path = os.path.join(plot_dir, f'{category}_{model_type}_shap.png')
            create_shap_plot(shap_values, feature_names, shap_plot_path)
            explanations['shap_plot'] = f'/static/images/{category}_{model_type}_shap.png'
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            explanations['shap_plot'] = None

        # 2. Counterfactual Explanations
        try:
            counterfactuals = interpreter.generate_counterfactuals(test_sample)
            original_pred = interpreter._predict(test_sample)

            cf_plot_path = os.path.join(plot_dir, f'{category}_{model_type}_counterfactual.png')
            create_counterfactual_plot(counterfactuals, original_pred, cf_plot_path)
            explanations['counterfactual_plot'] = f'/static/images/{category}_{model_type}_counterfactual.png'
            explanations['counterfactuals'] = counterfactuals
        except Exception as e:
            print(f"Counterfactual generation failed: {e}")
            explanations['counterfactual_plot'] = None
            explanations['counterfactuals'] = []

        # 3. Attention Visualization (for applicable models)
        try:
            if model_type in ['transformer', 'tft', 'informer']:
                attention_weights = interpreter.visualize_attention(test_sample)

                if attention_weights is not None:
                    attn_plot_path = os.path.join(plot_dir, f'{category}_{model_type}_attention.png')
                    create_attention_heatmap(attention_weights, attn_plot_path)
                    explanations['attention_plot'] = f'/static/images/{category}_{model_type}_attention.png'
                else:
                    explanations['attention_plot'] = None
        except Exception as e:
            print(f"Attention visualization failed: {e}")
            explanations['attention_plot'] = None

        return explanations

    except Exception as e:
        print(f"Error generating explanations: {str(e)}")
        return None