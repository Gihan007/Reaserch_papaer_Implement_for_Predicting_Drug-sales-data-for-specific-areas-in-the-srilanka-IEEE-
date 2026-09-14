"""
SHAP Explainability Module
Provides feature importance and model explanations using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_ARTIFACT_DIR = ROOT_DIR / "artifacts" / "models"
SHAP_IMAGE_DIR = ROOT_DIR / "services" / "frontend_service" / "app" / "static" / "images" / "shap"


def _load_shap():
    import shap

    return shap


def _category_csv_path(category, base_path=''):
    if base_path:
        candidate = Path(base_path) / f'{category}.csv'
        if candidate.exists():
            return candidate

    return DATA_DIR / f'{category}.csv'


def _model_path(model_family, filename, base_path=''):
    if base_path:
        candidate = Path(base_path) / "artifacts" / "models" / model_family / filename
        if candidate.exists():
            return candidate

        candidate = Path(base_path) / model_family / filename
        if candidate.exists():
            return candidate

    return MODEL_ARTIFACT_DIR / model_family / filename

class SHAPExplainer:
    """Generates SHAP explanations for pharmaceutical forecasting models"""
    
    def __init__(self, category, base_path=''):
        self.category = category
        self.base_path = base_path
        self.explainers = {}
        self.feature_names = None
        
    def create_features(self, df, n_lags=5):
        """Create lagged features for SHAP analysis - matching XGBoost training"""
        data = []
        targets = []
        
        for i in range(n_lags, len(df)):
            # Lagged sales values only (reversed order - most recent first)
            lags = df[self.category].iloc[i-n_lags:i].values[::-1]
            data.append(lags)
            targets.append(df[self.category].iloc[i])
        
        # Feature names - lags in reverse order
        self.feature_names = [f'sales_lag_{i+1}' for i in range(n_lags)]
        
        return np.array(data), np.array(targets)
    
    def explain_xgboost(self, model_path=None, n_samples=100):
        """Generate SHAP explanation for XGBoost model"""
        try:
            import xgboost as xgb
            import pickle
            shap = _load_shap()
            
            # Load model
            if model_path is None:
                model_path = _model_path('models_xgb', f'{self.category}_xgb.pkl', self.base_path)
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return None
            
            # Load pickled model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load data and create features
            csv_path = _category_csv_path(self.category, self.base_path)
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            X, y = self.create_features(df, n_lags=5)
            
            # Use last n_samples for SHAP analysis
            X_sample = X[-n_samples:]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Store explainer
            self.explainers['xgboost'] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'X': X_sample,
                'feature_names': self.feature_names
            }
            
            return shap_values
            
        except Exception as e:
            print(f"Error in XGBoost SHAP: {e}")
            return None
    
    def explain_lightgbm(self, model_path=None, n_samples=100):
        """Generate SHAP explanation for LightGBM model"""
        try:
            shap = _load_shap()

            # The checked-in native LightGBM text artifacts are not portable
            # across all LightGBM builds. Match the live forecast path by
            # fitting a small local model from the bundled category series.
            csv_path = _category_csv_path(self.category, self.base_path)
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            from src.models.lightgbm_model import train_lightgbm_model

            values = df[self.category].to_numpy(dtype=float)
            model, scaler = train_lightgbm_model(values, n_lags=5)
            scaled = scaler.transform(values.reshape(-1, 1)).ravel()
            scaled_frame = pd.DataFrame({self.category: scaled}, index=df.index)
            X, y = self.create_features(scaled_frame, n_lags=5)
            
            # Use last n_samples for SHAP analysis
            X_sample = X[-n_samples:]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Store explainer
            self.explainers['lightgbm'] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'X': X_sample,
                'feature_names': self.feature_names
            }
            
            return shap_values
            
        except Exception as e:
            print(f"Error in LightGBM SHAP: {e}")
            return None
    
    def get_feature_importance(self, model_type='xgboost'):
        """Get feature importance ranking from SHAP values"""
        try:
            if model_type not in self.explainers:
                # Try to generate explanation
                if model_type == 'xgboost':
                    self.explain_xgboost()
                elif model_type == 'lightgbm':
                    self.explain_lightgbm()
            
            if model_type not in self.explainers:
                return None
            
            shap_values = self.explainers[model_type]['shap_values']
            feature_names = self.explainers[model_type]['feature_names']
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create importance dictionary
            importance = {}
            for name, value in zip(feature_names, mean_abs_shap):
                importance[name] = float(value)
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None
    
    def generate_summary_plot(self, model_type='xgboost', max_display=10):
        """Generate SHAP summary plot"""
        try:
            if model_type not in self.explainers:
                if model_type == 'xgboost':
                    self.explain_xgboost()
                elif model_type == 'lightgbm':
                    self.explain_lightgbm()
            
            if model_type not in self.explainers:
                return None
            
            shap_values = self.explainers[model_type]['shap_values']
            X = self.explainers[model_type]['X']
            feature_names = self.explainers[model_type]['feature_names']
            
            # Create plot
            shap = _load_shap()
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                            max_display=max_display, show=False)
            plt.title(f'{self.category} - Feature Importance (SHAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'shap_summary_{self.category}_{model_type}_{timestamp}.png'
            filepath = SHAP_IMAGE_DIR / filename
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_waterfall_plot(self, model_type='xgboost', sample_idx=-1):
        """Generate SHAP waterfall plot for a specific prediction"""
        try:
            if model_type not in self.explainers:
                if model_type == 'xgboost':
                    self.explain_xgboost()
                elif model_type == 'lightgbm':
                    self.explain_lightgbm()
            
            if model_type not in self.explainers:
                return None
            
            explainer = self.explainers[model_type]['explainer']
            shap_values = self.explainers[model_type]['shap_values']
            X = self.explainers[model_type]['X']
            feature_names = self.explainers[model_type]['feature_names']
            
            # Create explanation object for the sample
            shap = _load_shap()
            if hasattr(shap, 'Explanation'):
                explanation = shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value,
                    data=X[sample_idx],
                    feature_names=feature_names
                )
                
                # Create plot
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(explanation, show=False)
                plt.title(f'{self.category} - Individual Prediction Explanation', 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save plot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'shap_waterfall_{self.category}_{model_type}_{timestamp}.png'
                filepath = SHAP_IMAGE_DIR / filename
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                return filename
            else:
                return None
            
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_force_plot(self, model_type='xgboost', sample_idx=-1):
        """Generate SHAP force plot for a specific prediction"""
        try:
            if model_type not in self.explainers:
                if model_type == 'xgboost':
                    self.explain_xgboost()
                elif model_type == 'lightgbm':
                    self.explain_lightgbm()
            
            if model_type not in self.explainers:
                return None
            
            explainer = self.explainers[model_type]['explainer']
            shap_values = self.explainers[model_type]['shap_values']
            X = self.explainers[model_type]['X']
            feature_names = self.explainers[model_type]['feature_names']
            
            # Generate force plot
            shap = _load_shap()
            shap.force_plot(
                explainer.expected_value,
                shap_values[sample_idx],
                X[sample_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'shap_force_{self.category}_{model_type}_{timestamp}.png'
            filepath = SHAP_IMAGE_DIR / filename
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Error generating force plot: {e}")
            return None
    
    def get_explanation_summary(self, model_type='xgboost'):
        """Get comprehensive explanation summary"""
        try:
            # Get feature importance
            importance = self.get_feature_importance(model_type)
            if importance is None:
                return None
            
            # Generate plots
            summary_plot = self.generate_summary_plot(model_type)
            waterfall_plot = self.generate_waterfall_plot(model_type)
            
            # Get top features and their contributions
            top_features = list(importance.items())[:5]
            
            # Create interpretation
            interpretation = self._generate_interpretation(top_features)
            
            return {
                'feature_importance': importance,
                'top_features': dict(top_features),
                'summary_plot': summary_plot,
                'waterfall_plot': waterfall_plot,
                'interpretation': interpretation,
                'model_type': model_type,
                'category': self.category
            }
            
        except Exception as e:
            print(f"Error getting explanation summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_interpretation(self, top_features):
        """Generate human-readable interpretation of feature importance"""
        interpretations = []
        
        lag_descriptions = {
            'sales_lag_1': 'Most recent week\'s sales are the strongest predictor of next week\'s demand',
            'sales_lag_2': 'Sales from 2 weeks ago capture short-term momentum and buying cycles',
            'sales_lag_3': 'Sales from 3 weeks ago help identify monthly purchasing patterns',
            'sales_lag_4': 'Sales from 4 weeks ago provide early signals of seasonal trends',
            'sales_lag_5': 'Sales from 5 weeks ago establish baseline historical context'
        }
        
        for feature_name, importance in top_features:
            if 'sales_lag' in feature_name:
                lag_num = feature_name.split('_')[-1]
                interpretations.append({
                    'feature': feature_name,
                    'importance': importance,
                    'description': lag_descriptions.get(feature_name, f'Sales from {lag_num} period(s) ago influences predictions'),
                    'type': 'temporal'
                })
            else:
                interpretations.append({
                    'feature': feature_name,
                    'importance': importance,
                    'description': f'{feature_name} impacts the forecast',
                    'type': 'other'
                })
        
        return interpretations


def get_model_explainability(category, model_type='xgboost', base_path=''):
    """Wrapper function to get model explainability"""
    try:
        explainer = SHAPExplainer(category, base_path)
        results = explainer.get_explanation_summary(model_type)
        if results is not None:
            return results

        return get_fallback_explainability(category, model_type, base_path)
    except Exception as e:
        print(f"Error in get_model_explainability: {e}")
        import traceback
        traceback.print_exc()
        return get_fallback_explainability(category, model_type, base_path)


def get_fallback_explainability(category, model_type='xgboost', base_path=''):
    """Return model explainability without SHAP when optional binary deps fail."""
    try:
        explainer = SHAPExplainer(category, base_path)
        csv_path = _category_csv_path(category, base_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        X, y = explainer.create_features(df, n_lags=5)

        importance = _fallback_feature_importance(category, model_type, X, y, base_path)
        sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        top_features = list(sorted_importance.items())[:5]
        summary_plot = _save_fallback_importance_plot(category, model_type, sorted_importance)
        waterfall_plot = _save_fallback_contribution_plot(category, model_type, top_features)

        return {
            'feature_importance': sorted_importance,
            'top_features': dict(top_features),
            'summary_plot': summary_plot,
            'waterfall_plot': waterfall_plot,
            'interpretation': explainer._generate_interpretation(top_features),
            'model_type': f'{model_type} fallback',
            'category': category,
            'method': 'fallback_lag_importance',
            'note': 'SHAP was unavailable, so lag importance was estimated from model importances or historical correlations.'
        }
    except Exception as exc:
        print(f"Fallback explainability failed: {exc}")
        return None


def _fallback_feature_importance(category, model_type, X, y, base_path=''):
    feature_names = [f'sales_lag_{i+1}' for i in range(X.shape[1])]

    model_importance = _load_model_feature_importance(category, model_type, base_path)
    if model_importance is not None and len(model_importance) == len(feature_names):
        values = np.abs(np.asarray(model_importance, dtype=float))
    else:
        values = []
        for idx in range(X.shape[1]):
            corr = np.corrcoef(X[:, idx], y)[0, 1]
            values.append(0.0 if np.isnan(corr) else abs(float(corr)))
        values = np.asarray(values, dtype=float)

    if float(values.sum()) == 0.0:
        values = np.ones(len(feature_names), dtype=float)

    normalized = values / values.sum()
    return {name: float(value) for name, value in zip(feature_names, normalized)}


def _load_model_feature_importance(category, model_type, base_path=''):
    try:
        if model_type == 'xgboost':
            import pickle
            model_path = _model_path('models_xgb', f'{category}_xgb.pkl', base_path)
            with open(model_path, 'rb') as handle:
                model = pickle.load(handle)
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            if hasattr(model, 'get_score'):
                scores = model.get_score(importance_type='gain')
                return [scores.get(f'f{idx}', 0.0) for idx in range(5)]

    except Exception as exc:
        print(f"Model feature importance unavailable for {model_type}: {exc}")

    return None


def _save_fallback_importance_plot(category, model_type, importance):
    try:
        os.makedirs(SHAP_IMAGE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fallback_importance_{category}_{model_type}_{timestamp}.png'
        filepath = SHAP_IMAGE_DIR / filename

        labels = list(importance.keys())
        values = list(importance.values())

        plt.figure(figsize=(10, 6))
        plt.barh(labels[::-1], values[::-1], color='#0D8ABC')
        plt.xlabel('Relative Importance')
        plt.title(f'{category} - Lag Feature Importance')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as exc:
        print(f"Fallback importance plot failed: {exc}")
        return None


def _save_fallback_contribution_plot(category, model_type, top_features):
    try:
        os.makedirs(SHAP_IMAGE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fallback_contribution_{category}_{model_type}_{timestamp}.png'
        filepath = SHAP_IMAGE_DIR / filename

        labels = [feature for feature, _ in top_features]
        values = [value for _, value in top_features]
        cumulative = np.cumsum(values)

        plt.figure(figsize=(10, 6))
        plt.plot(labels, cumulative, marker='o', color='#28A745', linewidth=2)
        plt.fill_between(labels, cumulative, color='#28A745', alpha=0.15)
        plt.ylabel('Cumulative Relative Contribution')
        plt.title(f'{category} - Top Lag Contributions')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    except Exception as exc:
        print(f"Fallback contribution plot failed: {exc}")
        return None
