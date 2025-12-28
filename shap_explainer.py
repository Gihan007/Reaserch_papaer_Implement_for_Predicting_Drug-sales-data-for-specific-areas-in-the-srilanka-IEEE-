"""
SHAP Explainability Module
Provides feature importance and model explanations using SHAP values
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
            
            # Load model
            if model_path is None:
                model_path = os.path.join(self.base_path, f'models_xgb/{self.category}_xgb.pkl')
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return None
            
            # Load pickled model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load data and create features
            csv_path = os.path.join(self.base_path, f'{self.category}.csv')
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
            import lightgbm as lgb
            
            # Load model
            if model_path is None:
                model_path = os.path.join(self.base_path, f'models_lightgbm/{self.category}_lightgbm.txt')
            
            if not os.path.exists(model_path):
                return None
            
            model = lgb.Booster(model_file=model_path)
            
            # Load data and create features
            csv_path = os.path.join(self.base_path, f'{self.category}.csv')
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            X, y = self.create_features(df, n_lags=5)
            
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
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                            max_display=max_display, show=False)
            plt.title(f'{self.category} - Feature Importance (SHAP)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'shap_summary_{self.category}_{model_type}_{timestamp}.png'
            filepath = os.path.join('static', 'images', 'shap', filename)
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
                filepath = os.path.join('static', 'images', 'shap', filename)
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
            filepath = os.path.join('static', 'images', 'shap', filename)
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
        return explainer.get_explanation_summary(model_type)
    except Exception as e:
        print(f"Error in get_model_explainability: {e}")
        import traceback
        traceback.print_exc()
        return None
