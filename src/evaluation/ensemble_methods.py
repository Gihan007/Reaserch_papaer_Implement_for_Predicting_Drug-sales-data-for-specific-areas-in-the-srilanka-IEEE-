import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import json
import os
import time

class EnsembleMethods:
    """Advanced ensemble methods for combining multiple model predictions"""

    def __init__(self):
        self.models = ['sarimax', 'xgboost', 'transformer', 'gru', 'lstm', 'lightgbm', 'prophet']
        self.ensemble_weights = None
        self.meta_model = None

    def get_model_predictions(self, category, n_steps=10):
        """Get predictions from all individual models"""

        predictions = {}

        try:
            # SARIMAX
            from models.sarimax_model import predict_sarimax, fit_sarimax
            csv_path = '../temp_window.csv' if category == 'temp_window' else f'../{category}.csv'
            df = pd.read_csv(csv_path, parse_dates=['datum'], index_col='datum')
            order = (1, 0, 0)
            seasonal_order = (1, 0, 0, 7)
            fitted_model = fit_sarimax(df[category], order, seasonal_order)
            predictions['sarimax'] = predict_sarimax(fitted_model, start=len(df)-n_steps, end=len(df)-1, dynamic=True)

        except Exception as e:
            print(f"SARIMAX prediction failed: {e}")
            predictions['sarimax'] = np.zeros(n_steps)

        try:
            # XGBoost
            from utils.xgb_forecast import forecast_xgboost
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_xgb/' if category == 'temp_window' else '../models_xgb/'
            predictions['xgboost'] = forecast_xgboost(category, n_lags=5, n_steps=n_steps, base_path=base_path, model_dir=model_dir)
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            predictions['xgboost'] = np.zeros(n_steps)

        try:
            # Transformer
            from models.transformer_model import forecast_transformer
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_transformer/' if category == 'temp_window' else '../models_transformer/'
            predictions['transformer'] = [forecast_transformer(category, seq_length=10, n_steps=i+1, base_path=base_path, model_dir=model_dir) for i in range(n_steps)]
        except Exception as e:
            print(f"Transformer prediction failed: {e}")
            predictions['transformer'] = np.zeros(n_steps)

        try:
            # GRU
            from models.gru_model import forecast_gru
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_gru/' if category == 'temp_window' else '../models_gru/'
            predictions['gru'] = [forecast_gru(category, seq_length=10, n_steps=i+1, base_path=base_path, model_dir=model_dir) for i in range(n_steps)]
        except Exception as e:
            print(f"GRU prediction failed: {e}")
            predictions['gru'] = np.zeros(n_steps)

        try:
            # LSTM
            from models.lstm_model import forecast_lstm
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_lstm/' if category == 'temp_window' else '../models_lstm/'
            predictions['lstm'] = [forecast_lstm(category, seq_length=10, n_steps=i+1, base_path=base_path, model_dir=model_dir) for i in range(n_steps)]
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            predictions['lstm'] = np.zeros(n_steps)

        try:
            # LightGBM
            from models.lightgbm_model import forecast_lightgbm
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_lightgbm/' if category == 'temp_window' else '../models_lightgbm/'
            predictions['lightgbm'] = [forecast_lightgbm(category, n_lags=5, n_steps=i+1, base_path=base_path, model_dir=model_dir) for i in range(n_steps)]
        except Exception as e:
            print(f"LightGBM prediction failed: {e}")
            predictions['lightgbm'] = np.zeros(n_steps)

        try:
            # Prophet
            from models.prophet_model import forecast_prophet
            base_path = '' if category == 'temp_window' else '../'
            model_dir = '../models_prophet/' if category == 'temp_window' else '../models_prophet/'
            predictions['prophet'] = [forecast_prophet(category, periods=i+1, base_path=base_path, model_dir=model_dir) for i in range(n_steps)]
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            predictions['prophet'] = np.zeros(n_steps)

        return predictions

    def weighted_average_ensemble(self, predictions, weights=None):
        """Simple weighted average ensemble"""

        if weights is None:
            # Equal weights
            weights = {model: 1.0 / len(self.models) for model in self.models}

        ensemble_pred = np.zeros(len(predictions[self.models[0]]))

        for model in self.models:
            if model in predictions:
                pred = np.array(predictions[model])
                ensemble_pred += weights.get(model, 0) * pred

        return ensemble_pred

    def performance_weighted_ensemble(self, predictions, category):
        """Weight models by their inverse error (better models get higher weights)"""

        # Load performance metrics if available
        try:
            with open('evaluation_results/performance_summary.json', 'r') as f:
                performance = json.load(f)

            # Use inverse of MAE as weights
            weights = {}
            total_weight = 0

            for model in self.models:
                if model in performance and performance[model]['final_mae'] is not None:
                    # Inverse MAE (lower MAE = higher weight)
                    weight = 1.0 / (performance[model]['final_mae'] + 1e-6)  # Add small epsilon to avoid division by zero
                    weights[model] = weight
                    total_weight += weight
                else:
                    weights[model] = 1.0
                    total_weight += 1.0

            # Normalize weights
            for model in weights:
                weights[model] /= total_weight

        except:
            # Fallback to equal weights
            weights = {model: 1.0 / len(self.models) for model in self.models}

        return self.weighted_average_ensemble(predictions, weights)

    def train_meta_model(self, category, n_folds=3):
        """Train a meta-model (stacking) using individual model predictions as features"""

        df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
        data = df[category].values

        # Create training data for meta-model
        meta_features = []
        targets = []

        # Use sliding window approach
        window_size = 20
        for i in range(window_size, len(data) - 10):
            # Get predictions for this window
            window_data = data[i-window_size:i]

            # Create temporary dataframe for this window
            temp_df = pd.DataFrame({'datum': df.index[i-window_size:i], category: window_data})
            temp_df.to_csv('temp_window.csv', index=False)

            try:
                # Get predictions from all models for next 10 steps
                window_predictions = self.get_model_predictions('temp_window', n_steps=10)

                # Use current window predictions as features
                features = []
                for model in self.models:
                    if model in window_predictions:
                        # Take the first prediction as feature
                        features.append(window_predictions[model][0] if isinstance(window_predictions[model], (list, np.ndarray)) else window_predictions[model])

                meta_features.append(features)
                targets.append(data[i:i+10])  # Next 10 actual values

            except Exception as e:
                print(f"Error creating meta features: {e}")
                continue

        if len(meta_features) < 10:
            print("Insufficient data for meta-model training, using simple averaging")
            return None

        # Train meta-model
        meta_features = np.array(meta_features)
        targets = np.array(targets)

        # Flatten targets for regression
        targets_flat = targets.flatten()
        meta_features_repeated = np.repeat(meta_features, 10, axis=0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(meta_features_repeated, targets_flat, test_size=0.2, random_state=42)

        # Train Ridge regression as meta-model
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_train, y_train)

        # Evaluate meta-model
        y_pred = meta_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Meta-model MAE: {mae:.4f}")

        # Clean up
        if os.path.exists('temp_window.csv'):
            os.remove('temp_window.csv')

        return meta_model

    def stacking_ensemble(self, predictions, meta_model):
        """Use trained meta-model for stacking ensemble"""

        if meta_model is None:
            return self.weighted_average_ensemble(predictions)

        # Create feature matrix from individual predictions
        features = []
        for model in self.models:
            if model in predictions:
                pred = predictions[model][0] if isinstance(predictions[model], (list, np.ndarray)) else predictions[model]
                features.append(pred)

        features = np.array(features).reshape(1, -1)

        # Get meta-model prediction for first step
        meta_pred = meta_model.predict(features)[0]

        # For simplicity, use weighted average for remaining steps
        # In a full implementation, you'd retrain for multi-step
        ensemble_pred = np.full(len(predictions[self.models[0]]), meta_pred)

        return ensemble_pred

    def create_ensemble_predictions(self, category, method='weighted_average', n_steps=10):
        """Create ensemble predictions using specified method"""

        print(f"🔄 Creating {method} ensemble for {category}...")

        # Get individual model predictions
        predictions = self.get_model_predictions(category, n_steps)

        if method == 'weighted_average':
            ensemble_pred = self.weighted_average_ensemble(predictions)
        elif method == 'performance_weighted':
            ensemble_pred = self.performance_weighted_ensemble(predictions, category)
        elif method == 'stacking':
            meta_model = self.train_meta_model(category)
            ensemble_pred = self.stacking_ensemble(predictions, meta_model)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return ensemble_pred

def evaluate_ensembles():
    """Evaluate all ensemble methods"""

    ensemble_methods = EnsembleMethods()
    categories = [f'C{i}' for i in range(1, 9)]
    methods = ['weighted_average', 'performance_weighted', 'stacking']

    results = {}

    print("🎯 Evaluating Ensemble Methods")
    print("=" * 50)

    for category in categories:
        print(f"\n📊 Category: {category}")
        results[category] = {}

        # Load actual data
        df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
        actual_values = df[category].values[-10:]

        for method in methods:
            try:
                start_time = time.time()
                predictions = ensemble_methods.create_ensemble_predictions(category, method, n_steps=10)
                inference_time = time.time() - start_time

                # Calculate metrics
                mae = mean_absolute_error(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                mape = mean_absolute_percentage_error(actual_values, predictions) * 100

                results[category][method] = {
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4),
                    'MAPE': round(mape, 4),
                    'Time': round(inference_time, 4)
                }

                print(f"  ✅ {method}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, Time={inference_time:.4f}s")

            except Exception as e:
                print(f"  ❌ {method}: Error - {str(e)}")
                results[category][method] = {'MAE': None, 'RMSE': None, 'MAPE': None, 'Time': None}

    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n🎉 Ensemble evaluation complete! Results saved to 'evaluation_results/ensemble_results.json'")
    return results

if __name__ == "__main__":
    evaluate_ensembles()