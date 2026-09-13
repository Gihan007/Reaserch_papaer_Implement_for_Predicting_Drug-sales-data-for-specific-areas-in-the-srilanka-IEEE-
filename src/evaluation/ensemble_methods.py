import pandas as pd
import numpy as np
import json

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
        """Fit Ridge on chronological out-of-fold XGBoost/LSTM/GRU forecasts."""
        from src.models.stacking_model import StackingConfig, StackingForecaster, load_weekly_series
        _, values = load_weekly_series(category)
        self.stacking_model = StackingForecaster(StackingConfig(n_folds=n_folds)).fit(values)
        self.meta_model = self.stacking_model.meta_model_
        return self.meta_model

    def stacking_ensemble(self, predictions, meta_model):
        """Combine the three base predictions separately at each future step."""
        from src.models.stacking_model import BASE_NAMES
        if meta_model is None:
            raise ValueError("Stacking requires a fitted meta-model")
        columns = [np.asarray(predictions[name], dtype=float) for name in BASE_NAMES]
        if any(column.ndim != 1 for column in columns) or len({len(c) for c in columns}) != 1:
            raise ValueError("Stacking predictions must be aligned one-dimensional arrays")
        matrix = np.column_stack(columns)
        if len(matrix) == 0 or not np.isfinite(matrix).all():
            raise ValueError("Stacking predictions must be finite and nonempty")
        return np.maximum(meta_model.predict(matrix), 0.0)

    def create_ensemble_predictions(self, category, method='weighted_average', n_steps=10):
        """Create ensemble predictions using specified method"""

        print(f"🔄 Creating {method} ensemble for {category}...")

        # Stacking trains fresh base models on chronological prefixes. It must
        # never consume legacy predictions from full-data saved artifacts.
        if method == 'stacking':
            self.train_meta_model(category)
            return self.stacking_model.predict(n_steps)

        # Get individual model predictions
        predictions = self.get_model_predictions(category, n_steps)

        if method == 'weighted_average':
            ensemble_pred = self.weighted_average_ensemble(predictions)
        elif method == 'performance_weighted':
            ensemble_pred = self.performance_weighted_ensemble(predictions, category)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return ensemble_pred

def evaluate_ensembles():
    """Compare stacking, its three bases, equal averaging and naive on one holdout.

    Writes stacking_results.json, preserving the archived seven-model experiment.
    """
    from src.evaluation.stacking_evaluation import evaluate_stacking
    return evaluate_stacking()

if __name__ == "__main__":
    evaluate_ensembles()
