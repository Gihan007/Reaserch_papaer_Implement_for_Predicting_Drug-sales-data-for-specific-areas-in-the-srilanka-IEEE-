import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import json
import os
import time

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna for key models"""

    def __init__(self):
        self.best_params = {}
        self.study_results = {}

    def optimize_lightgbm(self, category, n_trials=50):
        """Optimize LightGBM hyperparameters"""

        def objective(trial):
            # Load data
            df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
            data = df[category].values

            # Hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(data):
                train_data = data[train_idx]
                val_data = data[val_idx]

                # Create features (simple lag features for optimization)
                n_lags = 5
                X_train, y_train = self.create_lag_features(train_data, n_lags)
                X_val, y_val = self.create_lag_features(val_data, n_lags)

                if len(X_train) == 0 or len(X_val) == 0:
                    continue

                # Train model
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**params, random_state=42, verbosity=-1)
                model.fit(X_train, y_train)

                # Predict
                predictions = model.predict(X_val)

                # Calculate MAE
                mae = mean_absolute_error(y_val, predictions)
                scores.append(mae)

            return np.mean(scores) if scores else float('inf')

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params['lightgbm'] = study.best_params
        self.study_results['lightgbm'] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        print(f"✅ LightGBM optimization complete. Best MAE: {study.best_value:.4f}")
        return study.best_params

    def optimize_xgboost(self, category, n_trials=50):
        """Optimize XGBoost hyperparameters"""

        def objective(trial):
            df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
            data = df[category].values

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(data):
                train_data = data[train_idx]
                val_data = data[val_idx]

                n_lags = 5
                X_train, y_train = self.create_lag_features(train_data, n_lags)
                X_val, y_val = self.create_lag_features(val_data, n_lags)

                if len(X_train) == 0 or len(X_val) == 0:
                    continue

                from xgboost import XGBRegressor
                model = XGBRegressor(**params, random_state=42, verbosity=0)
                model.fit(X_train, y_train)

                predictions = model.predict(X_val)
                mae = mean_absolute_error(y_val, predictions)
                scores.append(mae)

            return np.mean(scores) if scores else float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params['xgboost'] = study.best_params
        self.study_results['xgboost'] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        print(f"✅ XGBoost optimization complete. Best MAE: {study.best_value:.4f}")
        return study.best_params

    def optimize_lstm(self, category, n_trials=30):
        """Optimize LSTM hyperparameters"""

        def objective(trial):
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
            data = df[category].values

            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

            params = {
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128)
            }

            seq_length = 10
            X, y = self.create_sequences(data_scaled, seq_length)

            if len(X) < 10:
                return float('inf')

            # Reshape X for LSTM input: (batch_size, seq_length, input_size)
            X = X.reshape(X.shape[0], seq_length, 1)

            # Time series split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            # Define model
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.fc(out[:, -1, :])
                    return out

            model = LSTMModel(hidden_size=params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout'])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

            # Train for a few epochs
            model.train()
            for epoch in range(10):  # Quick training for optimization
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            val_predictions = []
            val_actuals = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    val_predictions.extend(outputs.squeeze().numpy())
                    val_actuals.extend(y_batch.numpy())

            # Inverse transform
            val_predictions = scaler.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
            val_actuals = scaler.inverse_transform(np.array(val_actuals).reshape(-1, 1)).flatten()

            mae = mean_absolute_error(val_actuals, val_predictions)
            return mae

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params['lstm'] = study.best_params
        self.study_results['lstm'] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        print(f"✅ LSTM optimization complete. Best MAE: {study.best_value:.4f}")
        return study.best_params

    def create_lag_features(self, data, n_lags):
        """Create lag features for ML models"""
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i-n_lags:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM/GRU models"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    def run_optimization(self, categories=None, n_trials=50):
        """Run hyperparameter optimization for all optimizable models"""

        if categories is None:
            categories = ['C1']  # Use first category as representative

        print("🔬 Starting Hyperparameter Optimization")
        print("=" * 50)

        # Use first category for optimization (assuming similar patterns)
        category = categories[0]
        print(f"Using category {category} for optimization...")

        # Optimize LightGBM
        print("\n🚀 Optimizing LightGBM...")
        self.optimize_lightgbm(category, n_trials)

        # Optimize XGBoost
        print("\n🚀 Optimizing XGBoost...")
        self.optimize_xgboost(category, n_trials)

        # Optimize LSTM
        print("\n🚀 Optimizing LSTM...")
        self.optimize_lstm(category, max(10, n_trials//2))  # Fewer trials for LSTM due to computational cost

        # Save results
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/optimized_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        with open('evaluation_results/optimization_results.json', 'w') as f:
            json.dump(self.study_results, f, indent=2)

        print("\n🎉 Optimization complete! Results saved to 'evaluation_results/'")
        return self.best_params

    def get_optimized_params(self, model_name):
        """Get optimized parameters for a specific model"""
        try:
            with open('evaluation_results/optimized_params.json', 'r') as f:
                params = json.load(f)
            return params.get(model_name, {})
        except:
            return {}

def compare_optimized_vs_default():
    """Compare performance of optimized vs default parameters"""

    optimizer = HyperparameterOptimizer()
    category = 'C1'

    # Load optimized parameters
    optimized_params = optimizer.get_optimized_params('lightgbm')

    if not optimized_params:
        print("No optimized parameters found. Run optimization first.")
        return

    # Load data
    df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
    data = df[category].values

    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    }

    # Evaluate both
    results = {}

    for name, params in [('default', default_params), ('optimized', optimized_params)]:
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(data):
            train_data = data[train_idx]
            val_data = data[val_idx]

            n_lags = 5
            X_train, y_train = optimizer.create_lag_features(train_data, n_lags)
            X_val, y_val = optimizer.create_lag_features(val_data, n_lags)

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params, random_state=42, verbosity=-1)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            scores.append(mae)

        results[name] = np.mean(scores) if scores else float('inf')

    print("📊 Optimization Results Comparison:")
    print(f"  Default MAE: {results['default']:.4f}")
    print(f"  Optimized MAE: {results['optimized']:.4f}")
    print(f"  Improvement: {((results['default'] - results['optimized']) / results['default'] * 100):.2f}%")

    return results

if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    optimizer.run_optimization()

    # Compare results
    compare_optimized_vs_default()