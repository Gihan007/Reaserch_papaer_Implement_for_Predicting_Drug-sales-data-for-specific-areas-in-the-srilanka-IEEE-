import pandas as pd
import numpy as np
import joblib
import os
from src.utils.preprocessing import create_lagged_features

def forecast_xgboost(category, n_lags, n_steps, base_path='../../', model_dir='../../models_xgb/'):
    df = pd.read_csv(f'{base_path}{category}.csv', parse_dates=['datum'], index_col='datum')
    model = joblib.load(f'{model_dir}{category}_xgb.pkl')
    last_vals = df[category].values[-n_lags:]
    preds = []
    for _ in range(n_steps):
        X_pred = last_vals[-n_lags:][::-1]  # most recent first
        X_pred = X_pred[::-1].reshape(1, -1)
        y_pred = model.predict(X_pred)[0]
        preds.append(y_pred)
        last_vals = np.append(last_vals, y_pred)
    return preds
