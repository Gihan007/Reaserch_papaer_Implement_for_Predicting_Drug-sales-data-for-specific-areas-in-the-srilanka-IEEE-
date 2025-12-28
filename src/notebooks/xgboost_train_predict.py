import pandas as pd
from src.utils.preprocessing import create_lagged_features
from src.models.xgboost_model import fit_xgboost, predict_xgboost
import joblib
import os

categories = [f"C{i}" for i in range(1, 9)]
base_path = '../../'
model_dir = '../../models_xgb/'
os.makedirs(model_dir, exist_ok=True)

for cat in categories:
    df = pd.read_csv(f'{base_path}{cat}.csv', parse_dates=['datum'], index_col='datum')
    df_lagged = create_lagged_features(df[cat], n_lags=5)
    X = df_lagged.drop(cat, axis=1).values
    y = df_lagged[cat].values
    model = fit_xgboost(X, y)
    joblib.dump(model, f'{model_dir}{cat}_xgb.pkl')
    y_pred = predict_xgboost(model, X)
    print(f'{cat} - Train RMSE:', ((y_pred - y) ** 2).mean() ** 0.5)
