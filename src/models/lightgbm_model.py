import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import MinMaxScaler

def create_lagged_features(data, n_lags=5):
    df = pd.DataFrame(data, columns=['value'])
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['value'].shift(i)
    df = df.dropna()
    return df

def train_lightgbm_model(data, n_lags=5):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    df_lagged = create_lagged_features(data_scaled, n_lags)
    X = df_lagged.drop('value', axis=1).values
    y = df_lagged['value'].values

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=100)

    return model, scaler

def forecast_lightgbm(category, n_lags=5, n_steps=1, base_path='', model_dir='./models_lightgbm/'):
    df = pd.read_csv(f'{base_path}{category}.csv', parse_dates=['datum'], index_col='datum')
    model = lgb.Booster(model_file=f'{model_dir}{category}_lightgbm.txt')
    scaler = joblib.load(f'{model_dir}{category}_scaler.pkl')

    data = df[category].values
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    predictions = []
    current_data = data_scaled[-n_lags:].copy()

    for _ in range(n_steps):
        X_pred = current_data[-n_lags:][::-1].reshape(1, -1)
        pred_scaled = model.predict(X_pred)[0]
        predictions.append(pred_scaled)
        current_data = np.append(current_data, pred_scaled)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions[-1]  # return the last prediction for the week_offset