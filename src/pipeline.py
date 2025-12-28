import os
import pandas as pd
from utils.preprocessing import create_lagged_features
from models.xgboost_model import fit_xgboost
from models.transformer_model import train_transformer_model
from models.gru_model import train_gru_model
from models.lstm_model import train_lstm_model
from models.lightgbm_model import train_lightgbm_model
from models.prophet_model import train_prophet_model
from models.tft_model import train_tft_model
from models.nbeats_model import train_nbeats_model
from models.informer_model import train_informer_model
import joblib
import torch

def train_xgboost_models(categories=None, base_path='../', model_dir='../models_xgb/', n_lags=5):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        df_lagged = create_lagged_features(df[cat], n_lags=n_lags)
        X = df_lagged.drop(cat, axis=1).values
        y = df_lagged[cat].values
        model = fit_xgboost(X, y)
        joblib.dump(model, f'{model_dir}/{cat}_xgb.pkl')

def train_transformer_models(categories=None, base_path='../', model_dir='../models_transformer/', seq_length=10):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        data = df[cat].values
        model, scaler = train_transformer_model(data, seq_length=seq_length)
        torch.save(model.state_dict(), f'{model_dir}/{cat}_transformer.pth')
        joblib.dump(scaler, f'{model_dir}/{cat}_scaler.pkl')

def train_gru_models(categories=None, base_path='../', model_dir='../models_gru/', seq_length=10):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        data = df[cat].values
        model, scaler = train_gru_model(data, seq_length=seq_length)
        torch.save(model.state_dict(), f'{model_dir}/{cat}_gru.pth')
        joblib.dump(scaler, f'{model_dir}/{cat}_scaler.pkl')

def train_lstm_models(categories=None, base_path='../', model_dir='../models_lstm/', seq_length=10):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        data = df[cat].values
        model, scaler = train_lstm_model(data, seq_length=seq_length)
        torch.save(model.state_dict(), f'{model_dir}/{cat}_lstm.pth')
        joblib.dump(scaler, f'{model_dir}/{cat}_scaler.pkl')

def train_lightgbm_models(categories=None, base_path='../', model_dir='../models_lightgbm/', n_lags=5):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        data = df[cat].values
        model, scaler = train_lightgbm_model(data, n_lags=n_lags)
        model.save_model(f'{model_dir}/{cat}_lightgbm.txt')
        joblib.dump(scaler, f'{model_dir}/{cat}_scaler.pkl')

def train_prophet_models(categories=None, base_path='../', model_dir='../models_prophet/'):
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        df = pd.read_csv(f'{base_path}/{cat}.csv', parse_dates=['datum'], index_col='datum')
        data = df[cat].values
        model = train_prophet_model(data, category_name=cat)
        joblib.dump(model, f'{model_dir}/{cat}_prophet.pkl')

def train_tft_models(categories=None, base_path='../', model_dir='../models_tft/', seq_length=30):
    """
    Train Temporal Fusion Transformer models for all categories
    """
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        print(f"Training TFT model for {cat}...")
        success = train_tft_model(cat, base_path=base_path, model_dir=model_dir, seq_length=seq_length)
        if success:
            print(f"TFT model trained successfully for {cat}")
        else:
            print(f"Failed to train TFT model for {cat}")

def train_nbeats_models(categories=None, base_path='../', model_dir='../models_nbeats/', seq_length=30):
    """
    Train N-BEATS models for all categories
    """
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        print(f"Training N-BEATS model for {cat}...")
        success = train_nbeats_model(cat, base_path=base_path, model_dir=model_dir, seq_length=seq_length)
        if success:
            print(f"N-BEATS model trained successfully for {cat}")
        else:
            print(f"Failed to train N-BEATS model for {cat}")

def train_informer_models(categories=None, base_path='../', model_dir='../models_informer/', seq_length=30):
    """
    Train Informer models for all categories
    """
    if categories is None:
        categories = [f'C{i}' for i in range(1, 9)]
    os.makedirs(model_dir, exist_ok=True)
    for cat in categories:
        print(f"Training Informer model for {cat}...")
        success = train_informer_model(cat, base_path=base_path, model_dir=model_dir, seq_length=seq_length)
        if success:
            print(f"Informer model trained successfully for {cat}")
        else:
            print(f"Failed to train Informer model for {cat}")
