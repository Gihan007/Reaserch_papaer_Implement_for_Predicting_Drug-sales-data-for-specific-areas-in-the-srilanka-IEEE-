import pandas as pd
import numpy as np
from prophet import Prophet
import joblib

def train_prophet_model(data, category_name='y'):
    # Prepare data for Prophet
    df = pd.DataFrame({'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='W'), 'y': data})

    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    model.fit(df)

    return model

def forecast_prophet(category, periods=1, base_path='', model_dir='./models_prophet/'):
    # Load data
    df = pd.read_csv(f'{base_path}{category}.csv', parse_dates=['datum'], index_col='datum')

    # Load trained model
    model = joblib.load(f'{model_dir}{category}_prophet.pkl')

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='W')

    # Make predictions
    forecast = model.predict(future)

    # Return the last prediction
    return forecast['yhat'].iloc[-1]