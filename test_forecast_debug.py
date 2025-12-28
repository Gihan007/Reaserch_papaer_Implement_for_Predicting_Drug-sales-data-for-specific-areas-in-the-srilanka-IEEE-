"""Debug script to test if different models return different values"""
from forecast_utils import get_lstm_forecast, get_gru_forecast, get_transformer_forecast, get_xgboost_forecast
import traceback

category = 'C1'
days_ahead = 7

print(f"\nTesting forecasts for {category}, {days_ahead} days ahead:\n")
print("-" * 60)

try:
    lstm_result = get_lstm_forecast(category, days_ahead)
    print(f"LSTM: {lstm_result}")
except Exception as e:
    print(f"LSTM ERROR: {e}")
    traceback.print_exc()

print("-" * 60)

try:
    gru_result = get_gru_forecast(category, days_ahead)
    print(f"GRU: {gru_result}")
except Exception as e:
    print(f"GRU ERROR: {e}")
    traceback.print_exc()

print("-" * 60)

try:
    transformer_result = get_transformer_forecast(category, days_ahead)
    print(f"Transformer: {transformer_result}")
except Exception as e:
    print(f"Transformer ERROR: {e}")
    traceback.print_exc()

print("-" * 60)

try:
    xgb_result = get_xgboost_forecast(category, days_ahead)
    print(f"XGBoost: {xgb_result}")
except Exception as e:
    print(f"XGBoost ERROR: {e}")
    traceback.print_exc()

print("-" * 60)
