"""Test SARIMAX and Ensemble specifically"""
from forecast_utils import get_sarimax_forecast, get_ensemble_forecast
import traceback

print("=" * 70)
print("Testing SARIMAX:")
print("=" * 70)
try:
    result = get_sarimax_forecast('C1', 7)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing Ensemble:")
print("=" * 70)
try:
    result = get_ensemble_forecast('C1', 7)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
