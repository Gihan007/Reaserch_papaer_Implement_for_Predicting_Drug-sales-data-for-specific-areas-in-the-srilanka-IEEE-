"""Test API to verify different models return different values"""
import requests
import json

url = 'http://127.0.0.1:5000/api/forecast'

models = ['lstm', 'gru', 'transformer', 'xgboost', 'lightgbm', 'ensemble']
category = 'C1'
date = '2025-12-15'

print(f"\nTesting forecasts for {category} on {date}:\n")
print("=" * 70)

results = {}
for model in models:
    data = {
        'category': category,
        'date': date,
        'model_type': model
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            forecast_value = result.get('forecast_value')
            model_used = result.get('model_used')
            results[model] = forecast_value
            print(f"{model.upper():15} -> {forecast_value:10.2f}  (Model: {model_used})")
        else:
            print(f"{model.upper():15} -> ERROR: {response.status_code}")
    except Exception as e:
        print(f"{model.upper():15} -> ERROR: {e}")

print("=" * 70)
print(f"\nAll values unique? {len(set(results.values())) == len(results)}")
print(f"Number of unique values: {len(set(results.values()))} out of {len(results)}")
