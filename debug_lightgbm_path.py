"""Debug LightGBM path issue"""
import os

base_path = ''
model_dir = os.path.join(base_path, 'models_lightgbm', '') if base_path else './models_lightgbm/'
print(f"model_dir: '{model_dir}'")
print(f"Full path would be: '{model_dir}C1_lightgbm.txt'")

# Check if file exists
import os.path
expected_path = './models_lightgbm/C1_lightgbm.txt'
print(f"\nExpected path: {expected_path}")
print(f"File exists: {os.path.exists(expected_path)}")

# List what's actually in the directory
if os.path.exists('./models_lightgbm'):
    print(f"\nFiles in ./models_lightgbm:")
    for f in os.listdir('./models_lightgbm'):
        print(f"  - {f}")
