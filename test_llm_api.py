"""
Quick test for LLM Explanation API
"""
import requests
import json

# Test data
test_data = {
    'category': 'C1',
    'prediction': 50.38,
    'date': '2025-03-15',  # Week 12
    'model_type': 'Ensemble'
}

print("=" * 80)
print("Testing LLM Explanation API")
print("=" * 80)
print(f"\nTest Input:")
print(json.dumps(test_data, indent=2))

try:
    response = requests.post(
        'http://localhost:5000/api/explain',
        json=test_data,
        timeout=10
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        if result['success']:
            print("\n✅ SUCCESS!\n")
            print("Explanation:")
            print("=" * 80)
            print(result['explanation'])
            print("=" * 80)
            
            print(f"\nMetadata:")
            print(json.dumps(result['metadata'], indent=2))
        else:
            print(f"\n❌ Error: {result.get('error')}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Cannot connect to Flask server")
    print("Make sure Flask is running: python app.py")
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
