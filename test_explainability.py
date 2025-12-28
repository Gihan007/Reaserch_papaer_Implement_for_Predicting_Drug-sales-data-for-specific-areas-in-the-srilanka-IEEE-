"""
Test SHAP Explainability API
"""

import requests
import json

def test_explainability():
    """Test the SHAP explainability endpoint"""
    
    url = 'http://localhost:5000/api/explainability'
    
    # Test with C1 and XGBoost
    payload = {
        'category': 'C1',
        'model_type': 'xgboost'
    }
    
    print("=" * 60)
    print("Testing SHAP Explainability API")
    print("=" * 60)
    print(f"\nRequest: POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nSending request... (this may take 30-60 seconds)")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                results = data['results']
                
                print("\n✅ SUCCESS!")
                print("\n" + "=" * 60)
                print("FEATURE IMPORTANCE RANKINGS:")
                print("=" * 60)
                
                importance = results.get('feature_importance', {})
                for i, (feature, value) in enumerate(list(importance.items())[:10], 1):
                    print(f"{i:2d}. {feature:20s}: {value:.6f}")
                
                print("\n" + "=" * 60)
                print("TOP 5 FEATURE INTERPRETATIONS:")
                print("=" * 60)
                
                interpretations = results.get('interpretation', [])
                for interp in interpretations:
                    print(f"\n🔹 {interp['feature']}")
                    print(f"   Type: {interp['type']}")
                    print(f"   Importance: {interp['importance']:.6f}")
                    print(f"   📝 {interp['description']}")
                
                print("\n" + "=" * 60)
                print("VISUALIZATION FILES:")
                print("=" * 60)
                
                if results.get('summary_plot'):
                    print(f"✅ Summary Plot: static/images/shap/{results['summary_plot']}")
                else:
                    print("❌ Summary Plot: Not generated")
                
                if results.get('waterfall_plot'):
                    print(f"✅ Waterfall Plot: static/images/shap/{results['waterfall_plot']}")
                else:
                    print("❌ Waterfall Plot: Not generated")
                
                print("\n" + "=" * 60)
                print(f"Category: {results['category']}")
                print(f"Model Type: {results['model_type']}")
                print("=" * 60)
                
            else:
                print(f"\n❌ Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out after 120 seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    test_explainability()
