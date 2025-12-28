# Step 5: Integrate with Flask API

## 🎯 Purpose
Add pharmaceutical explanation endpoint to your existing Flask backend.

## 🔌 Integration Overview

Add **ONE new endpoint** to `app.py`:

```python
POST /api/explain
```

**Input:** Prediction results from any model (LSTM, GRU, XGBoost, etc.)  
**Output:** Detailed pharmaceutical explanation

## ⚡ Quick Integration

### Step 1: Copy LLM Module

```bash
# Copy inference code to src/
cp llm_explainer.py ../../src/llm_api.py
```

### Step 2: Add to app.py

Add this endpoint to your `src/app.py` or `app.py`:

```python
from llm_api import PharmaceuticalLLMAPI

# Initialize (loads model once at startup)
llm_explainer = PharmaceuticalLLMAPI(
    model_path='llm_finetuning/output/fine_tuned_model'
)

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Generate pharmaceutical explanation for prediction"""
    try:
        data = request.json
        
        # Required fields
        category = data.get('category')
        prediction = data.get('prediction')
        
        if not category or prediction is None:
            return jsonify({'error': 'Missing category or prediction'}), 400
        
        # Optional context
        week = data.get('week', 1)
        year = data.get('year', 2024)
        location = data.get('location', 'Colombo')
        
        # Generate explanation
        explanation = llm_explainer.explain({
            'category': category,
            'prediction': prediction,
            'week': week,
            'year': year,
            'location': location
        })
        
        return jsonify({
            'success': True,
            'explanation': explanation,
            'metadata': {
                'category': category,
                'prediction': prediction,
                'week': week,
                'year': year,
                'location': location
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Step 3: Test Endpoint

```powershell
# Test with PowerShell
$body = @{
    category = "C1"
    prediction = 50.38
    week = 12
    year = 2018
    location = "Colombo"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/explain" -Method POST -Body $body -ContentType "application/json"
```

Or with curl (if installed):
```bash
curl -X POST http://localhost:5000/api/explain \
  -H "Content-Type: application/json" \
  -d '{"category":"C1","prediction":50.38,"week":12,"year":2018}'
```

## 📊 Frontend Integration

### Update Prediction Display

Modify your frontend to fetch explanations:

```javascript
// After getting prediction from /api/predict
async function getPredictionWithExplanation(category, modelName, weeks) {
    // Get prediction
    const predictionResponse = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({category, model: modelName, weeks})
    });
    const predictionData = await predictionResponse.json();
    
    // Get explanation
    const explainResponse = await fetch('/api/explain', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            category: category,
            prediction: predictionData.prediction,
            week: getCurrentWeek(),
            year: new Date().getFullYear()
        })
    });
    const explainData = await explainResponse.json();
    
    return {
        prediction: predictionData,
        explanation: explainData.explanation
    };
}

// Display explanation
function displayResults(data) {
    // Show prediction
    document.getElementById('prediction-value').textContent = 
        data.prediction.prediction;
    
    // Show explanation
    document.getElementById('explanation-text').innerHTML = 
        formatExplanation(data.explanation);
}

function formatExplanation(text) {
    // Format with proper line breaks and styling
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/RECOMMENDATIONS:/g, '<h4>RECOMMENDATIONS:</h4>')
        .replace(/ANALYSIS:/g, '<h4>ANALYSIS:</h4>');
}
```

### Add Explanation UI Component

Add to your prediction results page:

```html
<!-- Existing prediction display -->
<div class="prediction-result">
    <h3>Prediction: <span id="prediction-value">--</span> units</h3>
</div>

<!-- NEW: Explanation section -->
<div class="explanation-section">
    <h3>📊 Analysis & Insights</h3>
    <div id="explanation-text" class="explanation-content">
        <p class="loading">Generating pharmaceutical analysis...</p>
    </div>
</div>

<style>
.explanation-section {
    margin-top: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #0078d4;
}

.explanation-content {
    font-size: 14px;
    line-height: 1.8;
    color: #333;
}

.explanation-content h4 {
    color: #0078d4;
    margin-top: 20px;
    margin-bottom: 10px;
}

.loading {
    text-align: center;
    color: #999;
    font-style: italic;
}
</style>
```

## 🚀 Production Optimizations

### 1. Caching (Redis)

```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    data = request.json
    
    # Create cache key
    cache_key = hashlib.md5(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
    
    # Check cache
    cached = redis_client.get(f"explanation:{cache_key}")
    if cached:
        return jsonify(json.loads(cached))
    
    # Generate explanation
    explanation = llm_explainer.explain(data)
    result = {'success': True, 'explanation': explanation}
    
    # Cache for 24 hours
    redis_client.setex(
        f"explanation:{cache_key}",
        86400,  # 24 hours
        json.dumps(result)
    )
    
    return jsonify(result)
```

### 2. Async Processing (Celery)

For slow inference, use background tasks:

```python
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379/0')

@celery.task
def generate_explanation_async(data):
    """Background task for explanation generation"""
    return llm_explainer.explain(data)

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    data = request.json
    
    # Start background task
    task = generate_explanation_async.delay(data)
    
    return jsonify({
        'task_id': task.id,
        'status': 'processing'
    })

@app.route('/api/explain/status/<task_id>')
def explanation_status(task_id):
    """Check task status"""
    task = generate_explanation_async.AsyncResult(task_id)
    
    if task.ready():
        return jsonify({
            'status': 'completed',
            'explanation': task.result
        })
    else:
        return jsonify({'status': 'processing'})
```

### 3. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/explain', methods=['POST'])
@limiter.limit("10 per minute")  # Expensive operation
def explain_prediction():
    # ... explanation logic
```

## 🔍 Example Complete Workflow

### User Flow
1. User selects drug category (e.g., C1)
2. User selects model (e.g., LSTM)
3. User clicks "Predict"
4. System shows prediction: **50.38 units**
5. System automatically generates explanation (2-5 sec)
6. User sees detailed analysis with recommendations

### Backend Flow
```
User Request
    ↓
/api/predict (existing)
    ↓
Prediction: 50.38 units
    ↓
/api/explain (NEW)
    ↓
LLM Fine-tuned Model
    ↓
Detailed Explanation
    ↓
Frontend Display
```

## 📦 Deployment Checklist

### Development Environment
```bash
# Install dependencies
pip install -r llm_finetuning/requirements.txt

# Load model (first time)
python -c "from llm_api import PharmaceuticalLLMAPI; PharmaceuticalLLMAPI()"

# Run Flask
python app.py
```

### Production Environment

**Option A: Same Server (GPU)**
```bash
# Ensure GPU drivers installed
nvidia-smi

# Run with gunicorn
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 app:app
```

**Option B: Separate Inference Server**
```python
# inference_server.py (runs on GPU machine)
from flask import Flask, request, jsonify
from llm_api import PharmaceuticalLLMAPI

app = Flask(__name__)
explainer = PharmaceuticalLLMAPI()

@app.route('/explain', methods=['POST'])
def explain():
    return jsonify(explainer.explain(request.json))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

# Main app.py calls this server
explanation_server = "http://gpu-server:8000"
response = requests.post(f"{explanation_server}/explain", json=data)
```

**Option C: Use OpenAI API (No GPU Needed)**
```python
# Replace fine-tuned model with GPT-4
import openai

def explain_with_gpt4(data):
    # Load few-shot examples
    with open('llm_finetuning/output/training_data/sales_explanations.json') as f:
        examples = json.load(f)[:5]
    
    prompt = f"Examples:\n{examples}\n\nExplain: {data}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

## 🐛 Troubleshooting

### "Model not found"
**Solution:** Check path in app.py
```python
# Adjust path relative to app.py location
llm_explainer = PharmaceuticalLLMAPI(
    model_path='llm_finetuning/output/fine_tuned_model'  # From project root
)
```

### "CUDA out of memory"
**Solutions:**
1. Use CPU inference (slower but works)
2. Reduce batch size
3. Use separate inference server
4. Switch to API-based solution (GPT-4)

### Slow response times
**Solutions:**
1. Implement caching (Redis)
2. Use async processing (Celery)
3. Pre-generate common explanations
4. Use smaller model (Phi-3 Mini)

### Explanation quality poor
**Check:**
1. Fine-tuning completed successfully (Step 3)
2. Training loss < 0.3
3. Model files loaded correctly
4. Input data format matches training format

## ✅ Verification Steps

Test each component:

### 1. Model Loading
```python
python -c "from llm_api import PharmaceuticalLLMAPI; print('✅ Model loaded')"
```

### 2. Single Prediction
```bash
curl -X POST http://localhost:5000/api/explain -H "Content-Type: application/json" -d '{"category":"C1","prediction":50}'
```

### 3. Full Workflow
```bash
# Predict
curl -X POST http://localhost:5000/api/predict -d '{"category":"C1","model":"lstm"}'

# Explain
curl -X POST http://localhost:5000/api/explain -d '{"category":"C1","prediction":50.38}'
```

### 4. Load Testing
```bash
# Install Apache Bench
# Test endpoint performance
ab -n 100 -c 10 -T "application/json" -p test_data.json http://localhost:5000/api/explain
```

## 📊 Monitoring

Add logging:

```python
import logging

logging.basicConfig(
    filename='llm_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    start_time = time.time()
    
    try:
        explanation = llm_explainer.explain(data)
        
        duration = time.time() - start_time
        logging.info(f"Explanation generated in {duration:.2f}s for {data['category']}")
        
        return jsonify({'success': True, 'explanation': explanation})
    
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        raise
```

## ✅ Success Criteria

Integration complete when:
- ✅ `/api/explain` endpoint works
- ✅ Frontend displays explanations
- ✅ Response time < 10 seconds
- ✅ Explanations are pharmaceutical-specific
- ✅ Error handling in place
- ✅ Monitoring/logging configured

---

**Integration complete? 🎉 Your pharmaceutical forecasting system now has AI-powered explanations!**

## 📖 Next Steps

1. **Test with real users** - Get feedback on explanation quality
2. **Optimize performance** - Add caching, async processing
3. **Enhance explanations** - Add confidence scores, visualizations
4. **Write IEEE paper** - Document this novel approach!
5. **Monitor usage** - Track which explanations are most helpful

See `../../IEEE_PAPER_TEMPLATE.md` for paper writing guidance!
