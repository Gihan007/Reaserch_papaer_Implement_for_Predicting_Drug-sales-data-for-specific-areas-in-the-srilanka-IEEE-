// ============================================
// Forecasting Page JavaScript
// ============================================

let forecastChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeForecastForm();
    initializeCompareModal();
    setDefaultDate();
});

// ============================================
// Initialize Form
// ============================================
function initializeForecastForm() {
    const form = document.getElementById('forecastForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await generateForecast();
    });
}

function setDefaultDate() {
    const dateInput = document.getElementById('date');
    if (dateInput) {
        const today = new Date();
        const futureDate = new Date(today.setDate(today.getDate() + 7));
        dateInput.value = futureDate.toISOString().split('T')[0];
    }
}

// ============================================
// Generate Forecast
// ============================================
async function generateForecast() {
    const category = document.getElementById('category').value;
    const date = document.getElementById('date').value;
    const modelType = document.getElementById('modelType').value;
    
    if (!category || !date) {
        window.PharmaPredictAI.showNotification('Please fill all required fields', 'warning');
        return;
    }
    
    // Show loading
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    window.PharmaPredictAI.showLoading(loadingState);
    resultsSection.style.display = 'none';
    
    // Show starting notification
    window.PharmaPredictAI.showNotification('Generating forecast... Please wait', 'info');
    
    try {
        console.log('Making API call to /api/forecast with:', { category, date, model_type: modelType });
        
        const response = await window.PharmaPredictAI.apiCall('/api/forecast', 'POST', {
            category,
            date,
            model_type: modelType
        });
        
        console.log('API Response:', response);
        
        if (response.success) {
            displayForecastResults(response);
            window.PharmaPredictAI.showNotification('✅ Forecast generated successfully! Scroll down to see results.', 'success');
        } else {
            window.PharmaPredictAI.showNotification(`❌ ${response.error || 'Forecast failed'}`, 'error');
            console.error('Forecast error from API:', response.error);
        }
    } catch (error) {
        console.error('Forecast error:', error);
        window.PharmaPredictAI.showNotification(`❌ Error: ${error.message}`, 'error');
    } finally {
        window.PharmaPredictAI.hideLoading(loadingState);
    }
}

// ============================================
// Display Results
// ============================================
function displayForecastResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Update prediction card
    document.getElementById('forecastValue').textContent = 
        window.PharmaPredictAI.formatNumber(data.forecast_value, 2);
    document.getElementById('forecastDate').textContent = 
        window.PharmaPredictAI.formatDate(data.closest_prediction_date);
    document.getElementById('forecastCategory').textContent = data.category;
    document.getElementById('forecastModel').textContent = data.model_used;
    document.getElementById('modelBadge').textContent = data.model_used;
    
    // Update metrics (mock data for now)
    updateMetrics(data);
    
    // Create chart
    createForecastChart(data);
    
    // Update confidence intervals
    updateConfidenceIntervals(data.forecast_value);
    
    // Generate AI explanation (NEW!)
    generateExplanation(data);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateMetrics(data) {
    // Mock metrics - in production, these would come from the API
    document.getElementById('metricAccuracy').textContent = '95.8%';
    document.getElementById('metricMAE').textContent = '2.34';
    document.getElementById('metricRMSE').textContent = '3.12';
    document.getElementById('metricMAPE').textContent = '4.56%';
}

function updateConfidenceIntervals(forecastValue) {
    const value = parseFloat(forecastValue);
    const sigma1 = value * 0.05;
    const sigma2 = value * 0.10;
    const sigma3 = value * 0.15;
    
    document.getElementById('ci68').textContent = 
        `${(value - sigma1).toFixed(2)} - ${(value + sigma1).toFixed(2)} units`;
    document.getElementById('ci95').textContent = 
        `${(value - sigma2).toFixed(2)} - ${(value + sigma2).toFixed(2)} units`;
    document.getElementById('ci99').textContent = 
        `${(value - sigma3).toFixed(2)} - ${(value + sigma3).toFixed(2)} units`;
}

// ============================================
// Chart Creation
// ============================================
function createForecastChart(data) {
    const ctx = document.getElementById('forecastChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Generate mock historical data
    const historicalData = generateMockHistoricalData(data.forecast_value);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: historicalData.labels,
            datasets: [
                {
                    label: 'Historical Sales',
                    data: historicalData.values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Forecast',
                    data: [...Array(historicalData.labels.length - 5).fill(null), 
                           ...historicalData.forecast],
                    borderColor: '#f5576c',
                    backgroundColor: 'rgba(245, 87, 108, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Upper Bound (95% CI)',
                    data: [...Array(historicalData.labels.length - 5).fill(null), 
                           ...historicalData.upperBound],
                    borderColor: 'rgba(245, 87, 108, 0.3)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Lower Bound (95% CI)',
                    data: [...Array(historicalData.labels.length - 5).fill(null), 
                           ...historicalData.lowerBound],
                    borderColor: 'rgba(245, 87, 108, 0.3)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    fill: '-1',
                    backgroundColor: 'rgba(245, 87, 108, 0.05)',
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Sales (units)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

function generateMockHistoricalData(forecastValue) {
    const numPoints = 20;
    const labels = [];
    const values = [];
    const forecast = [];
    const upperBound = [];
    const lowerBound = [];
    
    const today = new Date();
    
    // Historical data
    for (let i = numPoints - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i * 7);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        
        const baseValue = forecastValue * 0.8;
        const variation = Math.random() * (forecastValue * 0.4);
        values.push(baseValue + variation);
    }
    
    // Forecast data
    for (let i = 1; i <= 5; i++) {
        const date = new Date(today);
        date.setDate(date.getDate() + i * 7);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        
        const forecastVal = forecastValue + (i * 2);
        forecast.push(forecastVal);
        upperBound.push(forecastVal * 1.1);
        lowerBound.push(forecastVal * 0.9);
    }
    
    return { labels, values, forecast, upperBound, lowerBound };
}

// ============================================
// Model Comparison
// ============================================
function initializeCompareModal() {
    const compareBtn = document.getElementById('compareModelsBtn');
    const modal = document.getElementById('compareModal');
    const closeBtn = document.getElementById('closeCompareModal');
    
    if (compareBtn) {
        compareBtn.addEventListener('click', () => {
            showModelComparison();
        });
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    }
    
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }
}

function showModelComparison() {
    const modal = document.getElementById('compareModal');
    const tableBody = document.getElementById('comparisonTableBody');
    
    // Mock comparison data
    const models = [
        { name: 'Ensemble', mae: 2.34, rmse: 3.12, mape: '4.56%', time: '45s', status: '✓ Best' },
        { name: 'Transformer', mae: 2.67, rmse: 3.45, mape: '5.12%', time: '120s', status: '✓' },
        { name: 'LSTM', mae: 2.89, rmse: 3.67, mape: '5.43%', time: '95s', status: '✓' },
        { name: 'GRU', mae: 2.95, rmse: 3.74, mape: '5.67%', time: '88s', status: '✓' },
        { name: 'XGBoost', mae: 3.12, rmse: 4.01, mape: '6.01%', time: '25s', status: '✓' },
        { name: 'LightGBM', mae: 3.18, rmse: 4.08, mape: '6.12%', time: '22s', status: '✓' },
        { name: 'Prophet', mae: 3.45, rmse: 4.32, mape: '6.78%', time: '35s', status: '✓' },
        { name: 'SARIMAX', mae: 3.67, rmse: 4.56, mape: '7.12%', time: '42s', status: '✓' }
    ];
    
    tableBody.innerHTML = models.map(model => `
        <tr>
            <td><strong>${model.name}</strong></td>
            <td>${model.mae}</td>
            <td>${model.rmse}</td>
            <td>${model.mape}</td>
            <td>${model.time}</td>
            <td><span class="status-badge">${model.status}</span></td>
        </tr>
    `).join('');
    
    modal.classList.add('active');
}

// ============================================
// Chart Controls
// ============================================
document.getElementById('toggleUncertainty')?.addEventListener('click', function() {
    if (forecastChart) {
        const datasets = forecastChart.data.datasets;
        datasets[2].hidden = !datasets[2].hidden;
        datasets[3].hidden = !datasets[3].hidden;
        forecastChart.update();
    }
});

document.getElementById('fullscreenChart')?.addEventListener('click', function() {
    const chartContainer = document.querySelector('.chart-card');
    if (chartContainer) {
        if (!document.fullscreenElement) {
            chartContainer.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
});

// ============================================
// AI Explanation Generation (NEW!)
// ============================================
async function generateExplanation(forecastData) {
    const loadingDiv = document.getElementById('explanationLoading');
    const contentDiv = document.getElementById('explanationContent');
    const placeholderDiv = document.getElementById('explanationPlaceholder');
    
    // Show loading state
    placeholderDiv.style.display = 'none';
    contentDiv.style.display = 'none';
    loadingDiv.style.display = 'flex';
    
    try {
        console.log('Generating AI explanation...');
        
        const response = await window.PharmaPredictAI.apiCall('/api/explain', 'POST', {
            category: forecastData.category,
            prediction: forecastData.forecast_value,
            date: forecastData.closest_prediction_date || document.getElementById('date').value,
            model_type: forecastData.model_used
        });
        
        console.log('Explanation response:', response);
        
        if (response.success) {
            displayExplanation(response.explanation);
        } else {
            showExplanationError(response.error || 'Failed to generate explanation');
        }
    } catch (error) {
        console.error('Error generating explanation:', error);
        showExplanationError(error.message);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function displayExplanation(explanationText) {
    const contentDiv = document.getElementById('explanationContent');
    
    // Convert markdown-style text to HTML
    let html = explanationText
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')  // Bold
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')  // H4 headers
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')  // H3 headers
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')  // H2 headers
        .replace(/^• (.+)$/gm, '<li>$1</li>')  // Bullet points
        .replace(/⚠️ (.+)$/gm, '<div class="warning-item">⚠️ $1</div>')  // Warnings
        .replace(/^---$/gm, '<hr>')  // Horizontal rules
        .replace(/\n\n/g, '</p><p>')  // Paragraphs
        .replace(/\n/g, '<br>');  // Line breaks
    
    // Wrap bullet points in ul
    html = html.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');
    
    // Wrap in paragraphs
    html = '<p>' + html + '</p>';
    
    contentDiv.innerHTML = html;
    contentDiv.style.display = 'block';
}

function showExplanationError(errorMsg) {
    const contentDiv = document.getElementById('explanationContent');
    contentDiv.innerHTML = `
        <div class="explanation-error">
            <i class="fas fa-exclamation-triangle"></i>
            <p>Unable to generate explanation: ${errorMsg}</p>
            <button class="btn btn-sm" onclick="location.reload()">Retry</button>
        </div>
    `;
    contentDiv.style.display = 'block';
}

// Refresh explanation button
document.getElementById('refreshExplanation')?.addEventListener('click', async function() {
    const forecastValue = document.getElementById('forecastValue').textContent;
    const category = document.getElementById('forecastCategory').textContent;
    const date = document.getElementById('forecastDate').textContent;
    const model = document.getElementById('forecastModel').textContent;
    
    if (forecastValue && category !== '-') {
        await generateExplanation({
            category: category,
            forecast_value: parseFloat(forecastValue),
            closest_prediction_date: date,
            model_used: model
        });
    }
});

// ============================================
// Export Functionality
// ============================================
document.getElementById('exportBtn')?.addEventListener('click', function() {
    window.PharmaPredictAI.showNotification('Export feature coming soon!', 'info');
});
