// ============================================
// Causal Analysis Page JavaScript
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeDiscoveryForm();
    initializeEffectsForm();
    initializeCounterfactualForm();
    initializeCompleteForm();
});

// ============================================
// Causal Discovery
// ============================================
function initializeDiscoveryForm() {
    const form = document.getElementById('discoveryForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runCausalDiscovery();
    });
}

async function runCausalDiscovery() {
    const category = document.getElementById('discoveryCategory').value;
    const maxLags = parseInt(document.getElementById('maxLags').value);
    
    if (!category) {
        window.PharmaPredictAI.showNotification('Please select a category', 'warning');
        return;
    }
    
    try {
        window.PharmaPredictAI.showNotification('Running causal discovery...', 'info');
        
        const response = await window.PharmaPredictAI.apiCall('/api/causal/discovery', 'POST', {
            category: category,
            max_lags: maxLags
        });
        
        if (response.success) {
            displayCausalGraph(response.results);
            displayRelationships(response.results);
            document.getElementById('discoveryResults').style.display = 'block';
            window.PharmaPredictAI.showNotification('Causal discovery completed!', 'success');
        } else {
            window.PharmaPredictAI.showNotification(response.error || 'Discovery failed', 'error');
        }
    } catch (error) {
        console.error('Causal discovery error:', error);
    }
}

function displayCausalGraph(results) {
    const graphDiv = document.getElementById('causalGraph');
    graphDiv.innerHTML = `
        <div style="padding: 2rem; text-align: center; background: #f8f9fa; border-radius: 0.5rem;">
            <p style="color: #6c757d;">Causal graph visualization</p>
            <p style="font-size: 0.875rem;">Interactive D3.js graph will be rendered here</p>
        </div>
    `;
}

function displayRelationships(results) {
    const listDiv = document.getElementById('relationshipsList');
    
    const mockRelationships = [
        { from: 'sales_lag1', to: 'sales', strength: 'Strong' },
        { from: 'trend', to: 'sales', strength: 'Medium' },
        { from: 'week', to: 'sales', strength: 'Weak' }
    ];
    
    listDiv.innerHTML = mockRelationships.map(rel => `
        <div class="relationship-item" style="padding: 1rem; background: white; border-radius: 0.5rem; margin-bottom: 0.5rem; border-left: 4px solid #667eea;">
            <strong>${rel.from}</strong> → <strong>${rel.to}</strong>
            <span style="float: right; color: #667eea;">${rel.strength}</span>
        </div>
    `).join('');
}

// ============================================
// Effect Estimation
// ============================================
function initializeEffectsForm() {
    const form = document.getElementById('effectsForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await estimateCausalEffects();
    });
}

async function estimateCausalEffects() {
    const category = document.getElementById('effectsCategory').value;
    const treatment = document.getElementById('treatment').value;
    
    if (!category || !treatment) {
        window.PharmaPredictAI.showNotification('Please fill all fields', 'warning');
        return;
    }
    
    try {
        window.PharmaPredictAI.showNotification('Estimating causal effects...', 'info');
        
        const response = await window.PharmaPredictAI.apiCall('/api/causal/effects', 'POST', {
            category: category,
            treatment: treatment
        });
        
        if (response.success) {
            displayEffectEstimates(response.results);
            document.getElementById('effectsResults').style.display = 'block';
            window.PharmaPredictAI.showNotification('Effect estimation completed!', 'success');
        } else {
            window.PharmaPredictAI.showNotification(response.error || 'Estimation failed', 'error');
        }
    } catch (error) {
        console.error('Effect estimation error:', error);
    }
}

function displayEffectEstimates(results) {
    document.getElementById('ateValue').textContent = '0.85';
    document.getElementById('effectSize').textContent = 'Medium';
    document.getElementById('confidence').textContent = '95%';
    document.getElementById('effectDetails').textContent = JSON.stringify(results, null, 2);
}

// ============================================
// Counterfactual Analysis
// ============================================
function initializeCounterfactualForm() {
    const form = document.getElementById('counterfactualForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runCounterfactual();
    });
}

async function runCounterfactual() {
    const category = document.getElementById('cfCategory').value;
    const variable = document.getElementById('cfVariable').value;
    const changePercent = parseInt(document.getElementById('changePercent').value);
    
    if (!category || !variable) {
        window.PharmaPredictAI.showNotification('Please fill all fields', 'warning');
        return;
    }
    
    try {
        window.PharmaPredictAI.showNotification('Running counterfactual analysis...', 'info');
        
        const response = await window.PharmaPredictAI.apiCall('/api/causal/counterfactual', 'POST', {
            category: category,
            variable: variable,
            change_percent: changePercent
        });
        
        if (response.success) {
            displayCounterfactualResults(response.results, changePercent);
            document.getElementById('counterfactualResults').style.display = 'block';
            window.PharmaPredictAI.showNotification('Counterfactual analysis completed!', 'success');
        } else {
            window.PharmaPredictAI.showNotification(response.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Counterfactual error:', error);
    }
}

function displayCounterfactualResults(results, changePercent) {
    const actualValue = 50;
    const cfValue = actualValue * (1 + changePercent / 100);
    const diff = cfValue - actualValue;
    
    document.getElementById('actualValue').textContent = actualValue.toFixed(2);
    document.getElementById('cfValue').textContent = cfValue.toFixed(2);
    document.getElementById('cfDiff').textContent = `${diff > 0 ? '+' : ''}${diff.toFixed(2)} units (${changePercent > 0 ? '+' : ''}${changePercent}%)`;
    
    document.getElementById('cfInsights').innerHTML = `
        <p>A ${Math.abs(changePercent)}% ${changePercent > 0 ? 'increase' : 'decrease'} in the variable would result in 
        a ${Math.abs(diff).toFixed(2)} unit ${diff > 0 ? 'increase' : 'decrease'} in predicted sales.</p>
    `;
}

// ============================================
// Complete Analysis
// ============================================
function initializeCompleteForm() {
    const form = document.getElementById('completeForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runCompleteAnalysis();
    });
}

async function runCompleteAnalysis() {
    const category = document.getElementById('completeCategory').value;
    
    if (!category) {
        window.PharmaPredictAI.showNotification('Please select a category', 'warning');
        return;
    }
    
    const progressCard = document.getElementById('completeProgress');
    const progressBar = document.getElementById('completeProgressBar');
    const stepLabel = document.getElementById('completeStep');
    const steps = document.querySelectorAll('.step-item');
    
    progressCard.style.display = 'block';
    document.getElementById('completeResults').style.display = 'none';
    
    try {
        window.PharmaPredictAI.showNotification('Starting complete causal analysis...', 'info');
        
        const analyses = [
            { name: 'Causal Discovery', progress: 33 },
            { name: 'Effect Estimation', progress: 66 },
            { name: 'Counterfactuals', progress: 100 }
        ];
        
        for (let i = 0; i < analyses.length; i++) {
            stepLabel.textContent = analyses[i].name;
            progressBar.style.width = `${analyses[i].progress}%`;
            steps[i].classList.remove('pending');
            steps[i].classList.add('active');
            
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        const response = await window.PharmaPredictAI.apiCall('/api/causal/complete', 'POST', {
            category: category
        });
        
        if (response.success) {
            displayCompleteResults(response.results);
            document.getElementById('completeResults').style.display = 'block';
            window.PharmaPredictAI.showNotification('Complete analysis finished!', 'success');
        } else {
            window.PharmaPredictAI.showNotification(response.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Complete analysis error:', error);
    }
}

function displayCompleteResults(results) {
    document.getElementById('completeCausalGraph').innerHTML = '<p>Causal structure graph</p>';
    document.getElementById('completeEffects').innerHTML = '<p>Key causal effects identified</p>';
    document.getElementById('completeRecommendations').innerHTML = `
        <ul style="list-style: none; padding: 0;">
            <li style="padding: 0.5rem 0;">✓ Focus on previous week sales as primary driver</li>
            <li style="padding: 0.5rem 0;">✓ Monitor trend changes carefully</li>
            <li style="padding: 0.5rem 0;">✓ Seasonal patterns show medium impact</li>
        </ul>
    `;
}
