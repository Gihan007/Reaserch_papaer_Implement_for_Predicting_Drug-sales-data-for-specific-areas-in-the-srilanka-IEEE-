const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

test('stacking renders actual history and measured scores without simulated output', () => {
    const elements = new Map();
    const element = key => {
        if (!elements.has(key)) elements.set(key, {
            style: {}, textContent: '', scrollIntoView() {}, addEventListener() {},
            insertAdjacentElement(position, el) { elements.set(el.id, el); }
        });
        return elements.get(key);
    };
    let chart;
    const context = vm.createContext({
        console,
        document: {
            addEventListener() {}, getElementById: element, querySelector: element,
            createElement() { return {style: {}, textContent: ''}; }
        },
        window: {PharmaPredictAI: {formatNumber: String, formatDate: String}},
        Chart: function(node, config) { chart = config; this.destroy = () => {}; }
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../services/frontend_service/app/static/js/forecast.js'), 'utf8'), context);
    vm.runInContext(`
        updateMetrics = updateConfidenceIntervals = generateExplanation = createForecastChart = () => {
            throw new Error('Stacking must not call the placeholder result paths');
        };
        displayForecastResults({
            model_type: 'stacking', model_used: 'Stacking', category: 'C1',
            forecast_value: 9, closest_prediction_date: '2023-12-10',
            evaluation_metrics: {MAE: 2.5, RMSE: 3, MAPE: null},
            evaluation_label: 'Separate holdout',
            chart_data: {dates: ['2023-11-26', '2023-12-03'], actual: [7, 8],
                         forecast_date: '2023-12-10', forecast_value: 9}
        });
    `, context);
    assert.equal(element('metricMAE').textContent, '2.5000');
    assert.equal(element('metricMAPE').textContent, 'Unavailable');
    assert.equal(element('.confidence-card').style.display, 'none');
    assert.equal(element('stackingMetricCaption').textContent, 'Separate holdout');
    assert.deepEqual(Array.from(chart.data.datasets[0].data), [7, 8, null]);
    assert.deepEqual(Array.from(chart.data.datasets[1].data), [null, null, 9]);
    assert.equal(chart.data.datasets.length, 2);
});
