import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import json
import os

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage

    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }

def evaluate_model_performance():
    """Comprehensive model performance evaluation across all categories and models"""

    categories = [f'C{i}' for i in range(1, 9)]
    models = ['sarimax', 'xgboost', 'transformer', 'gru', 'lstm', 'lightgbm', 'prophet']

    results = {}
    training_times = {}
    inference_times = {}

    print("🔍 Starting comprehensive model evaluation...")
    print("=" * 60)

    for category in categories:
        print(f"\n📊 Evaluating Category: {category}")
        results[category] = {}
        training_times[category] = {}
        inference_times[category] = {}

        # Load actual data for comparison
        df = pd.read_csv(f'../{category}.csv', parse_dates=['datum'], index_col='datum')
        actual_values = df[category].values[-10:]  # Last 10 values for testing

        for model in models:
            try:
                print(f"  🤖 Testing {model.upper()}...")

                # Measure inference time
                start_time = time.time()

                # Import the appropriate forecast function
                if model == 'sarimax':
                    from models.sarimax_model import predict_sarimax
                    from models.sarimax_model import fit_sarimax
                    order = (1, 0, 0)
                    seasonal_order = (1, 0, 0, 7)
                    fitted_model = fit_sarimax(df[category], order, seasonal_order)
                    predictions = predict_sarimax(fitted_model, start=len(df)-10, end=len(df)-1, dynamic=True)
                elif model == 'xgboost':
                    from utils.xgb_forecast import forecast_xgboost
                    predictions = forecast_xgboost(category, n_lags=5, n_steps=10, base_path='../', model_dir='../models_xgb/')
                elif model == 'transformer':
                    from models.transformer_model import forecast_transformer
                    predictions = [forecast_transformer(category, seq_length=10, n_steps=i+1, base_path='../', model_dir='../models_transformer/') for i in range(10)]
                elif model == 'gru':
                    from models.gru_model import forecast_gru
                    predictions = [forecast_gru(category, seq_length=10, n_steps=i+1, base_path='../', model_dir='../models_gru/') for i in range(10)]
                elif model == 'lstm':
                    from models.lstm_model import forecast_lstm
                    predictions = [forecast_lstm(category, seq_length=10, n_steps=i+1, base_path='../', model_dir='../models_lstm/') for i in range(10)]
                elif model == 'lightgbm':
                    from models.lightgbm_model import forecast_lightgbm
                    predictions = [forecast_lightgbm(category, n_lags=5, n_steps=i+1, base_path='../', model_dir='../models_lightgbm/') for i in range(10)]
                elif model == 'prophet':
                    from models.prophet_model import forecast_prophet
                    predictions = [forecast_prophet(category, periods=i+1, base_path='../', model_dir='../models_prophet/') for i in range(10)]

                inference_time = time.time() - start_time

                # Calculate metrics
                metrics = calculate_metrics(actual_values, predictions)

                results[category][model] = metrics
                inference_times[category][model] = round(inference_time, 4)

                print(f"    ✅ {model.upper()}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%, Time={inference_time:.4f}s")

            except Exception as e:
                print(f"    ❌ {model.upper()}: Error - {str(e)}")
                results[category][model] = {'MAE': None, 'RMSE': None, 'MAPE': None}
                inference_times[category][model] = None

    # Save results
    os.makedirs('evaluation_results', exist_ok=True)

    with open('evaluation_results/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('evaluation_results/inference_times.json', 'w') as f:
        json.dump(inference_times, f, indent=2)

    # Create summary statistics
    create_performance_summary(results, inference_times)

    print("\n🎉 Evaluation complete! Results saved to 'evaluation_results/' directory")
    return results, inference_times

def create_performance_summary(results, inference_times):
    """Create summary statistics and rankings"""

    models = ['sarimax', 'xgboost', 'transformer', 'gru', 'lstm', 'lightgbm', 'prophet']
    summary = {}

    for model in models:
        summary[model] = {
            'avg_mae': [],
            'avg_rmse': [],
            'avg_mape': [],
            'avg_time': []
        }

    # Calculate averages across categories
    for category in results:
        for model in models:
            if results[category][model]['MAE'] is not None:
                summary[model]['avg_mae'].append(results[category][model]['MAE'])
                summary[model]['avg_rmse'].append(results[category][model]['RMSE'])
                summary[model]['avg_mape'].append(results[category][model]['MAPE'])

        for model in models:
            if inference_times[category][model] is not None:
                summary[model]['avg_time'].append(inference_times[category][model])

    # Calculate final averages
    for model in models:
        if summary[model]['avg_mae']:
            summary[model]['final_mae'] = round(np.mean(summary[model]['avg_mae']), 4)
            summary[model]['final_rmse'] = round(np.mean(summary[model]['avg_rmse']), 4)
            summary[model]['final_mape'] = round(np.mean(summary[model]['avg_mape']), 4)
            summary[model]['final_time'] = round(np.mean(summary[model]['avg_time']), 4)
        else:
            summary[model]['final_mae'] = None
            summary[model]['final_rmse'] = None
            summary[model]['final_mape'] = None
            summary[model]['final_time'] = None

    # Create ranking
    valid_models = [(model, summary[model]['final_mae']) for model in models if summary[model]['final_mae'] is not None]
    valid_models.sort(key=lambda x: x[1])  # Sort by MAE (lower is better)

    summary['ranking'] = {
        'by_mae': [model for model, _ in valid_models],
        'by_rmse': sorted(valid_models, key=lambda x: summary[x[0]]['final_rmse']),
        'by_mape': sorted(valid_models, key=lambda x: summary[x[0]]['final_mape']),
        'by_speed': sorted(valid_models, key=lambda x: summary[x[0]]['final_time'])
    }

    with open('evaluation_results/performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Create visualization
    create_performance_plots(summary)

def create_performance_plots(summary):
    """Create performance comparison plots"""

    models = ['sarimax', 'xgboost', 'transformer', 'gru', 'lstm', 'lightgbm', 'prophet']
    valid_models = [m for m in models if summary[m]['final_mae'] is not None]

    mae_values = [summary[m]['final_mae'] for m in valid_models]
    rmse_values = [summary[m]['final_rmse'] for m in valid_models]
    mape_values = [summary[m]['final_mape'] for m in valid_models]
    time_values = [summary[m]['final_time'] for m in valid_models]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # MAE Plot
    ax1.bar(valid_models, mae_values, color='skyblue')
    ax1.set_title('Mean Absolute Error (MAE) - Lower is Better')
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='x', rotation=45)

    # RMSE Plot
    ax2.bar(valid_models, rmse_values, color='lightcoral')
    ax2.set_title('Root Mean Square Error (RMSE) - Lower is Better')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)

    # MAPE Plot
    ax3.bar(valid_models, mape_values, color='lightgreen')
    ax3.set_title('Mean Absolute Percentage Error (MAPE) - Lower is Better')
    ax3.set_ylabel('MAPE (%)')
    ax3.tick_params(axis='x', rotation=45)

    # Time Plot
    ax4.bar(valid_models, time_values, color='orange')
    ax4.set_title('Inference Time (seconds) - Lower is Better')
    ax4.set_ylabel('Time (s)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('evaluation_results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Performance plots saved to 'evaluation_results/model_comparison.png'")

if __name__ == "__main__":
    evaluate_model_performance()