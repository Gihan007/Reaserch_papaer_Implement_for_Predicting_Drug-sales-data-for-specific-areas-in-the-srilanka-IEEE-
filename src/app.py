
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import train_xgboost_models, train_transformer_models, train_gru_models, train_lstm_models, train_lightgbm_models, train_prophet_models, train_tft_models, train_nbeats_models, train_informer_models
from models.sarimax_model import fit_sarimax, predict_sarimax
from utils.xgb_forecast import forecast_xgboost
from models.transformer_model import forecast_transformer
from models.gru_model import forecast_gru
from models.lstm_model import forecast_lstm
from models.lightgbm_model import forecast_lightgbm
from models.prophet_model import forecast_prophet
from models.tft_model import forecast_tft
from models.nbeats_model import forecast_nbeats
from models.informer_model import forecast_informer
from evaluation.model_evaluation import evaluate_model_performance
from evaluation.ensemble_methods import EnsembleMethods, evaluate_ensembles
from evaluation.hyperparameter_optimization import HyperparameterOptimizer
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import torch
import json
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
PLOT_FOLDER = os.path.join('static', 'images')
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Train XGBoost models if not present
def ensure_xgboost_models():
    model_dir = 'models_xgb/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_xgb.pkl')):
            missing = True
            break
    if missing:
        train_xgboost_models(categories, base_path='./', model_dir='models_xgb/', n_lags=5)

# Train Transformer models if not present
def ensure_transformer_models():
    model_dir = 'models_transformer/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_transformer.pth')):
            missing = True
            break
    if missing:
        train_transformer_models(categories, base_path='./', model_dir='models_transformer/', seq_length=10)

# Train GRU models if not present
def ensure_gru_models():
    model_dir = 'models_gru/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_gru.pth')):
            missing = True
            break
    if missing:
        train_gru_models(categories, base_path='./', model_dir='models_gru/', seq_length=10)

# Train LSTM models if not present
def ensure_lstm_models():
    model_dir = 'models_lstm/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_lstm.pth')):
            missing = True
            break
    if missing:
        train_lstm_models(categories, base_path='./', model_dir='models_lstm/', seq_length=10)

# Train LightGBM models if not present
def ensure_lightgbm_models():
    model_dir = 'models_lightgbm/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_lightgbm.txt')):
            missing = True
            break
    if missing:
        train_lightgbm_models(categories, base_path='./', model_dir='models_lightgbm/', n_lags=5)

# Train Prophet models if not present
def ensure_prophet_models():
    model_dir = 'models_prophet/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'{cat}_prophet.pkl')):
            missing = True
            break
    if missing:
        train_prophet_models(categories, base_path='./', model_dir='models_prophet/')

# Train TFT models if not present
def ensure_tft_models():
    model_dir = 'models_tft/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'tft_{cat}.pth')):
            missing = True
            break
    if missing:
        train_tft_models(categories, base_path='./', model_dir='models_tft/', seq_length=30)

# Train N-BEATS models if not present
def ensure_nbeats_models():
    model_dir = 'models_nbeats/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'nbeats_{cat}.pth')):
            missing = True
            break
    if missing:
        train_nbeats_models(categories, base_path='./', model_dir='models_nbeats/', seq_length=30)

# Train Informer models if not present
def ensure_informer_models():
    model_dir = 'models_informer/'
    categories = [f'C{i}' for i in range(1, 9)]
    missing = False
    for cat in categories:
        if not os.path.exists(os.path.join(model_dir, f'informer_{cat}.pth')):
            missing = True
            break
    if missing:
        train_informer_models(categories, base_path='./', model_dir='models_informer/', seq_length=30)

ensure_xgboost_models()
ensure_transformer_models()
ensure_gru_models()
ensure_lstm_models()
ensure_lightgbm_models()
ensure_prophet_models()
ensure_tft_models()
ensure_nbeats_models()
ensure_informer_models()

# Initialize ensemble methods
ensemble_methods = EnsembleMethods()

@app.route('/evaluate_models', methods=['POST'])
def evaluate_models():
    """Run comprehensive model evaluation"""
    try:
        results, inference_times = evaluate_model_performance()
        return jsonify({
            "status": "success",
            "message": "Model evaluation completed",
            "results": results,
            "inference_times": inference_times
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_evaluation_results', methods=['GET'])
def get_evaluation_results():
    """Get stored evaluation results"""
    try:
        results_path = 'evaluation_results/model_metrics.json'
        summary_path = 'evaluation_results/performance_summary.json'

        if os.path.exists(results_path) and os.path.exists(summary_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            return jsonify({
                "status": "success",
                "results": results,
                "summary": summary
            })
        else:
            return jsonify({"status": "error", "message": "Evaluation results not found. Run evaluation first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/evaluate_ensembles', methods=['POST'])
def evaluate_ensembles_endpoint():
    """Run ensemble methods evaluation"""
    try:
        results = evaluate_ensembles()
        return jsonify({
            "status": "success",
            "message": "Ensemble evaluation completed",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_ensemble_results', methods=['GET'])
def get_ensemble_results():
    """Get stored ensemble results"""
    try:
        results_path = 'evaluation_results/ensemble_results.json'

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)

            return jsonify({
                "status": "success",
                "results": results
            })
        else:
            return jsonify({"status": "error", "message": "Ensemble results not found. Run ensemble evaluation first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/forecast_ensemble', methods=['POST'])
def forecast_ensemble():
    """Generate ensemble forecast"""
    try:
        category = request.form['category']
        method = request.form.get('method', 'weighted_average')
        n_steps = int(request.form.get('n_steps', 10))

        predictions = ensemble_methods.create_ensemble_predictions(category, method, n_steps)

        # Load actual data for plotting
        df = pd.read_csv(f'./{category}.csv', parse_dates=['datum'], index_col='datum')
        actual_values = df[category].values[-n_steps:]

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mape = mean_absolute_percentage_error(actual_values, predictions) * 100

        # Create plot
        plot_file = f'{category}_ensemble_{method}_forecast.png'
        plot_path = os.path.join(PLOT_FOLDER, plot_file)

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual_values)), actual_values, label='Actual', marker='o')
        plt.plot(range(len(predictions)), predictions, label=f'Ensemble ({method})', marker='s', linestyle='--')
        plt.title(f'{category} Ensemble Forecast - {method.upper()}')
        plt.xlabel('Time Steps')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path)
        plt.close()

        return jsonify({
            "status": "success",
            "category": category,
            "method": method,
            "predictions": predictions.tolist(),
            "metrics": {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "MAPE": round(mape, 4)
            },
            "plot_url": f'/static/images/{plot_file}'
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/optimize_hyperparameters', methods=['POST'])
def optimize_hyperparameters():
    """Run hyperparameter optimization"""
    try:
        n_trials = int(request.form.get('n_trials', 50))
        optimizer = HyperparameterOptimizer()
        best_params = optimizer.run_optimization(n_trials=n_trials)

        return jsonify({
            "status": "success",
            "message": "Hyperparameter optimization completed",
            "best_params": best_params
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_optimization_results', methods=['GET'])
def get_optimization_results():
    """Get stored optimization results"""
    try:
        params_path = 'evaluation_results/optimized_params.json'
        results_path = 'evaluation_results/optimization_results.json'

        if os.path.exists(params_path) and os.path.exists(results_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
            with open(results_path, 'r') as f:
                results = json.load(f)

            return jsonify({
                "status": "success",
                "params": params,
                "results": results
            })
        else:
            return jsonify({"status": "error", "message": "Optimization results not found. Run optimization first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_model_comparison_plot', methods=['GET'])
def get_model_comparison_plot():
    """Get the model comparison plot"""
    plot_path = 'evaluation_results/model_comparison.png'
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Comparison plot not found"}), 404

@app.route('/')
def index():
    return jsonify({"message": "Drug Sales Prediction API"})

@app.route('/forecast', methods=['POST'])
def forecast():
    category = request.form['category']
    date = request.form['date']
    model_type = request.form.get('model_type', 'sarimax')
    week_offset = int(request.form.get('week_offset', 1))
    input_date = pd.to_datetime(date)

    # Load data for the selected category
    df = pd.read_csv(f'./{category}.csv', parse_dates=['datum'], index_col='datum')
    closest_prediction_date = df.index[-1]
    forecast_value = None
    plot_file = f'{category}_{model_type}_forecast.png'
    plot_path = os.path.join(PLOT_FOLDER, plot_file)

    if model_type == 'sarimax':
        order = (1, 0, 0)
        seasonal_order = (1, 0, 0, 7)
        results = fit_sarimax(df[category], order, seasonal_order)
        forecast = predict_sarimax(results, start=len(df), end=len(df)+week_offset-1, dynamic=True)
        forecast_value = forecast.iloc[-1]
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='red', label='Forecast')
        plt.legend()
        plt.title(f'{category} SARIMAX Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'xgboost':
        n_lags = 5
        n_steps = week_offset
        preds = forecast_xgboost(category, n_lags, n_steps, base_path='./', model_dir='models_xgb/')
        forecast_value = preds[-1]
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='green', label='XGBoost Forecast')
        plt.legend()
        plt.title(f'{category} XGBoost Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'transformer':
        seq_length = 10
        forecast_value = forecast_transformer(category, seq_length, week_offset, base_path='./', model_dir='models_transformer/')
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='blue', label='Transformer Forecast')
        plt.legend()
        plt.title(f'{category} Transformer Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'gru':
        seq_length = 10
        forecast_value = forecast_gru(category, seq_length, week_offset, base_path='./', model_dir='models_gru/')
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='purple', label='GRU Forecast')
        plt.legend()
        plt.title(f'{category} GRU Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'lstm':
        seq_length = 10
        forecast_value = forecast_lstm(category, seq_length, week_offset, base_path='./', model_dir='models_lstm/')
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='orange', label='LSTM Forecast')
        plt.legend()
        plt.title(f'{category} LSTM Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'lightgbm':
        n_lags = 5
        forecast_value = forecast_lightgbm(category, n_lags, week_offset, base_path='./', model_dir='models_lightgbm/')
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='cyan', label='LightGBM Forecast')
        plt.legend()
        plt.title(f'{category} LightGBM Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'prophet':
        forecast_value = forecast_prophet(category, periods=week_offset, base_path='./', model_dir='models_prophet/')
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='magenta', label='Prophet Forecast')
        plt.legend()
        plt.title(f'{category} Prophet Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'tft':
        seq_length = 30
        forecast_value = forecast_tft(category, n_steps=week_offset, base_path='./', model_dir='models_tft/', seq_length=seq_length)
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='brown', label='TFT Forecast')
        plt.legend()
        plt.title(f'{category} Temporal Fusion Transformer Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'nbeats':
        seq_length = 30
        forecast_value = forecast_nbeats(category, n_steps=week_offset, base_path='./', model_dir='models_nbeats/', seq_length=seq_length)
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='pink', label='N-BEATS Forecast')
        plt.legend()
        plt.title(f'{category} N-BEATS Forecast')
        plt.savefig(plot_path)
        plt.close()
    elif model_type == 'informer':
        seq_length = 30
        forecast_value = forecast_informer(category, n_steps=week_offset, base_path='./', model_dir='models_informer/', seq_length=seq_length)
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[category], label='Actual')
        plt.axvline(x=closest_prediction_date, color='gray', linestyle='--')
        plt.scatter([closest_prediction_date + pd.Timedelta(weeks=week_offset)], [forecast_value], color='olive', label='Informer Forecast')
        plt.legend()
        plt.title(f'{category} Informer Forecast')
        plt.savefig(plot_path)
        plt.close()

    return jsonify({
        "forecast_value": float(forecast_value),
        "closest_prediction_date": str(closest_prediction_date.date()),
        "category": category,
        "model_type": model_type,
        "week_offset": week_offset,
        "plot_url": f'/static/images/{plot_file}'
    })

@app.route('/static/images/<filename>')
def plot_image(filename):
    plot_path = os.path.join('static', 'images', filename)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Plot not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
