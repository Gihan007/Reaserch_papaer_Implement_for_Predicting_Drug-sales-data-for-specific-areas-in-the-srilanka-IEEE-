import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_ARTIFACT_DIR = ROOT_DIR / "artifacts" / "models"
STATIC_IMAGES_DIR = ROOT_DIR / "services" / "frontend_service" / "app" / "static" / "images"

for import_path in (ROOT_DIR, ROOT_DIR / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def _category_csv_path(category, base_path=''):
    if base_path:
        candidate = Path(base_path) / f'{category}.csv'
        if candidate.exists():
            return candidate

    return DATA_DIR / f'{category}.csv'


def _legacy_data_base_path(category, base_path=''):
    return str(_category_csv_path(category, base_path).parent) + os.sep


def _model_dir(model_name, base_path=''):
    dir_aliases = {
        "xgboost": "xgb",
    }
    model_dir_name = f"models_{dir_aliases.get(model_name, model_name)}"

    if base_path:
        candidate = Path(base_path) / "artifacts" / "models" / model_dir_name
        if candidate.exists():
            return str(candidate) + os.sep

        candidate = Path(base_path) / model_dir_name
        if candidate.exists():
            return str(candidate) + os.sep

    return str(MODEL_ARTIFACT_DIR / model_dir_name) + os.sep

def forecast_sales(category, date_str, model_type='ensemble', base_path=''):
    """
    Forecast sales for a given category and date using specified model
    Returns: forecast_value, closest_prediction_date, plot_filename, model_used
    """
    try:
        # Load the data
        file_path = _category_csv_path(category, base_path)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Parse input date
        input_date = pd.to_datetime(date_str)

        # For future dates, we need to forecast
        # For past dates, we can compare with actuals
        last_date = df.index.max()
        days_ahead = (input_date - last_date).days

        if days_ahead <= 0:
            # Historical date - return actual value
            closest_date = df.index[df.index <= input_date].max()
            if pd.isna(closest_date):
                closest_date = df.index.min()
            forecast_value = float(df.loc[closest_date, category])
            model_used = "Historical Data"
        else:
            # Future date - use ML model
            forecast_value, model_used = get_model_forecast(category, days_ahead, model_type, base_path)
            forecast_value = float(forecast_value)

            # Use last available date as reference
            closest_date = last_date
            if model_type == 'stacking':
                closest_date = last_date + pd.Timedelta(weeks=(days_ahead + 6) // 7)
                input_date = closest_date

        # Generate plot
        plt.figure(figsize=(12, 8))

        # Plot historical data
        plt.plot(df.index, df[category], label=f'{category} Historical Sales', linewidth=2)

        # Plot forecast point
        plt.scatter([input_date], [forecast_value], color='red', s=100, zorder=5,
                   label=f'Forecast ({model_used}): {forecast_value:.2f}')

        # Add vertical line for forecast date
        plt.axvline(x=input_date, color='red', linestyle='--', alpha=0.7,
                   label=f'Forecast Date: {input_date.strftime("%Y-%m-%d")}')

        # Add vertical line for last historical date
        plt.axvline(x=last_date, color='blue', linestyle='--', alpha=0.7,
                   label=f'Last Historical: {last_date.strftime("%Y-%m-%d")}')

        plt.title(f'{category} Drug Sales Forecast - {model_used}', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales Volume', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_filename = f'{category}_{date_str.replace("-", "_")}_{model_type}_forecast.png'
        plot_path = STATIC_IMAGES_DIR / plot_filename
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return forecast_value, closest_date, plot_filename, model_used

    except Exception as e:
        if model_type == 'stacking':
            raise
        print(f"Error in forecast_sales: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy values for testing
        dummy_date = pd.Timestamp('2023-12-03')
        return 50.0, dummy_date, 'dummy_plot.png', 'Error'

def stacking_response_details(category, forecast_value, prediction_date):
    """Actual chart data and separately labelled held-out metrics for the UI."""
    import json
    from src.models.stacking_model import load_weekly_series, series_fingerprint
    dates, values = load_weekly_series(category)
    evaluation = None
    path = ROOT_DIR / 'src/evaluation_results/stacking_results.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        entry = report.get('categories', {}).get(category, {})
        if entry.get('status') == 'ok' and entry.get('dataset_sha256') == series_fingerprint(dates, values):
            evaluation = entry['metrics']['stacking']
    return {
        'model_type': 'stacking', 'evaluation_metrics': evaluation,
        'evaluation_label': 'Separate chronological holdout; not accuracy of this individual forecast',
        'chart_data': {'dates': [str(d.date()) for d in dates], 'actual': values.tolist(),
                       'forecast_date': str(pd.Timestamp(prediction_date).date()),
                       'forecast_value': float(forecast_value)},
    }


def get_model_forecast(category, days_ahead, model_type, base_path=''):
    """
    Get forecast from specified model
    """
    try:
        if model_type == 'stacking':
            from src.models.stacking_model import forecast_stacking
            # The models operate on weekly rows, not individual calendar days.
            steps = (days_ahead + 6) // 7
            values = forecast_stacking(
                category, steps, data_dir=_category_csv_path(category, base_path).parent,
                model_dir=_model_dir('stacking', base_path),
            )
            return float(values[-1]), "Stacking (XGBoost + LSTM + GRU / Ridge)"
        elif model_type == 'sarimax':
            return get_sarimax_forecast(category, days_ahead, base_path)
        elif model_type == 'xgboost':
            return get_xgboost_forecast(category, days_ahead, base_path)
        elif model_type == 'transformer':
            return get_transformer_forecast(category, days_ahead, base_path)
        elif model_type == 'gru':
            return get_gru_forecast(category, days_ahead, base_path)
        elif model_type == 'lstm':
            return get_lstm_forecast(category, days_ahead, base_path)
        elif model_type == 'lightgbm':
            return get_lightgbm_forecast(category, days_ahead, base_path)
        elif model_type == 'prophet':
            return get_prophet_forecast(category, days_ahead, base_path)
        elif model_type == 'ensemble':
            return get_ensemble_forecast(category, days_ahead, base_path)
        else:
            # Default to ensemble
            return get_ensemble_forecast(category, days_ahead, base_path)

    except Exception as e:
        if model_type == 'stacking':
            raise
        print(f"Error getting {model_type} forecast: {e}")
        # Fallback to simple average
        df = pd.read_csv(_category_csv_path(category, base_path), index_col=0, parse_dates=True)
        avg_value = df[category].tail(30).mean()
        return float(avg_value), f"{model_type} (fallback)"

def get_sarimax_forecast(category, days_ahead, base_path=''):
    """Get SARIMAX forecast"""
    try:
        from src.models.sarimax_model import predict_sarimax, fit_sarimax
        csv_path = _category_csv_path(category, base_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        order = (1, 0, 0)
        seasonal_order = (1, 0, 0, 7)
        fitted_model = fit_sarimax(df[category], order, seasonal_order)
        predictions = predict_sarimax(fitted_model, start=len(df), end=len(df)+days_ahead-1, dynamic=True)
        return float(predictions.iloc[-1]), "SARIMAX"
    except:
        raise

def get_xgboost_forecast(category, days_ahead, base_path=''):
    """Get XGBoost forecast"""
    try:
        from src.utils.xgb_forecast import forecast_xgboost
        model_dir = _model_dir('xgb', base_path)
        result = forecast_xgboost(category, n_lags=5, n_steps=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        # Handle list or array results
        if isinstance(result, (list, np.ndarray)):
            result = result[-1] if len(result) > 0 else result[0]
        return float(result), "XGBoost"
    except:
        raise

def get_transformer_forecast(category, days_ahead, base_path=''):
    """Get Transformer forecast"""
    try:
        from src.models.transformer_model import forecast_transformer
        model_dir = _model_dir('transformer', base_path)
        result = forecast_transformer(category, seq_length=10, n_steps=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        return float(result), "Transformer"
    except:
        raise

def get_gru_forecast(category, days_ahead, base_path=''):
    """Get GRU forecast"""
    try:
        from src.models.gru_model import forecast_gru
        model_dir = _model_dir('gru', base_path)
        result = forecast_gru(category, seq_length=10, n_steps=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        return float(result), "GRU"
    except:
        raise

def get_lstm_forecast(category, days_ahead, base_path=''):
    """Get LSTM forecast"""
    try:
        from src.models.lstm_model import forecast_lstm
        model_dir = _model_dir('lstm', base_path)
        result = forecast_lstm(category, seq_length=10, n_steps=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        return float(result), "LSTM"
    except:
        raise

def get_lightgbm_forecast(category, days_ahead, base_path=''):
    """Get LightGBM forecast"""
    try:
        from src.models.lightgbm_model import forecast_lightgbm
        model_dir = _model_dir('lightgbm', base_path)
        result = forecast_lightgbm(category, n_lags=5, n_steps=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        # Handle list or array results
        if isinstance(result, (list, np.ndarray)):
            result = result[-1] if len(result) > 0 else result[0]
        return float(result), "LightGBM"
    except:
        raise

def get_prophet_forecast(category, days_ahead, base_path=''):
    """Get Prophet forecast"""
    try:
        from src.models.prophet_model import forecast_prophet
        model_dir = _model_dir('prophet', base_path)
        result = forecast_prophet(category, periods=days_ahead, base_path=_legacy_data_base_path(category, base_path), model_dir=model_dir)
        # Handle list or array results
        if isinstance(result, (list, np.ndarray)):
            result = result[-1] if len(result) > 0 else result[0]
        return float(result), "Prophet"
    except:
        raise

def get_ensemble_forecast(category, days_ahead, base_path=''):
    """Get ensemble forecast using weighted average"""
    try:
        def patched_get_predictions(cat, n_steps):
            predictions = {}
            
            # Update paths in the model calls
            try:
                # SARIMAX
                from src.models.sarimax_model import predict_sarimax, fit_sarimax
                csv_path = _category_csv_path(cat, base_path)
                df = pd.read_csv(csv_path, parse_dates=['datum'], index_col='datum')
                order = (1, 0, 0)
                seasonal_order = (1, 0, 0, 7)
                fitted_model = fit_sarimax(df[cat], order, seasonal_order)
                predictions['sarimax'] = predict_sarimax(fitted_model, start=len(df)-n_steps, end=len(df)-1, dynamic=True)
            except Exception as e:
                predictions['sarimax'] = np.zeros(n_steps)

            # For other models, try with corrected paths
            model_configs = [
                ('xgboost', 'src.utils.xgb_forecast', 'forecast_xgboost', {'n_lags': 5, 'n_steps': n_steps}),
                ('transformer', 'src.models.transformer_model', 'forecast_transformer', {'seq_length': 10, 'n_steps': n_steps}),
                ('gru', 'src.models.gru_model', 'forecast_gru', {'seq_length': 10, 'n_steps': n_steps}),
                ('lstm', 'src.models.lstm_model', 'forecast_lstm', {'seq_length': 10, 'n_steps': n_steps}),
                ('lightgbm', 'src.models.lightgbm_model', 'forecast_lightgbm', {'n_lags': 5, 'n_steps': n_steps}),
                ('prophet', 'src.models.prophet_model', 'forecast_prophet', {'periods': n_steps})
            ]
            
            for model_name, module_name, func_name, kwargs in model_configs:
                try:
                    module = __import__(module_name, fromlist=[func_name])
                    func = getattr(module, func_name)
                    model_dir = _model_dir(model_name, base_path)
                    result = func(cat, base_path=_legacy_data_base_path(cat, base_path), model_dir=model_dir, **kwargs)
                    predictions[model_name] = [result] * n_steps if not isinstance(result, (list, np.ndarray)) else result
                except Exception as e:
                    predictions[model_name] = np.zeros(n_steps)

            return predictions

        predictions = patched_get_predictions(category, n_steps=days_ahead)

        # Convert all predictions to single values and calculate simple average
        valid_predictions = []
        for model_name, pred in predictions.items():
            try:
                if isinstance(pred, (list, np.ndarray)):
                    if len(pred) > 0:
                        valid_predictions.append(float(pred[-1]))
                else:
                    valid_predictions.append(float(pred))
            except:
                pass
        
        if len(valid_predictions) > 0:
            ensemble_value = np.mean(valid_predictions)
            return float(ensemble_value), "Ensemble (Weighted Average)"
        else:
            raise Exception("No valid predictions from models")
    except Exception as e:
        print(f"Ensemble forecast failed: {e}")
        # Fallback to simple average
        csv_path = _category_csv_path(category, base_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        avg_value = df[category].tail(30).mean()
        return float(avg_value), "Ensemble (Fallback)"

def generate_plot(category, date_str, model_type='ensemble', base_path=''):
    """
    Generate plot for the forecast
    """
    try:
        forecast_value, closest_date, plot_file, model_used = forecast_sales(category, date_str, model_type, base_path)
        return plot_file
    except Exception as e:
        print(f"Error in generate_plot: {e}")
        return 'error_plot.png'
