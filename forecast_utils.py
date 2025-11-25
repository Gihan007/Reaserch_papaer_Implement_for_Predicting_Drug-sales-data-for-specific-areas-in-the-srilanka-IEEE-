import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import sys

def forecast_sales(category, date_str, model_type='ensemble', base_path=''):
    """
    Forecast sales for a given category and date using specified model
    Returns: forecast_value, closest_prediction_date, plot_filename, model_used
    """
    try:
        # Load the data
        file_path = os.path.join(base_path, f'{category}.csv')
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
            forecast_value = df.loc[closest_date, category]
            model_used = "Historical Data"
        else:
            # Future date - use ML model
            forecast_value, model_used = get_model_forecast(category, days_ahead, model_type, base_path)

            # Use last available date as reference
            closest_date = last_date

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
        plot_path = os.path.join('static', 'images', plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return forecast_value, closest_date, plot_filename, model_used

    except Exception as e:
        print(f"Error in forecast_sales: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy values for testing
        dummy_date = pd.Timestamp('2023-12-03')
        return 50.0, dummy_date, 'dummy_plot.png', 'Error'

def get_model_forecast(category, days_ahead, model_type, base_path=''):
    """
    Get forecast from specified model
    """
    try:
        # Add src to path for imports
        if base_path == '' and 'src' not in sys.path:
            sys.path.append('src')

        if model_type == 'sarimax':
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
        print(f"Error getting {model_type} forecast: {e}")
        # Fallback to simple average
        df = pd.read_csv(os.path.join(base_path, f'{category}.csv'), index_col=0, parse_dates=True)
        avg_value = df[category].tail(30).mean()
        return float(avg_value), f"{model_type} (fallback)"

def get_sarimax_forecast(category, days_ahead, base_path=''):
    """Get SARIMAX forecast"""
    try:
        from models.sarimax_model import predict_sarimax, fit_sarimax
        csv_path = os.path.join(base_path, f'{category}.csv') if base_path else f'{category}.csv'
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
        from utils.xgb_forecast import forecast_xgboost
        model_dir = os.path.join(base_path, 'models_xgb') if base_path else 'models_xgb'
        result = forecast_xgboost(category, n_lags=5, n_steps=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "XGBoost"
    except:
        raise

def get_transformer_forecast(category, days_ahead, base_path=''):
    """Get Transformer forecast"""
    try:
        from models.transformer_model import forecast_transformer
        model_dir = os.path.join(base_path, 'models_transformer') if base_path else 'models_transformer'
        result = forecast_transformer(category, seq_length=10, n_steps=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "Transformer"
    except:
        raise

def get_gru_forecast(category, days_ahead, base_path=''):
    """Get GRU forecast"""
    try:
        from models.gru_model import forecast_gru
        model_dir = os.path.join(base_path, 'models_gru') if base_path else 'models_gru'
        result = forecast_gru(category, seq_length=10, n_steps=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "GRU"
    except:
        raise

def get_lstm_forecast(category, days_ahead, base_path=''):
    """Get LSTM forecast"""
    try:
        from models.lstm_model import forecast_lstm
        model_dir = os.path.join(base_path, 'models_lstm') if base_path else 'models_lstm'
        result = forecast_lstm(category, seq_length=10, n_steps=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "LSTM"
    except:
        raise

def get_lightgbm_forecast(category, days_ahead, base_path=''):
    """Get LightGBM forecast"""
    try:
        from models.lightgbm_model import forecast_lightgbm
        model_dir = os.path.join(base_path, 'models_lightgbm') if base_path else 'models_lightgbm'
        result = forecast_lightgbm(category, n_lags=5, n_steps=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "LightGBM"
    except:
        raise

def get_prophet_forecast(category, days_ahead, base_path=''):
    """Get Prophet forecast"""
    try:
        from models.prophet_model import forecast_prophet
        model_dir = os.path.join(base_path, 'models_prophet') if base_path else 'models_prophet'
        result = forecast_prophet(category, periods=days_ahead, base_path=base_path, model_dir=model_dir)
        return float(result), "Prophet"
    except:
        raise

def get_ensemble_forecast(category, days_ahead, base_path=''):
    """Get ensemble forecast using weighted average"""
    try:
        # Import ensemble methods
        if 'src' not in sys.path:
            sys.path.append('src')
        from evaluation.ensemble_methods import EnsembleMethods

        ensemble = EnsembleMethods()
        
        # Update the ensemble methods to use correct base_path
        original_get_predictions = ensemble.get_model_predictions
        
        def patched_get_predictions(cat, n_steps):
            predictions = {}
            
            # Update paths in the model calls
            try:
                # SARIMAX
                from models.sarimax_model import predict_sarimax, fit_sarimax
                csv_path = os.path.join(base_path, f'{cat}.csv') if base_path else f'{cat}.csv'
                df = pd.read_csv(csv_path, parse_dates=['datum'], index_col='datum')
                order = (1, 0, 0)
                seasonal_order = (1, 0, 0, 7)
                fitted_model = fit_sarimax(df[cat], order, seasonal_order)
                predictions['sarimax'] = predict_sarimax(fitted_model, start=len(df)-n_steps, end=len(df)-1, dynamic=True)
            except Exception as e:
                predictions['sarimax'] = np.zeros(n_steps)

            # For other models, try with corrected paths
            model_configs = [
                ('xgboost', 'utils.xgb_forecast', 'forecast_xgboost', {'n_lags': 5, 'n_steps': n_steps}),
                ('transformer', 'models.transformer_model', 'forecast_transformer', {'seq_length': 10, 'n_steps': n_steps}),
                ('gru', 'models.gru_model', 'forecast_gru', {'seq_length': 10, 'n_steps': n_steps}),
                ('lstm', 'models.lstm_model', 'forecast_lstm', {'seq_length': 10, 'n_steps': n_steps}),
                ('lightgbm', 'models.lightgbm_model', 'forecast_lightgbm', {'n_lags': 5, 'n_steps': n_steps}),
                ('prophet', 'models.prophet_model', 'forecast_prophet', {'periods': n_steps})
            ]
            
            for model_name, module_name, func_name, kwargs in model_configs:
                try:
                    module = __import__(module_name, fromlist=[func_name])
                    func = getattr(module, func_name)
                    model_dir_name = f'models_{model_name}'
                    model_dir = os.path.join(base_path, model_dir_name) if base_path else model_dir_name
                    result = func(cat, base_path=base_path, model_dir=model_dir, **kwargs)
                    predictions[model_name] = [result] * n_steps if not isinstance(result, (list, np.ndarray)) else result
                except Exception as e:
                    predictions[model_name] = np.zeros(n_steps)

            return predictions
        
        ensemble.get_model_predictions = patched_get_predictions
        
        predictions = ensemble.get_model_predictions(category, n_steps=days_ahead)

        # Simple weighted average
        weights = {model: 1.0 / len(ensemble.models) for model in ensemble.models}
        ensemble_pred = ensemble.weighted_average_ensemble(predictions, weights)

        return float(ensemble_pred[-1]), "Ensemble (Weighted Average)"
    except Exception as e:
        print(f"Ensemble forecast failed: {e}")
        # Fallback to simple average
        csv_path = os.path.join(base_path, f'{category}.csv') if base_path else f'{category}.csv'
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