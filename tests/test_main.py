import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from forecast_utils import forecast_sales, get_model_forecast
from models.meta_learning import MetaLearningSystem
from evaluation.ensemble_methods import EnsembleMethods


class TestForecastUtils:
    """Test forecast utility functions"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        dates = pd.date_range('2020-01-01', periods=100, freq='W')
        values = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({'C1': values}, index=dates)
        return df

    def test_forecast_sales_historical(self, sample_data):
        """Test forecasting for historical dates"""
        # Save sample data
        sample_data.to_csv('C1.csv')

        try:
            forecast_value, date, plot_file, model_used = forecast_sales(
                'C1', '2020-06-01', 'ensemble'
            )

            assert isinstance(forecast_value, (int, float))
            assert model_used == "Historical Data"
            assert os.path.exists(f"static/images/{plot_file}")
        finally:
            # Cleanup
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')
            if os.path.exists(f"static/images/{plot_file}"):
                os.remove(f"static/images/{plot_file}")

    def test_forecast_sales_future(self, sample_data):
        """Test forecasting for future dates"""
        sample_data.to_csv('C1.csv')

        try:
            forecast_value, date, plot_file, model_used = forecast_sales(
                'C1', '2025-01-01', 'ensemble'
            )

            assert isinstance(forecast_value, (int, float))
            assert "Ensemble" in model_used
        finally:
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')
            if os.path.exists(f"static/images/{plot_file}"):
                os.remove(f"static/images/{plot_file}")

    @pytest.mark.parametrize("model_type", [
        'ensemble', 'xgboost', 'transformer', 'gru', 'lstm',
        'lightgbm', 'prophet', 'sarimax'
    ])
    def test_model_selection(self, model_type):
        """Test that all model types are handled"""
        # This should not raise an exception even if models aren't trained
        try:
            result = get_model_forecast('C1', 30, model_type)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], (int, float))
            assert isinstance(result[1], str)
        except Exception as e:
            # Models might fail if not trained, but should have fallback
            assert "fallback" in str(e) or "Error getting" in str(e)


class TestMetaLearning:
    """Test meta-learning functionality"""

    def test_meta_learning_initialization(self):
        """Test MetaLearningSystem initialization"""
        meta_sys = MetaLearningSystem()
        assert hasattr(meta_sys, 'maml_model')
        assert hasattr(meta_sys, 'transfer_models')
        assert hasattr(meta_sys, 'task_datasets')

    def test_data_loading(self):
        """Test category data loading"""
        meta_sys = MetaLearningSystem()

        # Create mock data
        dates = pd.date_range('2020-01-01', periods=50, freq='W')
        values = np.random.randn(50).cumsum() + 100
        df = pd.DataFrame({'C1': values}, index=dates)
        df.to_csv('C1.csv')

        try:
            data, scaler = meta_sys.load_category_data('C1')
            assert len(data) == 50
            assert hasattr(scaler, 'transform')
            assert 'C1' in meta_sys.task_datasets
        finally:
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')

    def test_transfer_learning(self):
        """Test transfer learning functionality"""
        meta_sys = MetaLearningSystem()

        # Create mock data for two categories
        for cat in ['C1', 'C2']:
            dates = pd.date_range('2020-01-01', periods=50, freq='W')
            values = np.random.randn(50).cumsum() + 100
            df = pd.DataFrame({cat: values}, index=dates)
            df.to_csv(f'{cat}.csv')

        try:
            model, scaler = meta_sys.transfer_learning('C1', 'C2', fine_tune_steps=3)
            assert model is not None
            assert scaler is not None
        finally:
            for cat in ['C1', 'C2']:
                if os.path.exists(f'{cat}.csv'):
                    os.remove(f'{cat}.csv')


class TestEnsembleMethods:
    """Test ensemble methods"""

    def test_ensemble_initialization(self):
        """Test EnsembleMethods initialization"""
        ensemble = EnsembleMethods()
        assert hasattr(ensemble, 'models')
        assert len(ensemble.models) > 0
        assert 'sarimax' in ensemble.models

    def test_weighted_average(self):
        """Test weighted average ensemble"""
        ensemble = EnsembleMethods()

        # Mock predictions
        predictions = {
            'model1': np.array([1.0, 2.0, 3.0]),
            'model2': np.array([1.5, 2.5, 3.5]),
        }

        weights = {'model1': 0.6, 'model2': 0.4}
        result = ensemble.weighted_average_ensemble(predictions, weights)

        expected = np.array([1.2, 2.2, 3.2])  # Weighted average
        np.testing.assert_array_almost_equal(result, expected)


class TestAPIEndpoints:
    """Test API endpoints"""

    def test_meta_learning_status(self):
        """Test meta-learning status endpoint"""
        from app import app

        with app.test_client() as client:
            response = client.get('/api/meta-learning/status')
            assert response.status_code == 200

            data = response.get_json()
            assert 'initialized' in data
            assert 'maml_trained' in data
            assert 'available_categories' in data

    def test_forecast_endpoint(self):
        """Test forecast endpoint"""
        from app import app

        # Create mock data
        dates = pd.date_range('2020-01-01', periods=50, freq='W')
        values = np.random.randn(50).cumsum() + 100
        df = pd.DataFrame({'C1': values}, index=dates)
        df.to_csv('C1.csv')

        try:
            with app.test_client() as client:
                response = client.post('/forecast', data={
                    'category': 'C1',
                    'date': '2020-06-01',
                    'model_type': 'ensemble'
                }, follow_redirects=True)

                assert response.status_code == 200
                assert b'Forecast Results' in response.data
        finally:
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')


# Performance and Integration Tests
class TestPerformance:
    """Performance and scalability tests"""

    def test_forecast_performance(self):
        """Test forecast performance under load"""
        import time

        # Create larger dataset
        dates = pd.date_range('2020-01-01', periods=500, freq='W')
        values = np.random.randn(500).cumsum() + 100
        df = pd.DataFrame({'C1': values}, index=dates)
        df.to_csv('C1.csv')

        try:
            start_time = time.time()
            forecast_sales('C1', '2025-01-01', 'ensemble')
            end_time = time.time()

            duration = end_time - start_time
            assert duration < 30  # Should complete within 30 seconds
        finally:
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')

    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform forecast operation
        dates = pd.date_range('2020-01-01', periods=100, freq='W')
        values = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({'C1': values}, index=dates)
        df.to_csv('C1.csv')

        try:
            forecast_sales('C1', '2025-01-01', 'ensemble')
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_increase = final_memory - initial_memory
            assert memory_increase < 500  # Should not increase memory by more than 500MB
        finally:
            if os.path.exists('C1.csv'):
                os.remove('C1.csv')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])