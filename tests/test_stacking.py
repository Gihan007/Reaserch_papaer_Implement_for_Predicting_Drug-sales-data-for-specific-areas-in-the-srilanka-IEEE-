from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.stacking_model import (
    BASE_NAMES, StackingConfig, StackingForecaster, forecast_stacking,
    load_weekly_series, save_forecaster,
)


class TracedBase:
    histories = []

    def __init__(self, config):
        self.config = config

    def fit(self, history):
        self.fitted = np.array(history, copy=True)
        self.histories.append(self.fitted)
        return self

    def predict(self, history, steps):
        # Check the prediction never receives the upcoming validation targets.
        np.testing.assert_array_equal(history, self.fitted)
        trend = history[-1] + np.arange(1, steps+1)
        return np.column_stack([trend, trend*0.95, trend*1.05])


def test_meta_training_preserves_time_and_aligns_each_horizon():
    values = np.arange(1., 101.)
    TracedBase.histories = []
    model = StackingForecaster(StackingConfig(horizon=5, n_folds=3)).fit(values, TracedBase)
    assert [len(v) for v in TracedBase.histories] == [85, 90, 95, 100]
    np.testing.assert_array_equal(model.oof_targets_, values[85:])
    np.testing.assert_array_equal(model.oof_predictions_[:, 0], values[85:])
    assert all(f['train_stop'] == f['validation_start'] for f in model.fold_ranges_)
    result = model.predict(5)
    assert result.shape == (5,)
    assert np.all(np.diff(result) > 0)  # No repetition of one forecast over the horizon.


def test_final_holdout_never_enters_model_training(tmp_path, monkeypatch):
    from src.evaluation import stacking_evaluation as evaluation
    original_fit = StackingForecaster.fit
    monkeypatch.setattr(StackingForecaster, 'fit', lambda self, values: original_fit(self, values, TracedBase))
    dates = pd.date_range('2020-01-05', periods=105, freq='7D')
    values = np.r_[np.arange(1., 101.), np.full(5, 10000.)]
    pd.DataFrame({'datum': dates, 'C1': values}).to_csv(tmp_path/'C1.csv', index=False)
    TracedBase.histories = []
    result = evaluation.evaluate_stacking(['C1'], StackingConfig(horizon=5), data_dir=tmp_path,
                                         output_path=tmp_path/'results.json', model_dir=tmp_path/'models')
    entry = result['categories']['C1']
    assert entry['status'] == 'ok'
    assert entry['train_rows'] == 100 and entry['test_rows'] == 5
    assert max(v.max() for v in TracedBase.histories) == 100
    assert entry['actual'] == [10000.] * 5
    assert all(v < 200 for v in entry['predictions']['stacking'])
    assert entry['metrics']['stacking']['MAE'] > 9000


def test_artifact_round_trip_and_stale_data_rejection(tmp_path):
    dates = pd.date_range('2020-01-05', periods=100, freq='7D')
    values = np.arange(1., 101.)
    frame = pd.DataFrame({'datum': dates, 'C1': values})
    frame.to_csv(tmp_path/'C1.csv', index=False)
    model = StackingForecaster().fit(values, TracedBase)
    save_forecaster(model, 'C1', dates, values, tmp_path)
    np.testing.assert_allclose(forecast_stacking('C1', 5, tmp_path, tmp_path), model.predict(5))
    frame.loc[99, 'C1'] += 1
    frame.to_csv(tmp_path/'C1.csv', index=False)
    with pytest.raises(ValueError, match='stale'):
        forecast_stacking('C1', 5, tmp_path, tmp_path)


def test_missing_artifact_and_untrained_combiner_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match='artifact missing'):
        forecast_stacking('C1', 3, model_dir=tmp_path)
    from src.evaluation.ensemble_methods import EnsembleMethods
    with pytest.raises(ValueError, match='fitted meta-model'):
        EnsembleMethods().stacking_ensemble({}, None)


def test_legacy_stacking_entry_combines_every_step():
    from src.evaluation.ensemble_methods import EnsembleMethods
    model = StackingForecaster().fit(np.arange(1., 101.), TracedBase)
    components = model.predict_components(5)
    predicted = EnsembleMethods().stacking_ensemble(dict(zip(BASE_NAMES, components.T)), model.meta_model_)
    np.testing.assert_allclose(predicted, model.predict(5))
    with pytest.raises(ValueError, match='aligned'):
        EnsembleMethods().stacking_ensemble({'xgboost': [1], 'lstm': [2, 3], 'gru': [4]}, model.meta_model_)


@pytest.mark.parametrize('days,expected', [(1, 1), (7, 1), (8, 2), (70, 10)])
def test_api_stacking_uses_weekly_steps(monkeypatch, days, expected):
    import src.models.stacking_model as stacking
    from services.forecast_service.app.forecasting import get_model_forecast
    def predict(category, n_steps, **kwargs):
        assert n_steps == expected
        return np.arange(n_steps, dtype=float)
    monkeypatch.setattr(stacking, 'forecast_stacking', predict)
    value, label = get_model_forecast('C1', days, 'stacking')
    assert value == expected-1
    assert label.startswith('Stacking')


def test_service_does_not_return_fake_success_on_stacking_failure(monkeypatch):
    from fastapi.testclient import TestClient
    from services.forecast_service.app.main import app
    import src.models.stacking_model as stacking
    def fail(*args, **kwargs):
        raise ValueError('stacking artifact unavailable')
    monkeypatch.setattr(stacking, 'forecast_stacking', fail)
    response = TestClient(app).post('/forecast', json={'category': 'C1', 'date': '2023-12-10', 'model_type': 'stacking'})
    assert response.status_code == 500
    assert 'artifact unavailable' in response.json()['detail']


def test_irregular_dates_and_invalid_history_rejected(tmp_path):
    pd.DataFrame({'datum': ['2020-01-05', '2020-01-13'], 'C1': [1, 2]}).to_csv(tmp_path/'C1.csv', index=False)
    with pytest.raises(ValueError, match='weekly'):
        load_weekly_series('C1', tmp_path)
    with pytest.raises(ValueError, match='finite'):
        StackingForecaster().fit([1, np.nan, 2])
    with pytest.raises(ValueError, match='Insufficient'):
        StackingForecaster().fit(np.arange(20.))
    model = StackingForecaster().fit(np.arange(100.), TracedBase)
    with pytest.raises(ValueError, match='weekly steps'):
        model.predict(11)


def test_zero_actual_percentage_error_is_explicit():
    from src.evaluation.stacking_evaluation import metrics
    result = metrics([0., 2.], [1., 2.])
    assert result['MAE'] == .5 and result['MAPE'] is None
    assert result['zero_actual_count'] == 1
