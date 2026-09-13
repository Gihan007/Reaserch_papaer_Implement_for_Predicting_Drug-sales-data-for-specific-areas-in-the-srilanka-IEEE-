"""Chronological stacking of XGBoost, LSTM and GRU sales forecasts.

The Ridge combiner sees only predictions from models fitted before each
validation block. The final base models are refitted on the supplied history;
the caller must exclude its final test period from that history.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "models" / "models_stacking"
BASE_NAMES = ("xgboost", "lstm", "gru")


@dataclass(frozen=True)
class StackingConfig:
    n_lags: int = 5
    seq_length: int = 10
    horizon: int = 10
    n_folds: int = 3
    epochs: int = 30
    seed: int = 42
    ridge_alpha: float = 1.0

    def __post_init__(self):
        for name in ("n_lags", "seq_length", "horizon", "n_folds", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_folds < 2 or self.ridge_alpha <= 0:
            raise ValueError("Use at least two folds and positive Ridge regularisation")


def checked_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales must be a finite, nonnegative one-dimensional series")
    return values


def load_weekly_series(category, data_dir=None):
    if category not in {f"C{i}" for i in range(1, 9)}:
        raise ValueError("Category must be C1 through C8")
    path = Path(data_dir or ROOT / "data" / "raw") / f"{category}.csv"
    frame = pd.read_csv(path, parse_dates=["datum"])
    dates = pd.DatetimeIndex(frame["datum"])
    if len(dates) < 2 or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique, valid and chronologically ordered")
    if not (dates.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("Stacking requires consistently spaced weekly observations")
    values = checked_values(frame[category].to_numpy())
    return dates, values


def series_fingerprint(dates, values):
    digest = hashlib.sha256()
    digest.update(np.asarray(pd.DatetimeIndex(dates).asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


class BaseForecasters:
    """Fit the project's sequence models in memory, never from full-data artifacts."""

    def __init__(self, config):
        self.config = config

    def fit(self, values):
        import torch
        from xgboost import XGBRegressor
        from src.models.lstm_model import train_lstm_model
        from src.models.gru_model import train_gru_model

        c = self.config
        values = checked_values(values)
        # Latest lag first, used identically in fitting and recursive prediction.
        X = np.array([values[i-c.n_lags:i][::-1] for i in range(c.n_lags, len(values))])
        self.xgboost = XGBRegressor(
            objective="reg:squarederror", n_estimators=100, max_depth=3,
            learning_rate=0.05, n_jobs=1, random_state=c.seed,
        ).fit(X, values[c.n_lags:])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.lstm, self.lstm_scaler = train_lstm_model(values, seq_length=c.seq_length, epochs=c.epochs)
            torch.manual_seed(c.seed)
            self.gru, self.gru_scaler = train_gru_model(values, seq_length=c.seq_length, epochs=c.epochs)
        return self

    def predict(self, history, n_steps):
        from src.models.lstm_model import predict_lstm
        from src.models.gru_model import predict_gru

        history = checked_values(history)
        c = self.config
        window = list(history)
        xgb_predictions = []
        for _ in range(n_steps):
            pred = float(self.xgboost.predict(np.asarray(window[-c.n_lags:][::-1]).reshape(1, -1))[0])
            xgb_predictions.append(pred)
            window.append(pred)
        columns = [xgb_predictions]
        for model, scaler, predict in (
            (self.lstm, self.lstm_scaler, predict_lstm),
            (self.gru, self.gru_scaler, predict_gru),
        ):
            scaled = scaler.transform(history.reshape(-1, 1)).ravel()
            forecast = predict(model, scaled, seq_length=c.seq_length, n_steps=n_steps)
            columns.append(scaler.inverse_transform(np.asarray(forecast).reshape(-1, 1)).ravel())
        matrix = np.column_stack(columns)
        if matrix.shape != (n_steps, len(BASE_NAMES)) or not np.isfinite(matrix).all():
            raise ValueError("A stacking base model returned invalid forecasts")
        return matrix


class StackingForecaster:
    def __init__(self, config=None):
        self.config = config or StackingConfig()

    def fit(self, history, base_factory=BaseForecasters):
        history = checked_values(history)
        c = self.config
        first_origin = len(history) - c.n_folds * c.horizon
        if first_origin < max(c.n_lags, c.seq_length) + 20:
            raise ValueError("Insufficient history for chronological stacking folds")
        features, targets = [], []
        self.fold_ranges_ = []
        for origin in range(first_origin, len(history), c.horizon):
            base = base_factory(c).fit(history[:origin])
            features.append(base.predict(history[:origin], c.horizon))
            targets.append(history[origin:origin+c.horizon])
            self.fold_ranges_.append({"train_stop": origin, "validation_start": origin, "validation_stop": origin+c.horizon})
        self.oof_predictions_ = np.vstack(features)
        self.oof_targets_ = np.concatenate(targets)
        if self.oof_predictions_.shape != (len(self.oof_targets_), len(BASE_NAMES)) or not np.isfinite(self.oof_predictions_).all():
            raise ValueError("Invalid out-of-fold predictions; stacking training stopped")
        self.meta_model_ = make_pipeline(StandardScaler(), Ridge(alpha=c.ridge_alpha))
        self.meta_model_.fit(self.oof_predictions_, self.oof_targets_)
        self.base_models_ = base_factory(c).fit(history)
        self.history_ = history.copy()
        return self

    def predict_components(self, n_steps):
        if not hasattr(self, "base_models_"):
            raise ValueError("Stacking model has not been fitted")
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or not 1 <= n_steps <= self.config.horizon:
            raise ValueError(f"Stacking supports 1–{self.config.horizon} future weekly steps; retrain with a larger horizon for longer forecasts")
        return self.base_models_.predict(self.history_, int(n_steps))

    def predict(self, n_steps):
        # A separate feature row for every horizon; never broadcast one prediction.
        values = self.meta_model_.predict(self.predict_components(n_steps))
        if not np.isfinite(values).all():
            raise ValueError("Stacking produced non-finite predictions")
        return np.maximum(values, 0.0)


def save_forecaster(model, category, dates, values, model_dir=None):
    directory = Path(model_dir or MODEL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{category}_stacking.joblib"
    bundle = {"version": 1, "category": category, "config": asdict(model.config),
              "trained_through": str(pd.Timestamp(dates[-1]).date()),
              "fingerprint": series_fingerprint(dates, values), "model": model}
    fd, temporary = tempfile.mkstemp(prefix=category+"_", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def forecast_stacking(category, n_steps, data_dir=None, model_dir=None):
    dates, values = load_weekly_series(category, data_dir)
    path = Path(model_dir or MODEL_DIR) / f"{category}_stacking.joblib"
    if not path.exists():
        raise FileNotFoundError("Stacking artifact missing. Run python -m src.evaluation.stacking_evaluation --train-production")
    bundle = joblib.load(path)
    if bundle.get("version") != 1 or bundle.get("category") != category or bundle.get("fingerprint") != series_fingerprint(dates, values):
        raise ValueError("Stacking artifact is stale or incompatible; retrain it for the current dataset")
    return bundle["model"].predict(n_steps)
