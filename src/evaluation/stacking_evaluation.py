"""Evaluate stacking on an untouched final weekly block, then optionally refit."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.stacking_model import (
    BASE_NAMES, ROOT, StackingConfig, StackingForecaster,
    load_weekly_series, save_forecaster, series_fingerprint,
)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    result = {"MAE": float(mean_absolute_error(actual, predicted)),
              "RMSE": float(np.sqrt(mean_squared_error(actual, predicted)))}
    # A zero actual has no defined percentage error. Never hide it with epsilon.
    result["MAPE"] = float(np.mean(np.abs((actual-predicted)/actual))*100) if np.all(actual != 0) else None
    result["zero_actual_count"] = int(np.sum(actual == 0))
    return result


def evaluate_stacking(categories=None, config=None, data_dir=None, output_path=None, train_production=False, model_dir=None):
    import sklearn
    import torch
    import xgboost

    config = config or StackingConfig()
    categories = list(categories or [f"C{i}" for i in range(1, 9)])
    output_path = Path(output_path or ROOT / "src/evaluation_results/stacking_results.json")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "Final 10-week holdout by default; expanding blocked out-of-fold Ridge training inside the pre-test history",
              "config": asdict(config), "base_models": list(BASE_NAMES),
              "meta_model": "StandardScaler + Ridge; alpha="+str(config.ridge_alpha),
              "nonnegative_output": True,
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "xgboost": xgboost.__version__, "sklearn": sklearn.__version__},
              "categories": {}}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for category in categories:
            print(f"Evaluating {category}: chronological stacking", flush=True)
            started = time.perf_counter()
            try:
                dates, values = load_weekly_series(category, data_dir)
                split = len(values)-config.horizon
                train, actual = values[:split], values[split:]
                model = StackingForecaster(config).fit(train)
                fit_seconds = time.perf_counter()-started
                prediction_started = time.perf_counter()
                components = model.predict_components(config.horizon)
                predicted = np.maximum(model.meta_model_.predict(components), 0)
                prediction_seconds = time.perf_counter()-prediction_started
                forecasts = {"stacking": predicted, "equal_average": np.maximum(components.mean(axis=1), 0),
                             "naive_last": np.full(config.horizon, train[-1])}
                forecasts.update({name: np.maximum(components[:, i], 0) for i, name in enumerate(BASE_NAMES)})
                folds = []
                for fold in model.fold_ranges_:
                    folds.append({**fold, "train_end": str(dates[fold["train_stop"]-1].date()),
                                  "validation_start_date": str(dates[fold["validation_start"]].date()),
                                  "validation_end_date": str(dates[fold["validation_stop"]-1].date())})
                entry = {"status": "ok", "train_rows": len(train), "test_rows": len(actual),
                         "train_end": str(dates[split-1].date()), "test_dates": [str(d.date()) for d in dates[split:]],
                         "dataset_sha256": series_fingerprint(dates, values), "folds": folds,
                         "actual": actual.tolist(), "predictions": {k: v.tolist() for k,v in forecasts.items()},
                         "metrics": {k: metrics(actual, v) for k,v in forecasts.items()},
                         "fit_seconds": fit_seconds, "predict_seconds": prediction_seconds}
                # Preserve the holdout-only bundle separately from deployment refits.
                evaluation_dir = ROOT / "artifacts/models/models_stacking_evaluation" if model_dir is None else Path(model_dir)/"evaluation"
                entry["evaluation_artifact"] = str(save_forecaster(model, category, dates[:split], train, evaluation_dir))
                report["categories"][category] = entry
                if train_production:
                    try:
                        production = StackingForecaster(config).fit(values)
                        entry["production_artifact"] = str(save_forecaster(production, category, dates, values, model_dir))
                    except Exception as exc:
                        entry["production_error"] = f"{type(exc).__name__}: {exc}"
                print(category, entry["metrics"]["stacking"], flush=True)
            except Exception as exc:
                report["categories"][category] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"{category} FAILED: {exc}", flush=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        torch.set_num_threads(old_threads)
    successful = [v for v in report["categories"].values() if v["status"] == "ok"]
    report["successful_categories"] = len(successful)
    report["summary"] = {}
    for method in ("stacking", "equal_average", "naive_last", *BASE_NAMES):
        if successful:
            report["summary"][method] = {m: float(np.mean([v["metrics"][method][m] for v in successful])) for m in ("MAE", "RMSE")}
    report["limitations"] = ["Single final holdout; no statistical superiority claim.",
                             "Hybrid real/synthetic dataset; regional generalisation not measured.",
                             "Not comparable to legacy ensemble_results.json because the evaluation protocol and base-model set differ.",
                             "MAPE is null when the holdout contains zero actual sales; MAE and RMSE remain defined."]
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=[f"C{i}" for i in range(1, 9)])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-production", action="store_true")
    args = parser.parse_args()
    result = evaluate_stacking(args.categories, StackingConfig(epochs=args.epochs, horizon=args.horizon), train_production=args.train_production)
    if result["successful_categories"] != len(args.categories) or any("production_error" in v for v in result["categories"].values()):
        raise SystemExit(1)
