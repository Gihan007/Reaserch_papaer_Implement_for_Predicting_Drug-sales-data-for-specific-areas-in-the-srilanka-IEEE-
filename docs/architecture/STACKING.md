# Stacking forecasts

The stacking implementation combines **XGBoost, LSTM and GRU** with a **StandardScaler + Ridge regression** meta-model. It is a new three-model experiment, separate from the archived seven-model ensemble evaluation.

## Train and evaluate

From the repository root:

This run was trained and tested with the system Python at `C:/Users/user/AppData/Local/Programs/Python/Python39/python.exe` (package versions are saved in the result JSON). A separate check of the existing project `.venv` failed while importing scikit-learn with a Windows long-path DLL error, before stacking could run. Use the tested system interpreter for this workspace; the older virtual environment needs separate repair. In PowerShell, an explicit invocation is:

```powershell
& 'C:/Users/user/AppData/Local/Programs/Python/Python39/python.exe' -m src.evaluation.stacking_evaluation --train-production
```

```powershell
python -m src.evaluation.stacking_evaluation --train-production
```

For a single category:

```powershell
python -m src.evaluation.stacking_evaluation --categories C1 --train-production
```

The command writes `src/evaluation_results/stacking_results.json`. A single-category run replaces that report with the requested category's run; use the default command to produce the complete eight-category report. Original `ensemble_results.json` and `performance_summary.json` are preserved because their evaluation protocol differs.

Defaults: 5 XGBoost lags; 10-step recurrent inputs; 30 recurrent training epochs; seed 42; 3 chronological meta-training blocks of 10 weeks; Ridge alpha 1. XGBoost uses 100 trees, depth 3 and learning rate 0.05. Existing LSTM/GRU architectures use 2 layers, hidden size 64 and dropout 0.1. Numeric-library versions and configuration are stored in the report. The command limits PyTorch to one CPU thread during training and restores the previous thread count afterwards.

With the bundled 517-row files:

1. Reserve the final **10 weekly rows** as an untouched test block. No base-model fit, scaler fit or combiner fit sees those targets.
2. Use the first **507 rows** for training. Fit fresh base models on prefixes ending at row counts **477**, **487** and **497**. Predict the following 10 weeks recursively at each origin, without feeding those future actuals into prediction.
3. Combine the 30 resulting out-of-fold prediction rows. Every row contains three predictions for the same future week; train Ridge against that week's actual sales.
4. Refit base models on all 507 pre-test rows. Predict the 10 held-out weeks and use Ridge to combine each horizon separately. Negative final sales forecasts are clipped to zero.
5. Score stacking, its three individual bases, equal averaging and a last-value naive baseline on exactly those same test dates. This provides a fair comparison within this experiment, not against old archived metrics.
6. With `--train-production`, fit a **separate** full-history model for the application after evaluation. Its training includes all 517 available rows; it is never used to compute the already-reported holdout scores.

The evaluation bundle is stored under `artifacts/models/models_stacking_evaluation/`; the full-history application bundle is stored under `artifacts/models/models_stacking/`. Each contains model settings, training end date and a SHA-256 fingerprint of its date/value history. Model saves use atomic replacement. The API rejects missing, incompatible or stale bundles, rather than silently returning another algorithm's output.

## Forecast

Choose **Stacking (XGBoost + LSTM + GRU)** in the forecasting page, or use either the gateway `/api/forecast` or forecast service `/forecast`:

```json
{"category": "C1", "date": "2023-12-10", "model_type": "stacking"}
```

The bundled history ends on **2023-12-03**. The default model supports the next **1–10 weekly steps**, through **2024-02-11**. A date between weekly endpoints selects the following weekly step: 1–7 days ahead means step 1, 8–14 means step 2. Historical requests retain the application's historical-lookup behaviour. A current calendar date in 2026 is outside this old dataset's trained horizon; add current weekly data and retrain, or explicitly train a larger horizon if enough data exists. Longer horizons need their own evaluation.

This weekly conversion is implemented for stacking. The older model routes have not been rewritten by this change.

For Python callers:

```python
from src.models.stacking_model import forecast_stacking
ten_week_path = forecast_stacking("C1", n_steps=10)
```

The old `EnsembleMethods.create_ensemble_predictions(category, method="stacking")` entry point is also repaired: it trains the chronological stack from the category history. This is slower than the artifact-backed API path. Its `stacking_ensemble` helper requires aligned XGBoost/LSTM/GRU arrays and a trained three-feature combiner; it no longer silently falls back to averaging or repeats a single combined value across all steps.

## Failure reporting and limitations

Evaluation failures include their exception type/message and `status: error`; they are not written as unexplained null metrics. MAPE alone remains undefined if a test period contains zero actual sales; in that case the report records `MAPE: null` and `zero_actual_count`, while MAE and RMSE remain valid. Summary values only include successful categories, and the successful count is explicit. The CLI exits unsuccessfully if any requested evaluation or production refit fails.

The current dataset is student-confirmed as approximately 50:50 real/synthetic and has no row-level source or regional labels. This is one final holdout, not repeated independent field validation. No claim is made that stacking must outperform its base models, that it improves real pharmacy stock outcomes, or that the original thesis benchmark has been retrospectively corrected.

## Tests

```powershell
python -m pytest tests/test_stacking.py tests/test_services_smoke.py -q
```

The tests check chronological fold boundaries, untouched holdout data, horizon-specific predictions, artifact round trips and stale-data rejection, weekly API steps, explicit errors, irregular-data rejection and zero-actual metric handling, plus existing service smoke checks.

The stacking UI also has a Node test: `node --test tests/test_stacking_ui.cjs`. Its result path uses actual CSV history and saved holdout metrics, bypassing the older page's simulated chart/metric routines and hiding unsupported intervals and explanation controls for stacking.
