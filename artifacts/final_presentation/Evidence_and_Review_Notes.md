# Presentation evidence and review notes

Prepared 13 September 2026 for Bope Ranasinghage Gihan Lakmal, CS/2020/015.

## Update: stacking implemented and evaluated

Following the student's request, the stacking path was implemented and the presentation updated. The original archive still contains null stacking entries; **the new measured experiment is `src/evaluation_results/stacking_results.json`**. These records use different protocols and must not be combined into one model ranking.

The new stack uses XGBoost, LSTM and GRU, combined by StandardScaler + Ridge. On each 517-week category series, 507 weeks are pre-test history and the last ten weeks (**1 October–3 December 2023**) are untouched test data. Three ten-week chronological validation blocks within the pre-test history supply Ridge's training predictions. Base models and their scalers are fitted only on earlier observations at each origin. A separate full-history application model is trained after scoring. See `docs/architecture/STACKING.md` for exact settings and commands.

All eight evaluations and all eight full-history artifact saves succeeded. Arithmetic mean category scores in the new experiment:

| Method | MAE | RMSE |
|---|---:|---:|
| Stacking / Ridge | 30.2059 | 36.0614 |
| Equal average of the three bases | 24.2382 | 29.4752 |
| XGBoost | 24.7716 | 30.4166 |
| LSTM | 26.4038 | 32.9673 |
| GRU | 25.7002 | 30.6456 |
| Last-value naive baseline | 35.0069 | 40.7899 |

Stacking improves on the naive baseline but does **not** beat simple averaging or the three bases on mean MAE in this run. No tuning was performed against the held-out results to manufacture an improvement. This is one holdout on hybrid data, not proof of general superiority or regional field performance.

Validation: **16 Python tests plus one JavaScript UI test passed**. All eight stored full-history bundles were loaded and produced finite, nonnegative ten-week forecast paths. An actual gateway request for C1 on 11 December 2023 returned **53.00357925194383** for the aligned weekly endpoint **17 December 2023**, with HTTP 200. An out-of-horizon 2026 request returned an explicit error. Evidence is in `stacking_integration_checks.json`.

The stacking API now converts day offsets to weekly steps. Missing/stale artifacts and unsupported horizons produce explicit failures. The stacking UI uses actual CSV history and the new held-out metrics; its placeholder accuracy, simulated charts, unsupported intervals and unsupported explanation path are bypassed. The older forecasting paths retain their earlier limitations. The embedded video remains an explicitly labelled offline walkthrough of saved project screens; it has not been presented as a recording of the new stacking run.

Slides 10, 12–14 and 18–20 and the speaker notes now distinguish this new experiment from the original thesis benchmark. The remaining sections below document the initial evidence audit and the original archived results.

## Files and submission

- `CS_2020_015.pptx`: editable presentation, 22 visible slides plus 5 hidden Q&A slides. Saved under the required student-number filename.
- `CS_2020_015.pdf`: static preview of the 22 visible slides; the video does not play in the PDF.
- `Speaker_Notes_CS_2020_015.docx`: timed speaking script, evidence references and suggested answers to panel questions. The script is also in PowerPoint's Notes pane.
- `Speaker_Notes_CS_2020_015_Updated.docx`: latest stacking revision. The original notes were locked by an open application, so the updated script was saved separately. Use this revised file; the PowerPoint's embedded notes are also updated.
- `project_walkthrough.mp4`: 45-second, silent H.264 offline walkthrough. A copy is embedded inside the PPTX, so the external file is not required for playback.
- `build_presentation.py`: reproducible generation source; reads the project artifacts and creates the presentation files.

Planned main presentation: **18:35**, including the walkthrough. Timings are a rehearsal plan, not a guarantee of delivery duration. Slide transitions remain manual. The user-provided deadline is **18 September 2026**; evaluation is a 20-minute presentation and 10-minute questioning session. Obtain and incorporate supervisor feedback as the provided guidelines recommend. No upload, submission or supervisor approval has been performed or claimed.

The example `C:/Users/user/Downloads/CS_2019_045.pptx` was used for its academic narrative structure and blue visual direction. Its student's identity, research claims, emotion-recognition metrics, acknowledgements and publication claims were not copied. The deck's own text was treated as example content, not as instructions.

## Evidence hierarchy

1. Current category CSVs and saved evaluation JSONs establish the numeric values shown.
2. Current source files establish what the implementation does, subject to differences between historical experiments and current code.
3. `CSCI 43018- final structured thesis CS_2020_015 - Final_1.pdf` supplies project identity, background, reported experiments and the main written project account. The other final-review PDF was inspected for context; supervisor approval status is not inferred from the filename.
4. The student's 13 September 2026 reply confirms the intended real/synthetic mix as 50:50. This is attributed as an approximate student-confirmed proportion, because the CSVs do not contain row-level provenance.

The initial presentation was generated without retraining or changing application code. The subsequent stacking implementation, new benchmark and fresh tests are described in the update above. Original thesis documents and original evaluation JSON files remain preserved.

## Numeric verification

All eight category files have **517 rows**, two columns (`datum`, category), and dates from **2014-01-12 to 2023-12-03**. Every date gap is seven days. No missing cells or duplicate dates were found. There are **4,136 category-week sales values**, representing **517 shared dates** rather than 4,136 independent weeks. The full C6 series has **46 zero values**; this is not a count for the benchmark test window.

The values of C1–C8 were compared with the combined CSV's columns and match the following mapping: C1=M01AB, C2=M01AE, C3=N02BA, C4=N02BE, C5=N05B, C6=N05C, C7=R03, C8=R06. These are category codes, not regions.

Overall figures below are arithmetic means of the eight category metrics. The means were recomputed from `src/evaluation_results/model_metrics.json` and checked against `performance_summary.json` with four-decimal rounding tolerance. Mean category RMSE is not the same operation as calculating a single RMSE after pooling all errors.

| Model | MAE | RMSE | MAPE (%) | Recorded mean time (s) |
|---|---:|---:|---:|---:|
| LightGBM | 23.3728 | 28.9163 | 73.9233 | 0.1074 |
| LSTM | 24.0362 | 29.5830 | 62.6915 | 0.1260 |
| GRU | 24.1104 | 29.7559 | 63.9476 | 0.1067 |
| XGBoost | 25.4679 | 29.9704 | 86.4031 | 0.0343 |
| Transformer | 28.8908 | 36.5221 | 53.6691 | 0.2426 |
| SARIMAX | 30.4946 | 38.4108 | 56.0618 | 0.1498 |
| Prophet | 32.5682 | 37.6763 | 88.0285 | 0.8725 |

Lowest recorded MAE per category:

| Category | Model | MAE |
|---|---|---:|
| C1 | GRU | 9.8758 |
| C2 | Prophet | 13.1088 |
| C3 | Prophet | 10.2570 |
| C4 | LightGBM | 72.6065 |
| C5 | LSTM | 24.8986 |
| C6 | LSTM | 3.1634 |
| C7 | XGBoost | 26.0462 |
| C8 | LSTM | 13.8154 |

From the original `ensemble_results.json`, arithmetic mean MAE is **23.8313** for `weighted_average` (equal weights in the default offline method) and **23.8447** for `performance_weighted` (inverse-MAE weighting). Neither is lower than LightGBM's recorded mean MAE. That archived file's stacking fields remain null; the new stacking experiment has its own separate results, listed above.

## Material discrepancies and how the slides handle them

| Issue found | Evidence | Presentation treatment |
|---|---|---|
| README advertises much smaller errors, such as ensemble MAE 2.34, which differ from the actual saved experiment. | README benchmark table versus evaluation JSONs. | Excludes those README numbers. Uses saved evaluation values and verified category means. |
| Thesis describes chronological testing, but the available evaluator does not establish a common unseen-future test. | `src/evaluation/model_evaluation.py`: targets are `values[-10:]`; SARIMAX fits on all data; several forecast helpers read all data and predict past its end. | Calls the numbers recorded / archived experimental results. Main slides 12 and 19 disclose the limitation; backup slide 25 explains it. No split percentage, leakage-free test or statistical significance is invented. |
| Weekly data is passed to a date interface that calculates a day count. | `services/forecast_service/app/forecasting.py`, `forecast_sales` and model dispatch: `days_ahead` is passed as `n_steps` / `periods`. | Main slide 19 identifies the need to map requested dates to weekly steps. No claim of verified calendar-aligned future forecasting is made. |
| Saved results may predate current implementation changes. | LightGBM currently retrains locally on a forecast call for artifact compatibility; separate optimisation files contain other configurations. | Lists current defaults as defaults, not as the proven hyperparameters behind all saved metrics. Does not claim archived latency describes the current code. |
| Data-source mix is not encoded in the CSV files. | Thesis §3.2 and the student's reply; only date/value columns in category files. | Reports approximately 50:50 real/synthetic, student-confirmed, and does not invent exact real/synthetic counts or a synthetic-generation algorithm. |
| Regional forecasting is a use-case context, not a demonstrated regional benchmark. | No region/pharmacy columns in category CSVs; gateway branch list contains example locations. | Describes category-wise forecasting for a Sri Lankan planning context. Does not present example branches as data-collection sites or claim generalisation across Sri Lanka. |
| The online ensemble path differs from the offline weighting experiment. | `get_ensemble_forecast` filters available nonzero outputs and takes a mean; offline class supports inverse-MAE weights. | Main slide 10 distinguishes the two. Does not say the online label proves inverse-MAE weighting. |
| The selected explanation figure is fallback importance. | `services/explainability_service/app/shap_explainer.py`, `get_fallback_explainability` and `_fallback_feature_importance`; saved PNG. | Explicitly calls it fallback lag importance. Model importance or absolute correlations can supply the ranking. It is not called a local SHAP explanation or causal effect. |
| Several advanced architectures have implementation files but no values in the main benchmark. | Model files versus the seven keys in the evaluation JSON. | TFT, N-BEATS and Informer are identified as additional implementations without comparable saved results. N-BEATS is not called a Transformer. |
| Experimental advanced-AI outputs do not prove field performance. | NAS JSON, federated JSON, causal discovery JSON and thesis test account. | NAS: 8 architectures; scaled errors are not compared with raw-sales MAE. Federated: 2 configured simulated clients, 1 round; the recorded round actually lists 1 participating client. No real multi-pharmacy privacy trial is claimed. Recorded MAML status was untrained. Causal outputs are described as associations. |
| Smoke tests mainly check health, rendering and historical retrieval. | `tests/test_services_smoke.py` and thesis §4.8. | The four passes are labelled recorded checks, not freshly rerun tests or a future-prediction accuracy test. |
| Existing MP4 in `media/videos` shows an earlier code/demo environment. | Frames inspected from the March 2025-named WhatsApp file. | It is not presented as the current final system. A new silent offline walkthrough is assembled from identified current-project screenshots and archived outputs. |

## Demonstration provenance

The embedded video is a sequence of saved screens, not a captured live session. Each screen is displayed for 15 seconds; the file has no audio stream.

1. Interface: `docs/thesisi version/chapter4_inserted_assets/figure_4_16_forecast_interface.png`. The video crops unused blank space below the actual interface to improve readability. It does not fabricate a new form submission.
2. Historical chart: `services/frontend_service/app/static/images/C1_2018_01_15_xgboost_forecast.png`. Requested date: 15 January 2018. Nearest recorded week: 14 January 2018. C1 sales: 28.33. Historical retrieval is explicitly distinguished from unseen forecasting.
3. Explanation: `services/frontend_service/app/static/images/shap/fallback_importance_C1_xgboost_20260712_042311.png`. Highest feature: `sales_lag_4`, approximately 0.2664, consistent with the thesis's recorded explanation test.

PowerPoint slide 27 is a hidden static fallback for the video. The explanation figure also appears on visible slide 15. The video is click-to-play; no slide is set to advance automatically.

## Literature verification

- Ekanayake, Nasmeen, Lakmal, Vimanshani and Perera, *Predicting medical drug sales in a specific area for categorical drugs using time series forecasting*: author names and SARIMA/Flask context read from the local paper in `docs/paper/`. SCSE/2025 metadata follows the final thesis reference. This deck does not claim a new paper submission or acceptance.
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, 2017: [primary NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
- Hewamalage, Bergmeir and Bandara, *Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions*, IJF 37(1), 2021, 388–427: [author manuscript](https://arxiv.org/abs/1909.00590) and [article DOI](https://doi.org/10.1016/j.ijforecast.2020.06.008).

## Review before presenting

Rehearse the 18:35 plan aloud, review the limitations with the supervisor, and use the final approved/revised slide version as required by the faculty instructions. Test click-to-play video on the evaluation computer; use the hidden static fallback if needed. The source limitations above are left visible rather than concealed by invented methods or results.
