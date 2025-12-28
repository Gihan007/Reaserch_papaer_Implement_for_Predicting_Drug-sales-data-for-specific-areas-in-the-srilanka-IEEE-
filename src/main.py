from src.data.data_loader import load_all_categories
from src.models.sarimax_model import fit_sarimax, predict_sarimax
from src.models.xgboost_model import fit_xgboost, predict_xgboost
from src.utils.preprocessing import create_lagged_features

# Example: Load all category data
data = load_all_categories()

# Example: SARIMAX usage
# results = fit_sarimax(data['C1']['C1'], order=(1,0,0), seasonal_order=(1,0,0,7))
# forecast = predict_sarimax(results, start=90, end=103)

# Example: XGBoost usage
# df_lagged = create_lagged_features(data['C1']['C1'], n_lags=5)
# X = df_lagged.drop('C1', axis=1).values
# y = df_lagged['C1'].values
# model = fit_xgboost(X, y)
# y_pred = predict_xgboost(model, X)

# Add more logic as needed for your workflow
