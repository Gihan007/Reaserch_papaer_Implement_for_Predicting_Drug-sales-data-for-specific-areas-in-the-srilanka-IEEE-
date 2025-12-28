import xgboost as xgb
import numpy as np

def fit_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_test):
    return model.predict(X_test)
