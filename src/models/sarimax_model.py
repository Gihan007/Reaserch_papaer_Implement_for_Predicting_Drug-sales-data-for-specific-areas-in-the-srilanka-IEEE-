import statsmodels.api as sm

def fit_sarimax(series, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

def predict_sarimax(results, start, end, dynamic=True):
    return results.predict(start=start, end=end, dynamic=dynamic)
