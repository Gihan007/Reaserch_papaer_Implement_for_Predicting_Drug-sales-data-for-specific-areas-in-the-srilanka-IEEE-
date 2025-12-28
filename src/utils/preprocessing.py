import pandas as pd

def create_lagged_features(series, n_lags=5):
    """Convert a time series into a DataFrame with lagged features for supervised learning."""
    df = pd.DataFrame(series)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[series.name].shift(lag)
    df = df.dropna()
    return df
