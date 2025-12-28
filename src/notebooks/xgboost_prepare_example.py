import pandas as pd
from src.utils.preprocessing import create_lagged_features

# Load C1 data
c1 = pd.read_csv('../../C1.csv', parse_dates=['datum'], index_col='datum')

# Prepare lagged features (e.g., 5 lags)
df_lagged = create_lagged_features(c1['C1'], n_lags=5)

# Features and target
y = df_lagged['C1']
X = df_lagged.drop('C1', axis=1)

print('X shape:', X.shape)
print('y shape:', y.shape)
print(X.head())
print(y.head())
