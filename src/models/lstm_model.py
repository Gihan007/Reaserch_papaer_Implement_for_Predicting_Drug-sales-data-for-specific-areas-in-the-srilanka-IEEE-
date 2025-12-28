import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model(data, seq_length=10, epochs=100, lr=0.001):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    model = TimeSeriesLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X, y = create_sequences(data_scaled, seq_length)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return model, scaler

def predict_lstm(model, data, seq_length=10, n_steps=1):
    model.eval()
    predictions = []
    current_seq = data[-seq_length:].copy()
    for _ in range(n_steps):
        X = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(X).item()
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)
    return predictions

def forecast_lstm(category, seq_length=10, n_steps=1, base_path='', model_dir='./models_lstm/'):
    df = pd.read_csv(f'{base_path}{category}.csv', parse_dates=['datum'], index_col='datum')
    model = TimeSeriesLSTM()
    model.load_state_dict(torch.load(f'{model_dir}{category}_lstm.pth'))
    scaler = joblib.load(f'{model_dir}{category}_scaler.pkl')
    model.eval()
    data = df[category].values
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    predictions_scaled = predict_lstm(model, data_scaled, seq_length, n_steps)
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return predictions[-1]  # return the last prediction for the week_offset