import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_transformer_model(data, seq_length=10, epochs=100, lr=0.001):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    model = TimeSeriesTransformer()
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

def predict_transformer(model, data, seq_length=10, n_steps=1):
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

def forecast_transformer(category, seq_length=10, n_steps=1, base_path='', model_dir='./models_transformer/'):
    df = pd.read_csv(f'{base_path}{category}.csv', parse_dates=['datum'], index_col='datum')
    model = TimeSeriesTransformer()
    model.load_state_dict(torch.load(f'{model_dir}{category}_transformer.pth'))
    scaler = joblib.load(f'{model_dir}{category}_scaler.pkl')
    model.eval()
    data = df[category].values
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    predictions_scaled = predict_transformer(model, data_scaled, seq_length, n_steps)
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    return predictions[-1]  # return the last prediction for the week_offset