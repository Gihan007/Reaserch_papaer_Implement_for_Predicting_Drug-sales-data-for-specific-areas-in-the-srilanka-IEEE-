import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30, pred_length=1):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.shape[0]

        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        # ProbSparse attention
        Q_scaled = Q / (self.d_model // self.n_heads) ** 0.5

        # Standard attention for simplicity (can be replaced with ProbSparse later)
        attn_weights = torch.matmul(Q_scaled, K.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(attn_output)
        return self.dropout(output)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + attn_output)

        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, seq_len, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, seq_len, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class Informer(nn.Module):
    def __init__(self, input_size, d_model=512, n_heads=8, n_layers=3, d_ff=2048, seq_len=30, pred_len=1, dropout=0.1):
        super(Informer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.encoder = InformerEncoder(d_model, n_heads, n_layers, d_ff, seq_len, dropout)
        self.output_projection = nn.Linear(d_model, pred_len)

        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        pos_encoding = self._get_positional_encoding(x.shape[1], self.d_model).to(x.device)
        x = x + pos_encoding.unsqueeze(0)

        # Encoder
        x = self.encoder(x)  # (batch_size, seq_len, d_model)

        # Global average pooling and output projection
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        output = self.output_projection(x)  # (batch_size, pred_len)

        return output

    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

def train_informer_model(category, base_path='../', model_dir='./models_informer/', seq_length=30, epochs=100):
    """
    Train Informer model
    """
    try:
        # Load and prepare data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Normalize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, seq_length=seq_length, pred_length=1)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = Informer(input_size=1, seq_len=seq_length, pred_len=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
                x = x.unsqueeze(-1)  # Add feature dimension
                output = model(x)
                loss = criterion(output.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Save model and scaler
        model_path = os.path.join(model_dir, f'informer_{category}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')

        torch.save(model.state_dict(), model_path)

        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Informer model trained and saved for {category}")
        return True

    except Exception as e:
        print(f"Error training Informer model for {category}: {str(e)}")
        return False

def forecast_informer(category, n_steps=1, base_path='../', model_dir='./models_informer/', seq_length=30):
    """
    Generate forecast using trained Informer model
    """
    try:
        # Load data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Load model and scaler
        model_path = os.path.join(model_dir, f'informer_{category}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model files not found for {category}")
            return None

        # Load scaler and scale data
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        data_scaled = scaler.transform(data.values.reshape(-1, 1)).flatten()

        # Load model
        model = Informer(input_size=1, seq_len=seq_length, pred_len=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Prepare input sequence
        input_seq = data_scaled[-seq_length:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # Generate forecast
        with torch.no_grad():
            forecast_scaled = model(input_tensor).squeeze().item()

        # Inverse transform
        forecast_value = scaler.inverse_transform(np.array([[forecast_scaled]]))[0][0]

        return float(forecast_value)

    except Exception as e:
        print(f"Error forecasting with Informer for {category}: {str(e)}")
        return None