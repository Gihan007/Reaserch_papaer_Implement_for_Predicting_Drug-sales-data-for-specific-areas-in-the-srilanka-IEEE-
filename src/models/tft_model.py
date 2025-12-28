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

class TFTDataset(Dataset):
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

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size=30, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()

        self.input_projection = nn.Linear(1, d_model)

        # Multi-head attention layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )

        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        pos_encoding = self._get_positional_encoding(x.shape[1], x.shape[2]).to(x.device)
        x = x + pos_encoding.unsqueeze(0)

        # Encoder
        x = self.encoder(x)  # (batch_size, seq_len, d_model)

        # Global average pooling and output projection
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        output = self.output_projection(x)  # (batch_size, 1)

        return output

    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

def train_tft_model(category, base_path='../', model_dir='./models_tft/', seq_length=30, epochs=100):
    """
    Train Temporal Fusion Transformer model
    """
    try:
        # Load and prepare data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Normalize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Create dataset
        dataset = TFTDataset(data_scaled, seq_length=seq_length, pred_length=1)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = TemporalFusionTransformer(input_size=1, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1)
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
        model_path = os.path.join(model_dir, f'tft_{category}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')

        torch.save(model.state_dict(), model_path)

        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"TFT model trained and saved for {category}")
        return True

    except Exception as e:
        print(f"Error training TFT model for {category}: {str(e)}")
        return False

def forecast_tft(category, n_steps=1, base_path='../', model_dir='./models_tft/', seq_length=30):
    """
    Generate forecast using trained TFT model
    """
    try:
        # Load data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Load model and scaler
        model_path = os.path.join(model_dir, f'tft_{category}.pth')
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
        model = TemporalFusionTransformer(input_size=1, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1)
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
        print(f"Error forecasting with TFT for {category}: {str(e)}")
        return None