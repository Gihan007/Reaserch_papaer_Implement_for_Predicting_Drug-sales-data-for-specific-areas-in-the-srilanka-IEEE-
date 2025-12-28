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

class NBEATSDataset(Dataset):
    def __init__(self, data, seq_length=30, forecast_length=1):
        self.data = data
        self.seq_length = seq_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NBEATSBlock(nn.Module):
    def __init__(self, input_size, n_neurons, n_layers):
        super(NBEATSBlock, self).__init__()

        self.layers = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(input_size if i == 0 else n_neurons, n_neurons))
            self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

        self.theta_f = nn.Linear(n_neurons, 1)  # Forecast 1 step
        self.theta_b = nn.Linear(n_neurons, input_size)  # Backcast to input size

    def forward(self, x):
        x = self.layers(x)
        theta_f = self.theta_f(x)
        theta_b = self.theta_b(x)

        backcast = theta_b
        forecast = theta_f

        return backcast, forecast

class NBEATS(nn.Module):
    def __init__(self, input_size=30, output_size=1, n_blocks=3, n_neurons=512, n_layers=4):
        super(NBEATS, self).__init__()

        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, n_neurons, n_layers)
            for _ in range(n_blocks)
        ])

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        backcast = x
        forecast = torch.zeros(x.shape[0], self.output_size, device=x.device)

        for block in self.blocks:
            b, f = block(backcast)
            backcast = backcast - b
            forecast = forecast + f

        return forecast

def train_nbeats_model(category, base_path='../', model_dir='./models_nbeats/', seq_length=30, epochs=100):
    """
    Train N-BEATS model
    """
    try:
        # Load and prepare data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Normalize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Create dataset
        dataset = NBEATSDataset(data_scaled, seq_length=seq_length, forecast_length=1)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = NBEATS(input_size=seq_length, output_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
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
        model_path = os.path.join(model_dir, f'nbeats_{category}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')

        torch.save(model.state_dict(), model_path)

        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"N-BEATS model trained and saved for {category}")
        return True

    except Exception as e:
        print(f"Error training N-BEATS model for {category}: {str(e)}")
        return False

def forecast_nbeats(category, n_steps=1, base_path='../', model_dir='./models_nbeats/', seq_length=30):
    """
    Generate forecast using trained N-BEATS model
    """
    try:
        # Load data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Load model and scaler
        model_path = os.path.join(model_dir, f'nbeats_{category}.pth')
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
        model = NBEATS(input_size=seq_length, output_size=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Prepare input sequence
        input_seq = data_scaled[-seq_length:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length)

        # Generate forecast
        with torch.no_grad():
            forecast_scaled = model(input_tensor).squeeze().item()

        # Inverse transform
        forecast_value = scaler.inverse_transform(np.array([[forecast_scaled]]))[0][0]

        return float(forecast_value)

    except Exception as e:
        print(f"Error forecasting with N-BEATS for {category}: {str(e)}")
        return None