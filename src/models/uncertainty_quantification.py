import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class MCDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    def __init__(self, model, dropout_rate=0.1):
        super(MCDropout, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

        # Enable dropout during inference
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x, n_samples=50):
        """Forward pass with Monte Carlo sampling"""
        predictions = []

        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std, predictions

class BayesianLinear(nn.Module):
    """Bayesian Linear Layer"""
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Sample weights and biases from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return torch.matmul(x, weight.t()) + bias

class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network for uncertainty quantification"""
    def __init__(self, input_size=30, hidden_size=128, output_size=1):
        super(BayesianNeuralNetwork, self).__init__()

        self.bayesian_layer1 = BayesianLinear(input_size, hidden_size)
        self.bayesian_layer2 = BayesianLinear(hidden_size, hidden_size)
        self.bayesian_output = BayesianLinear(hidden_size, output_size)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bayesian_layer1(x))
        x = self.activation(self.bayesian_layer2(x))
        x = self.bayesian_output(x)
        return x

    def predict_with_uncertainty(self, x, n_samples=100):
        """Predict with uncertainty using Bayesian sampling"""
        predictions = []

        for _ in range(n_samples):
            pred = self.forward(x)
            predictions.append(pred.detach().cpu().numpy())

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std, predictions

class ConformalPredictor:
    """Conformal Prediction for guaranteed coverage"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Significance level (1 - confidence)
        self.calibration_scores = []

    def calibrate(self, model, X_cal, y_cal):
        """Calibrate the conformal predictor"""
        predictions = []

        for x in X_cal:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            pred = model(x_tensor).item()
            predictions.append(pred)

        # Compute nonconformity scores
        self.calibration_scores = [abs(pred - actual) for pred, actual in zip(predictions, y_cal)]
        self.calibration_scores.sort()

    def predict_interval(self, x, model):
        """Predict with conformal prediction interval"""
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        point_prediction = model(x_tensor).item()

        # Find the quantile
        n = len(self.calibration_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        quantile_idx = min(quantile_idx, n - 1)

        if quantile_idx >= 0:
            q_hat = self.calibration_scores[quantile_idx]
        else:
            q_hat = 0

        lower_bound = point_prediction - q_hat
        upper_bound = point_prediction + q_hat

        return point_prediction, lower_bound, upper_bound

def train_bayesian_nn(category, base_path='../', model_dir='./models_bayesian/', seq_length=30, epochs=100):
    """Train Bayesian Neural Network"""
    try:
        # Load and prepare data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Normalize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i:i + seq_length])
            y.append(data_scaled[i + seq_length])

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Create model
        model = BayesianNeuralNetwork(input_size=seq_length)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Save model and scaler
        model_path = os.path.join(model_dir, f'bayesian_{category}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')

        torch.save(model.state_dict(), model_path)

        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Bayesian NN trained and saved for {category}")
        return True

    except Exception as e:
        print(f"Error training Bayesian NN for {category}: {str(e)}")
        return False

def forecast_with_uncertainty(category, model_type='mc_dropout', n_samples=50, base_path='../', model_dir='./models_bayesian/', seq_length=30):
    """Generate forecast with uncertainty quantification"""
    try:
        # Load data
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Load scaler
        scaler_path = os.path.join(model_dir, f'scaler_{category}.pkl')
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        data_scaled = scaler.transform(data.values.reshape(-1, 1)).flatten()

        if model_type == 'bayesian':
            # Load Bayesian model
            model_path = os.path.join(model_dir, f'bayesian_{category}.pth')
            model = BayesianNeuralNetwork(input_size=seq_length)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Prepare input
            input_seq = data_scaled[-seq_length:]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

            # Predict with uncertainty
            mean, std, samples = model.predict_with_uncertainty(input_tensor, n_samples)

            forecast_value = scaler.inverse_transform(mean.reshape(-1, 1))[0][0]
            uncertainty = scaler.inverse_transform((mean + std).reshape(-1, 1))[0][0] - forecast_value

        elif model_type == 'mc_dropout':
            # Load regular model (using TFT as example)
            from models.tft_model import TemporalFusionTransformer
            model_path = os.path.join('../models_tft', f'tft_{category}.pth')
            model = TemporalFusionTransformer()
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Apply MC Dropout
            mc_model = MCDropout(model)

            # Prepare input
            input_seq = data_scaled[-seq_length:]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            # Predict with uncertainty
            mean, std, samples = mc_model(input_tensor, n_samples)

            forecast_value = scaler.inverse_transform(mean.reshape(-1, 1))[0][0]
            uncertainty = scaler.inverse_transform((mean + std).reshape(-1, 1))[0][0] - forecast_value

        elif model_type == 'conformal':
            # Load regular model
            from models.tft_model import TemporalFusionTransformer
            model_path = os.path.join('../models_tft', f'tft_{category}.pth')
            model = TemporalFusionTransformer()
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Prepare calibration data (last 20% for calibration)
            cal_size = int(0.2 * len(data_scaled))
            X_cal = [data_scaled[i:i + seq_length] for i in range(len(data_scaled) - seq_length - cal_size, len(data_scaled) - seq_length)]
            y_cal = [data_scaled[i + seq_length] for i in range(len(data_scaled) - seq_length - cal_size, len(data_scaled) - seq_length)]

            # Calibrate conformal predictor
            conformal_pred = ConformalPredictor(alpha=0.1)
            conformal_pred.calibrate(model, X_cal, y_cal)

            # Predict with conformal intervals
            input_seq = data_scaled[-seq_length:]
            point_pred, lower_bound, upper_bound = conformal_pred.predict_interval(input_seq, model)

            forecast_value = scaler.inverse_transform(np.array([[point_pred]]))[0][0]
            lower_bound_scaled = scaler.inverse_transform(np.array([[lower_bound]]))[0][0]
            upper_bound_scaled = scaler.inverse_transform(np.array([[upper_bound]]))[0][0]

            return {
                'forecast': forecast_value,
                'lower_bound': lower_bound_scaled,
                'upper_bound': upper_bound_scaled,
                'uncertainty': upper_bound_scaled - lower_bound_scaled
            }

        return {
            'forecast': forecast_value,
            'uncertainty': uncertainty,
            'confidence_interval': [forecast_value - 1.96 * uncertainty, forecast_value + 1.96 * uncertainty]
        }

    except Exception as e:
        print(f"Error forecasting with uncertainty for {category}: {str(e)}")
        return None