"""
Federated Learning for Drug Sales Prediction
Privacy-preserving collaborative training across multiple pharmacies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import time
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PharmacyClient:
    """Represents a single pharmacy in the federated learning system"""

    def __init__(self, client_id: str, data: np.ndarray, model_config: Dict[str, Any]):
        self.client_id = client_id
        self.data = data
        self.scaler = MinMaxScaler()
        self.model_config = model_config
        self.local_model = None
        self.training_history = []

    def prepare_data(self, seq_length: int = 14) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare local data for training"""
        # Normalize data
        data_normalized = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(len(data_normalized) - seq_length):
            X.append(data_normalized[i:i + seq_length])
            y.append(data_normalized[i + seq_length])

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return X, y

    def initialize_model(self, global_model_state: Dict[str, torch.Tensor]):
        """Initialize local model with global parameters"""
        self.local_model = self._create_model()
        self.local_model.load_state_dict(global_model_state)

    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        class SimpleDrugPredictor(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
                super(SimpleDrugPredictor, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )

            def forward(self, x):
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
                return self.network(x)

        seq_length = self.model_config.get('seq_length', 14)
        hidden_dim = self.model_config.get('hidden_dim', 64)
        dropout = self.model_config.get('dropout', 0.2)

        return SimpleDrugPredictor(seq_length, hidden_dim, 1, dropout)

    def train_local_model(self, epochs: int = 5, batch_size: int = 32,
                         learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train local model on client's data"""

        if self.local_model is None:
            raise ValueError("Model not initialized")

        # Prepare data
        X, y = self.prepare_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)

        # Training loop
        epoch_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.local_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            epoch_losses.append(epoch_loss)

        # Calculate final metrics
        self.local_model.eval()
        with torch.no_grad():
            predictions = []
            actuals = []

            for X_batch, y_batch in dataloader:
                outputs = self.local_model(X_batch)
                predictions.extend(outputs.numpy().flatten())
                actuals.extend(y_batch.numpy().flatten())

            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))

        training_result = {
            'final_loss': epoch_losses[-1],
            'mae': mae,
            'rmse': rmse,
            'epochs': epochs,
            'model_state': self.local_model.state_dict(),
            'loss_history': epoch_losses
        }

        self.training_history.append(training_result)
        return training_result

    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get the trained model parameters for aggregation"""
        if self.local_model is None:
            raise ValueError("Model not trained")

        return self.local_model.state_dict()

class FederatedAggregator:
    """Handles aggregation of model updates from multiple clients"""

    def __init__(self, aggregation_method: str = 'fedavg'):
        self.aggregation_method = aggregation_method
        self.supported_methods = ['fedavg', 'fednova', 'scaffold']

    def aggregate_models(self, client_updates: List[Dict[str, torch.Tensor]],
                        client_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Aggregate model updates from multiple clients"""

        if not client_updates:
            raise ValueError("No client updates provided")

        if self.aggregation_method == 'fedavg':
            return self._fedavg_aggregation(client_updates, client_weights)
        elif self.aggregation_method == 'fednova':
            return self._fednova_aggregation(client_updates, client_weights)
        else:
            # Default to FedAvg
            return self._fedavg_aggregation(client_updates, client_weights)

    def _fedavg_aggregation(self, client_updates: List[Dict[str, torch.Tensor]],
                           client_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Federated Averaging (FedAvg) aggregation"""

        if client_weights is None:
            # Equal weighting
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        # Initialize aggregated parameters
        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            param_sum = None

            for update, weight in zip(client_updates, client_weights):
                if param_sum is None:
                    param_sum = update[param_name] * weight
                else:
                    param_sum += update[param_name] * weight

            aggregated_params[param_name] = param_sum

        return aggregated_params

    def _fednova_aggregation(self, client_updates: List[Dict[str, torch.Tensor]],
                            client_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """FedNova aggregation (simplified version)"""
        # For simplicity, using FedAvg. Full FedNova would require gradient tracking
        return self._fedavg_aggregation(client_updates, client_weights)

class FederatedLearningSystem:
    """Main federated learning coordinator"""

    def __init__(self, num_clients: int = 5, aggregation_method: str = 'fedavg',
                 model_config: Optional[Dict[str, Any]] = None):
        self.num_clients = num_clients
        self.aggregator = FederatedAggregator(aggregation_method)
        self.model_config = model_config or {
            'seq_length': 14,
            'hidden_dim': 64,
            'dropout': 0.2
        }

        self.clients = []
        self.global_model_state = None
        self.training_history = []
        self.round_history = []

    def initialize_clients(self, data_distributions: List[np.ndarray]):
        """Initialize clients with their local data"""

        if len(data_distributions) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} data distributions, got {len(data_distributions)}")

        self.clients = []
        for i, data in enumerate(data_distributions):
            client = PharmacyClient(f"pharmacy_{i+1}", data, self.model_config)
            self.clients.append(client)

        print(f"✅ Initialized {len(self.clients)} pharmacy clients")

    def create_synthetic_data_distributions(self, base_data: np.ndarray,
                                          num_clients: int = 5,
                                          distribution_type: str = 'iid') -> List[np.ndarray]:
        """Create synthetic data distributions for clients"""

        if distribution_type == 'iid':
            # Independent and identically distributed
            data_per_client = len(base_data) // num_clients
            distributions = []

            for i in range(num_clients):
                start_idx = i * data_per_client
                end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(base_data)

                # Add some noise to make it more realistic
                client_data = base_data[start_idx:end_idx].copy()
                noise = np.random.normal(0, 0.1, len(client_data))
                client_data = client_data + noise

                distributions.append(client_data)

        elif distribution_type == 'non_iid':
            # Non-independent and identically distributed
            # Some clients have more data from certain periods
            distributions = []

            for i in range(num_clients):
                if i < num_clients // 2:
                    # First half of clients get early data
                    start_idx = 0
                    end_idx = len(base_data) // 2
                else:
                    # Second half get later data
                    start_idx = len(base_data) // 4
                    end_idx = len(base_data)

                client_data = base_data[start_idx:end_idx].copy()
                # Add different amounts of noise
                noise_level = 0.05 + (i * 0.05)  # Increasing noise
                noise = np.random.normal(0, noise_level, len(client_data))
                client_data = client_data + noise

                distributions.append(client_data)

        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

        return distributions

    def initialize_global_model(self):
        """Initialize the global model"""
        # Create a dummy client to get model architecture
        dummy_client = PharmacyClient("dummy", np.array([1.0, 2.0, 3.0]), self.model_config)
        dummy_model = dummy_client._create_model()
        self.global_model_state = dummy_model.state_dict()

        print("✅ Global model initialized")

    def run_federated_training(self, num_rounds: int = 10,
                             local_epochs: int = 5,
                             client_fraction: float = 0.8) -> Dict[str, Any]:
        """Run federated training for specified number of rounds"""

        print(f"🚀 Starting Federated Learning Training")
        print(f"Rounds: {num_rounds}, Clients: {len(self.clients)}, Fraction: {client_fraction}")
        print("=" * 60)

        for round_num in range(num_rounds):
            print(f"\n🔄 Round {round_num + 1}/{num_rounds}")

            # Select clients for this round
            num_clients_this_round = max(1, int(len(self.clients) * client_fraction))
            selected_clients = random.sample(self.clients, num_clients_this_round)

            print(f"Selected {len(selected_clients)} clients for training")

            # Send global model to selected clients
            client_updates = []

            for client in selected_clients:
                try:
                    # Initialize client with global model
                    client.initialize_model(self.global_model_state)

                    # Train locally
                    training_result = client.train_local_model(epochs=local_epochs)

                    # Get model update
                    model_update = client.get_model_update()
                    client_updates.append(model_update)

                    print(f"  📊 {client.client_id}: Loss = {training_result['final_loss']:.4f}, "
                          f"MAE = {training_result['mae']:.4f}")

                except Exception as e:
                    print(f"  ❌ Error training {client.client_id}: {e}")
                    continue

            # Aggregate updates
            if client_updates:
                self.global_model_state = self.aggregator.aggregate_models(client_updates)
                print("  ✅ Model updates aggregated")
            else:
                print("  ⚠️ No valid updates to aggregate")

            # Evaluate global model
            round_metrics = self._evaluate_global_model()
            self.round_history.append({
                'round': round_num + 1,
                'clients_participated': len(client_updates),
                'global_metrics': round_metrics
            })

            print(f"  📊 Round {round_num + 1} - MAE: {round_metrics['mae']:.4f}, "
                  f"RMSE: {round_metrics['rmse']:.4f}")

        # Final evaluation
        final_metrics = self._evaluate_global_model()

        results = {
            'num_rounds': num_rounds,
            'num_clients': len(self.clients),
            'final_metrics': final_metrics,
            'round_history': self.round_history,
            'training_time': time.time(),
            'aggregation_method': self.aggregator.aggregation_method
        }

        print("\n✅ Federated training completed!")
        print(f"   📊 Final MAE: {results['final_metrics']['mae']:.4f}")
        print(f"   📈 Final RMSE: {results['final_metrics']['rmse']:.4f}")

        return results

    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the current global model on all clients' data"""

        all_predictions = []
        all_actuals = []

        # Create temporary model for evaluation
        dummy_client = PharmacyClient("eval", np.array([1.0]), self.model_config)
        eval_model = dummy_client._create_model()
        eval_model.load_state_dict(self.global_model_state)
        eval_model.eval()

        for client in self.clients:
            try:
                X, y = client.prepare_data()

                with torch.no_grad():
                    predictions = eval_model(X)
                    all_predictions.extend(predictions.numpy().flatten())
                    all_actuals.extend(y.numpy().flatten())

            except Exception as e:
                print(f"Error evaluating on {client.client_id}: {e}")
                continue

        if not all_predictions:
            return {'mae': float('inf'), 'rmse': float('inf')}

        mae = mean_absolute_error(all_actuals, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))

        return {'mae': mae, 'rmse': rmse}

    def save_federated_model(self, filepath: str):
        """Save the trained federated model"""
        if self.global_model_state is None:
            raise ValueError("No trained model to save")

        torch.save({
            'model_state_dict': self.global_model_state,
            'model_config': self.model_config,
            'num_clients': len(self.clients),
            'training_history': self.training_history,
            'round_history': self.round_history
        }, filepath)

        print(f"💾 Federated model saved to {filepath}")

    def load_federated_model(self, filepath: str):
        """Load a trained federated model"""
        checkpoint = torch.load(filepath)
        self.global_model_state = checkpoint['model_state_dict']
        self.model_config = checkpoint['model_config']
        self.training_history = checkpoint.get('training_history', [])
        self.round_history = checkpoint.get('round_history', [])

        print(f"📂 Federated model loaded from {filepath}")

    def plot_training_progress(self, save_path: str = './federated_training_progress.png'):
        """Plot federated training progress"""

        if not self.round_history:
            print("No training history to plot")
            return

        rounds = [r['round'] for r in self.round_history]
        mae_scores = [r['global_metrics']['mae'] for r in self.round_history]
        rmse_scores = [r['global_metrics']['rmse'] for r in self.round_history]
        clients_participated = [r['clients_participated'] for r in self.round_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # MAE over rounds
        ax1.plot(rounds, mae_scores, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Global Model MAE Over Training Rounds')
        ax1.grid(True, alpha=0.3)

        # RMSE over rounds
        ax2.plot(rounds, rmse_scores, marker='s', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Root Mean Square Error')
        ax2.set_title('Global Model RMSE Over Training Rounds')
        ax2.grid(True, alpha=0.3)

        # Clients participated
        ax3.bar(rounds, clients_participated, alpha=0.7, color='green')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Clients Participated')
        ax3.set_title('Client Participation Per Round')
        ax3.grid(True, alpha=0.3)

        # Performance improvement
        if len(mae_scores) > 1:
            improvement = [(mae_scores[0] - mae) / mae_scores[0] * 100 for mae in mae_scores]
            ax4.plot(rounds, improvement, marker='^', linewidth=2, markersize=8, color='purple')
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Performance Improvement (%)')
            ax4.set_title('Performance Improvement Over Rounds')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Training progress plot saved to {save_path}")

def run_federated_drug_prediction(category: str = 'C1', num_clients: int = 5,
                                num_rounds: int = 8, distribution_type: str = 'iid'):
    """Run federated learning for drug sales prediction"""

    print(f"🏥 Federated Learning for Drug Sales Prediction - {category}")
    print("=" * 60)

    # Load base data
    try:
        df = pd.read_csv(f'{category}.csv')
        base_data = df[category].values
        print(f"📊 Loaded {len(base_data)} data points for {category}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

    # Initialize federated learning system
    fed_system = FederatedLearningSystem(num_clients=num_clients)

    # Create synthetic data distributions for clients
    print(f"🔄 Creating {distribution_type} data distributions for {num_clients} clients...")
    data_distributions = fed_system.create_synthetic_data_distributions(
        base_data, num_clients, distribution_type
    )

    # Initialize clients
    fed_system.initialize_clients(data_distributions)

    # Initialize global model
    fed_system.initialize_global_model()

    # Run federated training
    results = fed_system.run_federated_training(num_rounds=num_rounds)

    # Save results
    os.makedirs('./federated_results', exist_ok=True)
    results_file = f'./federated_results/fed_results_{category}_{distribution_type}_{int(time.time())}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save model
    model_file = f'./federated_results/fed_model_{category}_{distribution_type}.pth'
    fed_system.save_federated_model(model_file)

    # Plot training progress
    plot_file = f'./federated_results/fed_training_{category}_{distribution_type}.png'
    fed_system.plot_training_progress(plot_file)

    print("\n📈 Federated Learning Summary:")
    print(f"  Final MAE: {results['final_metrics']['mae']:.4f}")
    print(f"  Final RMSE: {results['final_metrics']['rmse']:.4f}")
    print(f"  Results saved to: {results_file}")
    print(f"  Model saved to: {model_file}")
    print(f"  Plot saved to: {plot_file}")

    return results

def compare_federated_vs_centralized(category: str = 'C1'):
    """Compare federated learning with centralized training"""

    print(f"🔍 Comparing Federated vs Centralized Learning for {category}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(f'{category}.csv')
    data = df[category].values

    # Split data for centralized training
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Centralized training
    print("🏛️ Training centralized model...")
    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

    # Simple centralized model
    class CentralizedPredictor(nn.Module):
        def __init__(self):
            super(CentralizedPredictor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(14, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            return self.network(x)

    model = CentralizedPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare training data
    X, y = [], []
    for i in range(len(train_normalized) - 14):
        X.append(train_normalized[i:i + 14])
        y.append(train_normalized[i + 14])

    X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train centralized model
    for epoch in range(20):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate centralized model
    test_normalized = scaler.transform(test_data.reshape(-1, 1)).flatten()
    X_test, y_test = [], []
    for i in range(len(test_normalized) - 14):
        X_test.append(test_normalized[i:i + 14])
        y_test.append(test_normalized[i + 14])

    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        centralized_mae = mean_absolute_error(y_test.numpy().flatten(), predictions.numpy().flatten())
        centralized_rmse = np.sqrt(mean_squared_error(y_test.numpy().flatten(), predictions.numpy().flatten()))

    # Federated training
    print("🌐 Training federated model...")
    fed_results = run_federated_drug_prediction(category, num_clients=5, num_rounds=8)
    federated_mae = fed_results['final_metrics']['mae']
    federated_rmse = fed_results['final_metrics']['rmse']

    # Comparison
    print("\n📊 Comparison Results:")
    print(f"  Centralized - MAE: {centralized_mae:.4f}, RMSE: {centralized_rmse:.4f}")
    print(f"  Federated   - MAE: {federated_mae:.4f}, RMSE: {federated_rmse:.4f}")

    mae_improvement = (centralized_mae - federated_mae) / centralized_mae * 100
    rmse_improvement = (centralized_rmse - federated_rmse) / centralized_rmse * 100

    print(f"  MAE Improvement: {mae_improvement:.1f}%")
    print(f"  RMSE Improvement: {rmse_improvement:.1f}%")
    # Save comparison
    comparison = {
        'category': category,
        'centralized': {'mae': centralized_mae, 'rmse': centralized_rmse},
        'federated': {'mae': federated_mae, 'rmse': federated_rmse},
        'improvement': {'mae_percent': mae_improvement, 'rmse_percent': rmse_improvement}
    }

    with open(f'./federated_results/comparison_{category}.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison

if __name__ == "__main__":
    # Example usage
    print("🏥 Federated Learning for Drug Sales Prediction")
    print("=" * 60)

    # Run federated learning
    results = run_federated_drug_prediction('C1', num_clients=5, num_rounds=5)

    # Compare with centralized learning
    comparison = compare_federated_vs_centralized('C1')

    print("\n🎉 Federated learning demonstration completed!")
    print("Check the 'federated_results' directory for outputs.")