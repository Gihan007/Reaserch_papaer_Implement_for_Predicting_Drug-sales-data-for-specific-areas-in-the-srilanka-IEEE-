"""
Neural Architecture Search for Drug Sales Prediction
Automatically discovers optimal neural network architectures for different drug categories
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
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ArchitectureConfig:
    """Configuration for a neural network architecture"""
    num_layers: int
    hidden_dims: List[int]
    dropout_rates: List[float]
    activation: str
    learning_rate: float
    batch_size: int
    seq_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_layers': self.num_layers,
            'hidden_dims': self.hidden_dims,
            'dropout_rates': self.dropout_rates,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'seq_length': self.seq_length
        }

class DynamicNeuralNetwork(nn.Module):
    """Dynamically configurable neural network for NAS"""

    def __init__(self, config: ArchitectureConfig, input_dim: int = 1, output_dim: int = 1):
        super(DynamicNeuralNetwork, self).__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers dynamically
        layers = []
        current_dim = input_dim * config.seq_length

        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dims[i]),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rates[i])
            ])
            current_dim = config.hidden_dims[i]

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

class ArchitectureEvaluator:
    """Evaluates neural network architectures on drug sales data"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.scaler = MinMaxScaler()

    def prepare_data(self, data: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare time series data for training"""
        # Normalize data
        data_normalized = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(len(data_normalized) - seq_length):
            X.append(data_normalized[i:i + seq_length])
            y.append(data_normalized[i + seq_length])

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return X, y

    def evaluate_architecture(self, config: ArchitectureConfig, train_data: np.ndarray,
                            val_data: np.ndarray, max_epochs: int = 50) -> Dict[str, float]:
        """Evaluate a single architecture"""

        # Prepare data
        X_train, y_train = self.prepare_data(train_data, config.seq_length)
        X_val, y_val = self.prepare_data(val_data, config.seq_length)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize model
        model = DynamicNeuralNetwork(config).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_actuals = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_actuals.extend(y_batch.cpu().numpy().flatten())

            val_loss /= len(val_loader)

            # Calculate metrics
            val_predictions = np.array(val_predictions)
            val_actuals = np.array(val_actuals)

            mae = mean_absolute_error(val_actuals, val_predictions)
            rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return {
            'val_loss': best_val_loss,
            'mae': mae,
            'rmse': rmse,
            'epochs_trained': epoch + 1
        }

class EvolutionarySearch:
    """Evolutionary algorithm for neural architecture search"""

    def __init__(self, population_size: int = 20, generations: int = 10,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Define search space
        self.search_space = {
            'num_layers': [2, 3, 4, 5],
            'hidden_dims': [32, 64, 128, 256],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activations': ['relu', 'tanh', 'leaky_relu', 'elu'],
            'learning_rates': [0.001, 0.0005, 0.0001],
            'batch_sizes': [16, 32, 64],
            'seq_lengths': [7, 14, 21, 30]
        }

    def generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random architecture configuration"""
        num_layers = random.choice(self.search_space['num_layers'])

        # Ensure hidden_dims list matches num_layers
        hidden_dims = [random.choice(self.search_space['hidden_dims']) for _ in range(num_layers)]
        dropout_rates = [random.choice(self.search_space['dropout_rates']) for _ in range(num_layers)]

        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=random.choice(self.search_space['activations']),
            learning_rate=random.choice(self.search_space['learning_rates']),
            batch_size=random.choice(self.search_space['batch_sizes']),
            seq_length=random.choice(self.search_space['seq_lengths'])
        )

    def mutate_architecture(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture configuration"""
        new_config = ArchitectureConfig(
            num_layers=config.num_layers,
            hidden_dims=config.hidden_dims.copy(),
            dropout_rates=config.dropout_rates.copy(),
            activation=config.activation,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            seq_length=config.seq_length
        )

        # Randomly mutate different parameters
        if random.random() < self.mutation_rate:
            # Mutate number of layers
            new_config.num_layers = random.choice(self.search_space['num_layers'])
            # Adjust hidden_dims and dropout_rates to match
            while len(new_config.hidden_dims) < new_config.num_layers:
                new_config.hidden_dims.append(random.choice(self.search_space['hidden_dims']))
                new_config.dropout_rates.append(random.choice(self.search_space['dropout_rates']))
            new_config.hidden_dims = new_config.hidden_dims[:new_config.num_layers]
            new_config.dropout_rates = new_config.dropout_rates[:new_config.num_layers]

        if random.random() < self.mutation_rate:
            # Mutate hidden dimensions
            for i in range(len(new_config.hidden_dims)):
                if random.random() < 0.5:
                    new_config.hidden_dims[i] = random.choice(self.search_space['hidden_dims'])

        if random.random() < self.mutation_rate:
            # Mutate dropout rates
            for i in range(len(new_config.dropout_rates)):
                if random.random() < 0.5:
                    new_config.dropout_rates[i] = random.choice(self.search_space['dropout_rates'])

        if random.random() < self.mutation_rate:
            new_config.activation = random.choice(self.search_space['activations'])

        if random.random() < self.mutation_rate:
            new_config.learning_rate = random.choice(self.search_space['learning_rates'])

        if random.random() < self.mutation_rate:
            new_config.batch_size = random.choice(self.search_space['batch_sizes'])

        if random.random() < self.mutation_rate:
            new_config.seq_length = random.choice(self.search_space['seq_lengths'])

        return new_config

    def crossover_architectures(self, parent1: ArchitectureConfig,
                              parent2: ArchitectureConfig) -> Tuple[ArchitectureConfig, ArchitectureConfig]:
        """Perform crossover between two architectures"""
        child1 = ArchitectureConfig(
            num_layers=parent1.num_layers,
            hidden_dims=parent1.hidden_dims.copy(),
            dropout_rates=parent1.dropout_rates.copy(),
            activation=parent1.activation,
            learning_rate=parent1.learning_rate,
            batch_size=parent1.batch_size,
            seq_length=parent1.seq_length
        )

        child2 = ArchitectureConfig(
            num_layers=parent2.num_layers,
            hidden_dims=parent2.hidden_dims.copy(),
            dropout_rates=parent2.dropout_rates.copy(),
            activation=parent2.activation,
            learning_rate=parent2.learning_rate,
            batch_size=parent2.batch_size,
            seq_length=parent2.seq_length
        )

        # Crossover hidden dimensions if same length
        if len(parent1.hidden_dims) == len(parent2.hidden_dims):
            crossover_point = random.randint(1, len(parent1.hidden_dims) - 1)
            child1.hidden_dims = parent1.hidden_dims[:crossover_point] + parent2.hidden_dims[crossover_point:]
            child2.hidden_dims = parent2.hidden_dims[:crossover_point] + parent1.hidden_dims[crossover_point:]

            child1.dropout_rates = parent1.dropout_rates[:crossover_point] + parent2.dropout_rates[crossover_point:]
            child2.dropout_rates = parent2.dropout_rates[:crossover_point] + parent1.dropout_rates[crossover_point:]

        return child1, child2

class DrugPredictionNAS:
    """Neural Architecture Search for Drug Sales Prediction"""

    def __init__(self, save_dir: str = './nas_results'):
        self.save_dir = save_dir
        self.evaluator = ArchitectureEvaluator()
        self.search_algorithm = EvolutionarySearch()
        self.results_history = []

        os.makedirs(save_dir, exist_ok=True)

    def load_drug_data(self, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and split drug sales data"""
        try:
            df = pd.read_csv(f'{category}.csv')
            data = df[category].values

            # Split into train/validation (80/20)
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            val_data = data[split_idx:]

            return train_data, val_data

        except Exception as e:
            print(f"Error loading data for {category}: {e}")
            raise

    def search_optimal_architecture(self, category: str, generations: int = 5) -> Dict[str, Any]:
        """Search for optimal architecture for a drug category"""

        print(f"🔬 Starting NAS for {category}")
        print(f"Population size: {self.search_algorithm.population_size}")
        print(f"Generations: {generations}")

        # Load data
        train_data, val_data = self.load_drug_data(category)

        # Initialize population
        population = [self.search_algorithm.generate_random_architecture()
                     for _ in range(self.search_algorithm.population_size)]

        # Evaluate initial population
        print("\n📊 Evaluating initial population...")
        population_fitness = []

        for i, config in enumerate(population):
            print(f"Evaluating architecture {i+1}/{len(population)}")
            try:
                metrics = self.evaluator.evaluate_architecture(config, train_data, val_data)
                population_fitness.append((config, metrics))
            except Exception as e:
                print(f"Error evaluating architecture {i+1}: {e}")
                # Assign poor fitness to failed architectures
                population_fitness.append((config, {'val_loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}))

        # Sort by validation loss (lower is better)
        population_fitness.sort(key=lambda x: x[1]['val_loss'])

        best_fitness_history = []

        # Evolutionary search
        for generation in range(generations):
            print(f"\n🧬 Generation {generation + 1}/{generations}")

            # Keep best performers (elitism)
            elite_size = max(2, self.search_algorithm.population_size // 5)
            elites = population_fitness[:elite_size]

            # Create new population
            new_population = [config for config, _ in elites]

            # Generate offspring through crossover and mutation
            while len(new_population) < self.search_algorithm.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population_fitness)
                parent2 = self._tournament_selection(population_fitness)

                # Crossover
                if random.random() < self.search_algorithm.crossover_rate:
                    child1, child2 = self.search_algorithm.crossover_architectures(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                child1 = self.search_algorithm.mutate_architecture(child1)
                child2 = self.search_algorithm.mutate_architecture(child2)

                new_population.extend([child1, child2])

            # Trim to population size
            new_population = new_population[:self.search_algorithm.population_size]

            # Evaluate new population
            print("Evaluating new population...")
            new_population_fitness = []

            for i, config in enumerate(new_population):
                try:
                    metrics = self.evaluator.evaluate_architecture(config, train_data, val_data)
                    new_population_fitness.append((config, metrics))
                except Exception as e:
                    print(f"Error evaluating architecture {i+1}: {e}")
                    new_population_fitness.append((config, {'val_loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}))

            # Update population
            population_fitness = new_population_fitness
            population_fitness.sort(key=lambda x: x[1]['val_loss'])

            # Record best fitness
            best_config, best_metrics = population_fitness[0]
            best_fitness_history.append(best_metrics['val_loss'])

            print(".4f"
                  ".4f")

        # Get final best architecture
        best_config, best_metrics = population_fitness[0]

        result = {
            'category': category,
            'best_architecture': best_config.to_dict(),
            'best_metrics': best_metrics,
            'fitness_history': best_fitness_history,
            'total_architectures_evaluated': len(population) * (generations + 1),
            'search_time': time.time()
        }

        # Save results
        self._save_results(result)

        return result

    def _tournament_selection(self, population_fitness: List[Tuple[ArchitectureConfig, Dict]],
                            tournament_size: int = 3) -> ArchitectureConfig:
        """Tournament selection for evolutionary algorithm"""
        # Randomly select tournament participants
        tournament = random.sample(population_fitness, tournament_size)

        # Return the best (lowest validation loss)
        return min(tournament, key=lambda x: x[1]['val_loss'])[0]

    def _save_results(self, result: Dict[str, Any]):
        """Save NAS results to file"""
        filename = f"{self.save_dir}/nas_results_{result['category']}_{int(time.time())}.json"

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"💾 Results saved to {filename}")

    def plot_search_progress(self, result: Dict[str, Any]):
        """Plot the search progress over generations"""
        plt.figure(figsize=(10, 6))
        plt.plot(result['fitness_history'], marker='o', linewidth=2, markersize=8)
        plt.xlabel('Generation')
        plt.ylabel('Best Validation Loss')
        plt.title(f'NAS Progress for {result["category"]}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/nas_progress_{result['category']}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def compare_architectures(self, results: List[Dict[str, Any]]):
        """Compare architectures across different categories"""
        categories = [r['category'] for r in results]
        val_losses = [r['best_metrics']['val_loss'] for r in results]
        maes = [r['best_metrics']['mae'] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Validation Loss
        ax1.bar(range(len(categories)), val_losses)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories)
        ax1.set_ylabel('Best Validation Loss')
        ax1.set_title('Best Architectures by Category (Validation Loss)')

        # MAE
        ax2.bar(range(len(categories)), maes)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Best Architectures by Category (MAE)')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/architecture_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def run_nas_for_all_categories(categories: List[str] = ['C1', 'C2', 'C3', 'C4'],
                              generations: int = 3) -> List[Dict[str, Any]]:
    """Run NAS for all drug categories"""

    nas = DrugPredictionNAS()
    all_results = []

    print("🚀 Starting Neural Architecture Search for all categories")
    print("=" * 60)

    for category in categories:
        try:
            print(f"\n🔍 Searching for {category}...")
            result = nas.search_optimal_architecture(category, generations)
            all_results.append(result)

            # Plot progress
            nas.plot_search_progress(result)

        except Exception as e:
            print(f"❌ Failed to search for {category}: {e}")
            continue

    # Compare all results
    if len(all_results) > 1:
        nas.compare_architectures(all_results)

    # Save summary
    summary = {
        'total_categories': len(all_results),
        'average_best_loss': np.mean([r['best_metrics']['val_loss'] for r in all_results]),
        'average_mae': np.mean([r['best_metrics']['mae'] for r in all_results]),
        'total_architectures_evaluated': sum(r['total_architectures_evaluated'] for r in all_results),
        'results': all_results
    }

    with open(f"{nas.save_dir}/nas_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n✅ NAS completed for all categories!")
    print(f"📊 Summary saved to {nas.save_dir}/nas_summary.json")

    return all_results

if __name__ == "__main__":
    # Example usage
    results = run_nas_for_all_categories(generations=2)  # Reduced for demo
    print("\n🏆 Best architectures found:")
    for result in results:
        print(f"{result['category']}: Val Loss = {result['best_metrics']['val_loss']:.4f}, "
              f"MAE = {result['best_metrics']['mae']:.4f}")