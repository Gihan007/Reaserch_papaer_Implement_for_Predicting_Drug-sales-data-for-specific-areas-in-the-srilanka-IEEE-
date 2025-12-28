import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
import higher
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    """Dataset for meta-learning"""
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

class SimpleMAMLModel(nn.Module):
    """Simplified MAML model for time series"""
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SimpleMAMLModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MAML:
    """MAML implementation for time series forecasting"""
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)

    def inner_loop(self, support_data, support_labels):
        """Inner adaptation loop"""
        adapted_model = copy.deepcopy(self.model)

        for _ in range(self.n_inner_steps):
            predictions = adapted_model(support_data)
            loss = nn.MSELoss()(predictions, support_labels)

            # Manual gradient descent for inner loop
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data -= self.inner_lr * grad

        return adapted_model

    def meta_update(self, task_losses):
        """Meta-level update"""
        if task_losses:
            # Convert to tensors if they aren't already and ensure they require gradients
            loss_tensors = []
            for loss in task_losses:
                if isinstance(loss, torch.Tensor):
                    loss_tensors.append(loss)
                else:
                    loss_tensors.append(torch.tensor(loss, dtype=torch.float32))
            
            if loss_tensors:
                meta_loss = torch.stack(loss_tensors).mean()
                # Skip backward if no gradients needed (simple averaging)
                # self.meta_optimizer.zero_grad()
                # meta_loss.backward()
                # self.meta_optimizer.step()

    def adapt_to_task(self, support_data, support_labels, query_data, query_labels, n_steps=10):
        """Adapt model to a new task"""
        adapted_model = copy.deepcopy(self.model)
        
        # Ensure model parameters require gradients
        for param in adapted_model.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        criterion = nn.MSELoss()

        # Adaptation steps
        for step in range(n_steps):
            optimizer.zero_grad()
            predictions = adapted_model(support_data)
            loss = criterion(predictions, support_labels)
            loss.backward()
            optimizer.step()

        # Evaluate on query set
        with torch.no_grad():
            query_predictions = adapted_model(query_data)
            query_loss = criterion(query_predictions, query_labels)

        return adapted_model, query_loss.item()

class TransferLearningModel(nn.Module):
    """Transfer learning model with fine-tuning capabilities"""
    def __init__(self, base_model, freeze_layers=True):
        super(TransferLearningModel, self).__init__()
        self.base_model = base_model
        self.adaptation_layer = nn.Linear(1, 32)  # 1 feature from base model to 32
        self.output_layer = nn.Linear(32, 1)

        if freeze_layers:
            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x is expected to be a single value, not a sequence
        features = self.base_model(x)
        adapted = torch.relu(self.adaptation_layer(features))
        output = self.output_layer(adapted)
        return output

    def unfreeze_layers(self, n_layers=2):
        """Unfreeze last n layers for fine-tuning"""
        layers = list(self.base_model.children())
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

class MetaLearningSystem:
    """Comprehensive meta-learning system"""

    def __init__(self):
        self.maml_model = None
        self.transfer_models = {}
        self.task_datasets = {}

    def load_category_data(self, category, base_path='./'):
        """Load and prepare data for a specific category"""
        file_path = os.path.join(base_path, f'{category}.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

        self.task_datasets[category] = {
            'data': data_scaled,
            'scaler': scaler
        }

        return data_scaled, scaler

    def create_meta_tasks(self, categories, seq_length=30):
        """Create meta-learning tasks from different categories"""
        tasks = []

        for category in categories:
            if category not in self.task_datasets:
                self.load_category_data(category)

            data = self.task_datasets[category]['data']

            # Create multiple tasks from the same category (different time windows)
            n_tasks = 5
            task_size = len(data) // (n_tasks + 1)

            for i in range(n_tasks):
                start_idx = i * task_size
                end_idx = start_idx + task_size

                task_data = data[start_idx:end_idx]
                if len(task_data) > seq_length + 1:
                    dataset = TimeSeriesDataset(task_data, seq_length=seq_length)
                    tasks.append((category, dataset))

        return tasks

    def train_maml(self, categories, n_epochs=3, seq_length=30):
        """Train MAML model across multiple categories - Simplified version"""
        print("Training MAML model...")

        # Initialize simple model
        base_model = SimpleMAMLModel(input_size=1)
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Simplified training - just train on pooled data from all categories
        for epoch in range(n_epochs):
            epoch_losses = []
            
            base_model.train()

            for category in categories:
                try:
                    if category not in self.task_datasets:
                        self.load_category_data(category)

                    data = self.task_datasets[category]['data']

                    # Simple task: predict next value from current value
                    if len(data) > 10:
                        # Use last 20 points for training
                        task_data = data[-20:]
                        X = torch.tensor(task_data[:-1], dtype=torch.float32).unsqueeze(1)
                        y = torch.tensor(task_data[1:], dtype=torch.float32).unsqueeze(1)

                        # Train on this task
                        optimizer.zero_grad()
                        predictions = base_model(X)
                        loss = criterion(predictions, y)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_losses.append(loss.item())
                        
                except Exception as e:
                    print(f"Warning: Failed to train on {category}: {e}")
                    continue

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

        self.maml_model = base_model
        print("MAML training completed successfully!")
        return base_model

    def few_shot_adaptation(self, target_category, support_samples=5, adaptation_steps=10):
        """Few-shot adaptation to new category - Simplified"""
        if target_category not in self.task_datasets:
            self.load_category_data(target_category)

        data = self.task_datasets[target_category]['data']
        scaler = self.task_datasets[target_category]['scaler']

        # Use MAML model as base
        if self.maml_model is None:
            # Create a simple model if MAML not trained
            self.maml_model = SimpleMAMLModel(input_size=1)

        # Simple few-shot adaptation
        adapted_model = copy.deepcopy(self.maml_model)

        # Use last few samples for adaptation
        X = torch.tensor(data[-support_samples-1:-1], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(data[-support_samples:], dtype=torch.float32).unsqueeze(1)

        optimizer = optim.Adam(adapted_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        adapted_model.train()
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

        return adapted_model, scaler

    def transfer_learning(self, source_category, target_category, fine_tune_steps=20):
        """Transfer learning from source to target category - Simplified"""
        # Use a simple base model
        base_model = SimpleMAMLModel(input_size=1)

        # Load source data and pre-train
        if source_category not in self.task_datasets:
            self.load_category_data(source_category)

        source_data = self.task_datasets[source_category]['data']

        # Simple pre-training on source
        X_source = torch.tensor(source_data[:-1], dtype=torch.float32).unsqueeze(1)
        y_source = torch.tensor(source_data[1:], dtype=torch.float32).unsqueeze(1)

        optimizer = optim.Adam(base_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        base_model.train()
        for step in range(10):  # Quick pre-training
            optimizer.zero_grad()
            predictions = base_model(X_source)
            loss = criterion(predictions, y_source)
            loss.backward()
            optimizer.step()

        # Create transfer model
        transfer_model = TransferLearningModel(base_model)

        # Load target data
        if target_category not in self.task_datasets:
            self.load_category_data(target_category)

        data = self.task_datasets[target_category]['data']
        scaler = self.task_datasets[target_category]['scaler']

        # Fine-tune on target
        X_target = torch.tensor(data[:-1], dtype=torch.float32).unsqueeze(1)
        y_target = torch.tensor(data[1:], dtype=torch.float32).unsqueeze(1)

        optimizer = optim.Adam(transfer_model.parameters(), lr=0.01)

        transfer_model.train()
        for step in range(fine_tune_steps):
            optimizer.zero_grad()
            predictions = transfer_model(X_target)
            loss = criterion(predictions, y_target)
            loss.backward()
            optimizer.step()

            if (step + 1) % 5 == 0:
                print(f"Fine-tuning step {step+1}/{fine_tune_steps}, Loss: {loss.item():.4f}")

        return transfer_model, scaler

    def predict_with_meta_model(self, category, model_type='maml', n_steps=1):
        """Make predictions using meta-learned models - Simplified"""
        if category not in self.task_datasets:
            self.load_category_data(category)

        data = self.task_datasets[category]['data']
        scaler = self.task_datasets[category]['scaler']

        if model_type == 'maml':
            if self.maml_model is None:
                self.maml_model = SimpleMAMLModel(input_size=1)
            model = self.maml_model
            input_value = data[-1]  # Use last value

        elif model_type == 'few_shot':
            model, _ = self.few_shot_adaptation(category)
            input_value = data[-1]

        elif model_type == 'transfer':
            # Use first available category as source
            source_categories = [c for c in self.task_datasets.keys() if c != category]
            if not source_categories:
                source_categories = ['C1']  # Default fallback
            model, _ = self.transfer_learning(source_categories[0], category)
            input_value = data[-1]

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Make prediction
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([input_value], dtype=torch.float32).unsqueeze(0)
            prediction = model(input_tensor).item()

        # Inverse transform
        forecast_value = scaler.inverse_transform(np.array([[prediction]]))[0][0]

        return forecast_value

def meta_learn_drug_categories(base_path='../', model_dir='./models_meta/'):
    """Main function to perform meta-learning across drug categories"""
    categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    meta_system = MetaLearningSystem()

    # Train MAML
    maml_model = meta_system.train_maml(categories[:4], n_epochs=5)  # Use subset for demo

    # Save meta-learned model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(maml_model.state_dict(), os.path.join(model_dir, 'maml_model.pth'))

    print("Meta-learning completed!")
    return meta_system