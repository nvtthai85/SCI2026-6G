"""
CNN-LSTM Threat Classifier for CryptoTrust-6G
Used for both anomaly detection and threat level estimation for PPO controller
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASSIFIER_CONFIG, EXPERIMENT_CONFIG
from utils import ExperimentLogger, set_random_seed

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using NumPy-based classifier.")


@dataclass
class TrainingResult:
    """Result of model training"""
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    best_epoch: int
    best_val_acc: float


if TORCH_AVAILABLE:
    class CNNLSTMClassifier(nn.Module):
        """
        CNN-LSTM hybrid model for threat classification
        
        Architecture:
        - 3 CNN layers for spatial feature extraction
        - 2 LSTM layers for temporal dependencies
        - Fully connected layers for classification
        """
        
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            sequence_length: int = 10,
            cnn_filters: List[int] = None,
            lstm_hidden: int = 128,
            lstm_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.sequence_length = sequence_length
            
            cnn_filters = cnn_filters or CLASSIFIER_CONFIG["cnn_filters"]
            
            # Calculate features per timestep
            self.features_per_step = max(1, input_dim // sequence_length)
            
            # CNN layers
            self.conv1 = nn.Conv1d(self.features_per_step, cnn_filters[0], kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1)
            
            self.bn1 = nn.BatchNorm1d(cnn_filters[0])
            self.bn2 = nn.BatchNorm1d(cnn_filters[1])
            self.bn3 = nn.BatchNorm1d(cnn_filters[2])
            
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
            # Calculate LSTM input size after CNN
            cnn_output_length = sequence_length // 4  # After 2 pooling layers
            if cnn_output_length < 1:
                cnn_output_length = 1
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=cnn_filters[2],
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            # Fully connected layers
            self.fc1 = nn.Linear(lstm_hidden * 2, 64)  # *2 for bidirectional
            self.fc2 = nn.Linear(64, num_classes)
            
            # For threat level estimation (regression output)
            self.threat_head = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x: torch.Tensor, return_threat_level: bool = False) -> torch.Tensor:
            """
            Forward pass
            
            Args:
                x: Input tensor (batch, features) or (batch, seq_len, features_per_step)
                return_threat_level: If True, also return threat level estimate
            
            Returns:
                Classification logits and optionally threat level
            """
            batch_size = x.size(0)
            
            # Reshape if needed
            if len(x.shape) == 2:
                # Reshape (batch, features) to (batch, seq_len, features_per_step)
                total_features = x.size(1)
                seq_len = min(self.sequence_length, total_features)
                features_per_step = total_features // seq_len
                
                # Truncate or pad
                x = x[:, :seq_len * features_per_step]
                x = x.view(batch_size, seq_len, features_per_step)
            
            # CNN expects (batch, channels, length)
            x = x.permute(0, 2, 1)
            
            # CNN layers
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.dropout(x)
            
            # Prepare for LSTM (batch, seq_len, features)
            x = x.permute(0, 2, 1)
            
            # LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use last hidden state from both directions
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_combined = torch.cat([h_forward, h_backward], dim=1)
            
            # Fully connected
            x = self.relu(self.fc1(h_combined))
            x = self.dropout(x)
            
            # Classification output
            logits = self.fc2(x)
            
            if return_threat_level:
                threat_level = self.sigmoid(self.threat_head(self.relu(self.fc1(h_combined))))
                return logits, threat_level.squeeze()
            
            return logits
        
        def predict_threat_level(self, x: torch.Tensor) -> torch.Tensor:
            """Predict threat level (0-1) for input"""
            self.eval()
            with torch.no_grad():
                _, threat_level = self.forward(x, return_threat_level=True)
            return threat_level


class ThreatClassifierTrainer:
    """
    Trainer for CNN-LSTM threat classifier
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        logger: Optional[ExperimentLogger] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=CLASSIFIER_CONFIG["learning_rate"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        early_stopping_patience: int = None
    ) -> TrainingResult:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        
        Returns:
            TrainingResult with training history
        """
        epochs = epochs or CLASSIFIER_CONFIG["epochs"]
        patience = early_stopping_patience or CLASSIFIER_CONFIG["early_stopping_patience"]
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        
        self.log(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.log(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.log(f"Training completed. Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        return TrainingResult(
            train_loss=self.history["train_loss"],
            val_loss=self.history["val_loss"],
            train_acc=self.history["train_acc"],
            val_acc=self.history["val_acc"],
            best_epoch=best_epoch,
            best_val_acc=best_val_acc
        )
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = 100.0 * np.mean(all_preds == all_labels)
        
        # Per-class metrics
        num_classes = all_probs.shape[1]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        
        for c in range(num_classes):
            tp = np.sum((all_preds == c) & (all_labels == c))
            fp = np.sum((all_preds == c) & (all_labels != c))
            fn = np.sum((all_preds != c) & (all_labels == c))
            
            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        return {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist()
        }


# NumPy-based fallback classifier
class NumpyClassifier:
    """
    Simple NumPy-based classifier when PyTorch is not available
    Uses a basic neural network approach
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * 0.01
        self.b2 = np.zeros(num_classes)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def fit(self, X, y, epochs=100, lr=0.01, batch_size=128):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                probs = self.forward(X_batch)
                
                # Backward pass
                m = X_batch.shape[0]
                dz2 = probs.copy()
                dz2[np.arange(m), y_batch] -= 1
                dz2 /= m
                
                dW2 = self.a1.T @ dz2
                db2 = np.sum(dz2, axis=0)
                
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self.z1 > 0)
                
                dW1 = X_batch.T @ dz1
                db1 = np.sum(dz1, axis=0)
                
                # Update weights
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
        
        return self


def create_classifier(
    input_dim: int,
    num_classes: int,
    device: str = "cpu"
) -> Tuple[object, object]:
    """
    Create appropriate classifier based on available libraries
    
    Returns:
        Tuple of (model, trainer_class)
    """
    if TORCH_AVAILABLE:
        model = CNNLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
        return model, ThreatClassifierTrainer
    else:
        model = NumpyClassifier(input_dim, num_classes)
        return model, None


if __name__ == "__main__":
    # Test classifier
    from utils import ExperimentLogger
    
    logger = ExperimentLogger("classifier_test", "threat_classifier")
    
    logger.phase_start("Classifier Test")
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 41).astype(np.float32)
    y_train = np.random.randint(0, 5, 1000)
    X_val = np.random.randn(200, 41).astype(np.float32)
    y_val = np.random.randint(0, 5, 200)
    
    if TORCH_AVAILABLE:
        logger.info("Using PyTorch CNN-LSTM Classifier")
        
        model = CNNLSTMClassifier(input_dim=41, num_classes=5)
        trainer = ThreatClassifierTrainer(model, device="cpu", logger=logger)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Train for a few epochs
        result = trainer.train(train_loader, val_loader, epochs=10)
        logger.info(f"Best validation accuracy: {result.best_val_acc:.2f}%")
        
        # Evaluate
        metrics = trainer.evaluate(val_loader)
        logger.info(f"Test accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    else:
        logger.info("Using NumPy Classifier")
        model = NumpyClassifier(41, 5)
        model.fit(X_train, y_train, epochs=100)
        
        preds = model.predict(X_val)
        accuracy = 100.0 * np.mean(preds == y_val)
        logger.info(f"Validation accuracy: {accuracy:.2f}%")
    
    logger.phase_end("Classifier Test")
    logger.finalize()
    
    print("\nClassifier test completed!")
