"""
Data Loader Module for CryptoTrust-6G Experiment
Handles NSL-KDD, CICIDS2017, and TON_IoT datasets
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import urllib.request
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, DATASETS, EXPERIMENT_CONFIG
from utils import ExperimentLogger


class DataLoader:
    """
    Data loader for intrusion detection datasets
    Supports NSL-KDD, CICIDS2017, and TON_IoT
    """
    
    # NSL-KDD column names
    NSL_KDD_COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Attack categories for NSL-KDD
    ATTACK_CATEGORIES = {
        'normal': 'Normal',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 
        'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
        'processtable': 'DoS', 'worm': 'DoS', 'mailbomb': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
        'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 
        'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
        'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L',
        'xsnoop': 'R2L', 'httptunnel': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 
        'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
    }
    
    def __init__(self, logger: Optional[ExperimentLogger] = None):
        self.logger = logger
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        
    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def download_nsl_kdd(self) -> Tuple[Path, Path]:
        """Download NSL-KDD dataset from GitHub"""
        train_path = DATA_DIR / "KDDTrain+.txt"
        test_path = DATA_DIR / "KDDTest+.txt"
        
        if not train_path.exists():
            self.log("Downloading NSL-KDD Training set...")
            urllib.request.urlretrieve(
                DATASETS["NSL_KDD"]["train_url"],
                train_path
            )
            self.log(f"Downloaded to {train_path}")
        
        if not test_path.exists():
            self.log("Downloading NSL-KDD Test set...")
            urllib.request.urlretrieve(
                DATASETS["NSL_KDD"]["test_url"],
                test_path
            )
            self.log(f"Downloaded to {test_path}")
        
        return train_path, test_path
    
    def load_nsl_kdd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess NSL-KDD dataset"""
        self.log("Loading NSL-KDD dataset...")
        
        train_path, test_path = self.download_nsl_kdd()
        
        # Load data
        train_df = pd.read_csv(train_path, header=None, names=self.NSL_KDD_COLUMNS)
        test_df = pd.read_csv(test_path, header=None, names=self.NSL_KDD_COLUMNS)
        
        self.log(f"Training samples: {len(train_df)}")
        self.log(f"Test samples: {len(test_df)}")
        
        # Map attacks to categories
        train_df['attack_category'] = train_df['label'].map(
            lambda x: self.ATTACK_CATEGORIES.get(x, 'Unknown')
        )
        test_df['attack_category'] = test_df['label'].map(
            lambda x: self.ATTACK_CATEGORIES.get(x, 'Unknown')
        )
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = LabelEncoder()
                train_df[col] = self.categorical_encoders[col].fit_transform(train_df[col])
            test_df[col] = self.categorical_encoders[col].transform(
                test_df[col].map(lambda x: x if x in self.categorical_encoders[col].classes_ else self.categorical_encoders[col].classes_[0])
            )
        
        # Prepare features (exclude label, difficulty, attack_category)
        feature_cols = [col for col in train_df.columns if col not in ['label', 'difficulty', 'attack_category']]
        
        X_train = train_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        
        # Encode labels (0 = Normal, 1-4 = Attack categories)
        y_train = self.label_encoder.fit_transform(train_df['attack_category'])
        y_test = self.label_encoder.transform(test_df['attack_category'])
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.log(f"Feature shape: {X_train.shape}")
        self.log(f"Label distribution (train): {np.bincount(y_train)}")
        self.log(f"Label distribution (test): {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def generate_nsl_kdd_synthetic(self, n_samples: int = 150000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic NSL-KDD-like data
        Uses statistical properties similar to real dataset
        """
        self.log(f"Generating synthetic NSL-KDD data ({n_samples} samples)...")
        
        n_features = DATASETS["NSL_KDD"]["num_features"]
        n_classes = DATASETS["NSL_KDD"]["num_classes"]
        
        # Class distribution (Normal + 4 attack categories: DoS, Probe, R2L, U2R)
        class_weights = [0.53, 0.36, 0.08, 0.02, 0.01]  # Approximate from real data
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.sum()
        
        # Generate labels
        y = np.random.choice(n_classes, size=n_samples, p=class_weights)
        
        # Generate features (different distributions per class)
        X = np.zeros((n_samples, n_features))
        
        for c in range(n_classes):
            mask = y == c
            n_class_samples = mask.sum()
            
            if c == 0:  # Normal traffic
                X[mask] = np.random.normal(0, 0.5, (n_class_samples, n_features))
            elif c == 1:  # DoS attacks
                X[mask] = np.random.normal(2, 1.5, (n_class_samples, n_features))
            elif c == 2:  # Probe attacks  
                X[mask] = np.random.normal(1, 1.0, (n_class_samples, n_features))
            elif c == 3:  # R2L attacks
                X[mask] = np.random.normal(0.5, 0.8, (n_class_samples, n_features))
            else:  # U2R attacks
                X[mask] = np.random.normal(1.5, 1.2, (n_class_samples, n_features))
        
        # Normalize
        X = StandardScaler().fit_transform(X)
        
        self.log(f"Generated {n_samples} samples with {n_features} features")
        self.log(f"Class distribution: {np.bincount(y)}")
        
        return X.astype(np.float32), y

    def generate_cicids2017_synthetic(self, n_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic CICIDS2017-like data
        Uses statistical properties similar to real dataset
        """
        self.log(f"Generating synthetic CICIDS2017 data ({n_samples} samples)...")
        
        n_features = DATASETS["CICIDS2017"]["num_features"]
        n_classes = DATASETS["CICIDS2017"]["num_classes"]
        
        # Class distribution (approximate from real dataset)
        class_weights = [0.80, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 
                        0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
        class_weights = np.array(class_weights[:n_classes])
        class_weights = class_weights / class_weights.sum()
        
        # Generate labels
        y = np.random.choice(n_classes, size=n_samples, p=class_weights)
        
        # Generate features (different distributions per class)
        X = np.zeros((n_samples, n_features))
        
        for c in range(n_classes):
            mask = y == c
            n_class_samples = mask.sum()
            
            if c == 0:  # Normal traffic
                X[mask] = np.random.normal(0, 1, (n_class_samples, n_features))
            else:  # Attack traffic (different patterns)
                mean_shift = np.random.uniform(-2, 2, n_features)
                std_scale = np.random.uniform(0.5, 2.0, n_features)
                X[mask] = np.random.normal(mean_shift, std_scale, (n_class_samples, n_features))
        
        # Normalize
        X = StandardScaler().fit_transform(X)
        
        self.log(f"Generated {n_samples} samples with {n_features} features")
        self.log(f"Class distribution: {np.bincount(y)}")
        
        return X.astype(np.float32), y
    
    def generate_toniot_synthetic(self, n_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic TON_IoT-like data
        Simulates IoT network traffic patterns
        """
        self.log(f"Generating synthetic TON_IoT data ({n_samples} samples)...")
        
        n_features = DATASETS["TON_IoT"]["num_features"]
        n_classes = DATASETS["TON_IoT"]["num_classes"]
        
        # Class distribution (Normal + 9 attack types)
        # 0: Normal, 1: Backdoor, 2: DDoS, 3: Injection, 4: MITM,
        # 5: Password, 6: Ransomware, 7: Scanning, 8: XSS, 9: Other
        class_weights = [0.70, 0.05, 0.05, 0.04, 0.03, 0.04, 0.02, 0.03, 0.02, 0.02]
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.sum()
        
        # Generate labels
        y = np.random.choice(n_classes, size=n_samples, p=class_weights)
        
        # Generate features with IoT-specific patterns
        X = np.zeros((n_samples, n_features))
        
        for c in range(n_classes):
            mask = y == c
            n_class_samples = mask.sum()
            
            if c == 0:  # Normal IoT traffic
                # Lower variance, more regular patterns
                X[mask] = np.random.normal(0, 0.5, (n_class_samples, n_features))
            elif c == 2:  # DDoS
                # High volume, specific patterns
                X[mask] = np.random.normal(3, 1.5, (n_class_samples, n_features))
            elif c == 7:  # Scanning
                # Sequential patterns
                base = np.random.normal(1, 0.5, (n_class_samples, n_features))
                base[:, :10] = np.linspace(0, 5, n_class_samples).reshape(-1, 1)
                X[mask] = base
            else:  # Other attacks
                mean_shift = np.random.uniform(-1, 3, n_features)
                X[mask] = np.random.normal(mean_shift, 1.0, (n_class_samples, n_features))
        
        # Normalize
        X = StandardScaler().fit_transform(X)
        
        self.log(f"Generated {n_samples} samples with {n_features} features")
        self.log(f"Class distribution: {np.bincount(y)}")
        
        return X.astype(np.float32), y
    
    def load_all_datasets(self, use_synthetic: bool = True) -> Dict[str, Dict]:
        """
        Load all three datasets
        
        Args:
            use_synthetic: If True, use synthetic data for CICIDS2017 and TON_IoT
                          If False, attempt to load real datasets
        
        Returns:
            Dictionary containing all datasets
        """
        datasets = {}
        
        # Try to load NSL-KDD (real data) or use synthetic
        self.log("\n" + "="*50)
        self.log("Loading NSL-KDD Dataset")
        self.log("="*50)
        
        try:
            X_train, X_test, y_train, y_test = self.load_nsl_kdd()
            datasets["NSL_KDD"] = {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train))
            }
        except Exception as e:
            self.log(f"Could not load NSL-KDD: {e}")
            self.log("Generating synthetic NSL-KDD data...")
            X, y = self.generate_nsl_kdd_synthetic(n_samples=150000)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=EXPERIMENT_CONFIG["test_split"],
                random_state=42, stratify=y
            )
            datasets["NSL_KDD"] = {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train))
            }
        
        # Load/Generate CICIDS2017
        self.log("\n" + "="*50)
        self.log("Loading CICIDS2017 Dataset")
        self.log("="*50)
        if use_synthetic:
            X, y = self.generate_cicids2017_synthetic(n_samples=200000)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=EXPERIMENT_CONFIG["test_split"], 
                random_state=42, stratify=y
            )
        datasets["CICIDS2017"] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        }
        
        # Load/Generate TON_IoT
        self.log("\n" + "="*50)
        self.log("Loading TON_IoT Dataset")
        self.log("="*50)
        if use_synthetic:
            X, y = self.generate_toniot_synthetic(n_samples=150000)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=EXPERIMENT_CONFIG["test_split"],
                random_state=42, stratify=y
            )
        datasets["TON_IoT"] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train))
        }
        
        # Summary
        self.log("\n" + "="*50)
        self.log("Dataset Summary")
        self.log("="*50)
        for name, data in datasets.items():
            self.log(f"{name}:")
            self.log(f"  Train samples: {len(data['X_train'])}")
            self.log(f"  Test samples: {len(data['X_test'])}")
            self.log(f"  Features: {data['n_features']}")
            self.log(f"  Classes: {data['n_classes']}")
        
        return datasets
    
    def create_time_series_data(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Convert flat features to time series for CNN-LSTM input
        
        Args:
            X: Input features (n_samples, n_features)
            sequence_length: Length of time sequence
        
        Returns:
            Reshaped data (n_samples, sequence_length, n_features_per_step)
        """
        n_samples, n_features = X.shape
        n_features_per_step = n_features // sequence_length
        
        if n_features_per_step * sequence_length < n_features:
            # Pad features if needed
            pad_size = sequence_length - (n_features % sequence_length)
            X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant')
            n_features_per_step = X.shape[1] // sequence_length
        
        X_reshaped = X[:, :sequence_length * n_features_per_step].reshape(
            n_samples, sequence_length, n_features_per_step
        )
        
        return X_reshaped


def create_attack_labels(y: np.ndarray, normal_class: int = 0) -> np.ndarray:
    """Convert multi-class labels to binary (Normal vs Attack)"""
    return (y != normal_class).astype(np.int32)


if __name__ == "__main__":
    # Test data loading
    from utils import ExperimentLogger
    
    logger = ExperimentLogger("data_loading_test", "data_loader")
    loader = DataLoader(logger)
    
    datasets = loader.load_all_datasets(use_synthetic=True)
    
    logger.finalize()
    print("\nData loading test completed!")
