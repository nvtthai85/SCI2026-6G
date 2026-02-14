"""
Configuration file for CryptoTrust-6G Experiment
An Adaptive Cryptographic Framework for Trustworthy AI in 6G Networks
"""

import os
from pathlib import Path

# ============= Project Paths =============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if not exist
for dir_path in [DATA_DIR, LOGS_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============= Dataset Configuration =============
DATASETS = {
    "NSL_KDD": {
        "train_url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "test_url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
        "num_features": 41,
        "num_classes": 5,  # Normal + 4 attack categories
        "train_samples": 125973,
        "test_samples": 22544
    },
    "CICIDS2017": {
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",  # Manual download required
        "num_features": 78,
        "num_classes": 15,
        "total_samples": 2830743
    },
    "TON_IoT": {
        "url": "https://research.unsw.edu.au/projects/toniot-datasets",  # Manual download required
        "num_features": 44,
        "num_classes": 10,  # Normal + 9 attack types
        "total_samples": 461043
    }
}

# ============= Security Levels Configuration =============
SECURITY_LEVELS = {
    "L0": {
        "name": "PLS Only",
        "crypto_config": [],
        "latency_ms": 0.02,
        "security_score": 0.0,
        "trigger_risk_min": 0.0,
        "trigger_risk_max": 0.1
    },
    "L1": {
        "name": "Hash Only (SHA-3)",
        "crypto_config": ["SHA3"],
        "latency_ms": 0.05,
        "security_score": 0.20,
        "trigger_risk_min": 0.1,
        "trigger_risk_max": 0.3
    },
    "L2": {
        "name": "SHA-3 + EdDSA",
        "crypto_config": ["SHA3", "EdDSA"],
        "latency_ms": 0.40,
        "security_score": 0.40,
        "trigger_risk_min": 0.3,
        "trigger_risk_max": 0.6
    },
    "L3": {
        "name": "SHA-3 + EdDSA + AES-256",
        "crypto_config": ["SHA3", "EdDSA", "AES256"],
        "latency_ms": 0.55,
        "security_score": 0.60,
        "trigger_risk_min": 0.6,
        "trigger_risk_max": 0.8
    },
    "L4": {
        "name": "Full PQC (SHA-3 + Dilithium-2 + AES-256)",
        "crypto_config": ["SHA3", "Dilithium", "AES256"],
        "latency_ms": 2.50,
        "security_score": 1.0,
        "trigger_risk_min": 0.8,
        "trigger_risk_max": 1.0
    }
}

# ============= PPO Controller Configuration =============
PPO_CONFIG = {
    "state_dim": 12,  # 12-dimensional state vector
    "action_dim": 5,  # 5 security levels
    "hidden_layers": [256, 128],
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_episodes": 500000,
    "convergence_threshold": 0.02,
    "batch_size": 64,
    "update_epochs": 10
}

# ============= Reward Function Weights =============
REWARD_WEIGHTS = {
    "alpha": 1.0,   # TDR weight
    "beta": 0.3,    # Latency penalty weight
    "gamma": 0.1,   # Level change penalty weight
    "delta": 2.0,   # Attack under low security penalty
    "L_max": 1.0    # Maximum acceptable latency (ms)
}

# ============= CNN-LSTM Classifier Configuration =============
CLASSIFIER_CONFIG = {
    "cnn_filters": [64, 128, 256],
    "kernel_size": 3,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 50,
    "early_stopping_patience": 5
}

# ============= Experiment Configuration =============
EXPERIMENT_CONFIG = {
    "num_runs": 10,  # For statistical significance
    "random_seeds": [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    "test_split": 0.2,
    "validation_split": 0.1,
    "urllc_threshold_ms": 1.0,  # URLLC compliance threshold
    "target_6g_latency_ms": 0.1  # 6G target latency
}

# ============= Expected Results (from paper) =============
EXPECTED_RESULTS = {
    "ablation": {
        "Baseline": {"accuracy": 99.1, "tdr": 0.0, "latency": 0.02, "urllc": 100.0, "f1": 0.991},
        "Hash-only": {"accuracy": 97.8, "tdr": 78.5, "latency": 0.05, "urllc": 100.0, "f1": 0.976},
        "EdDSA": {"accuracy": 96.8, "tdr": 96.2, "latency": 0.40, "urllc": 100.0, "f1": 0.965},
        "AES-256": {"accuracy": 95.7, "tdr": 84.7, "latency": 0.35, "urllc": 100.0, "f1": 0.954},
        "Full_PQC": {"accuracy": 95.3, "tdr": 98.5, "latency": 2.50, "urllc": 0.0, "f1": 0.951},
        "Adaptive": {"accuracy": 94.6, "tdr": 97.8, "latency": 0.12, "urllc": 94.2, "f1": 0.944}
    },
    "threat_distribution": {
        "L0": 42.5,
        "L1": 35.8,
        "L2": 13.4,
        "L3": 5.8,
        "L4": 2.5
    },
    "sota_comparison": {
        "EdgeGuard-IoT": {"tdr": 98.2, "latency": 2.10, "pq_safe": True, "adaptive": False},
        "AIDA6G": {"tdr": 89.5, "latency": 0.03, "pq_safe": "Partial", "adaptive": "Limited"},
        "VeriLLM": {"tdr": 92.8, "latency": 0.80, "pq_safe": False, "adaptive": False},
        "iTrust6G": {"tdr": 94.1, "latency": None, "pq_safe": "Partial", "adaptive": True},
        "Ours": {"tdr": 97.8, "latency_normal": 0.12, "latency_attack": 0.95, "pq_safe": True, "adaptive": True}
    }
}

# ============= Cryptographic Algorithm Benchmarks =============
CRYPTO_BENCHMARKS = {
    "SHA3-256": {"compute_time_ms": 0.015, "output_size_bytes": 32},
    "EdDSA": {"sign_time_ms": 0.180, "verify_time_ms": 0.200, "sig_size_bytes": 64},
    "AES-256-GCM": {"encrypt_time_ms": 0.008, "decrypt_time_ms": 0.008},
    "Dilithium-2": {"sign_time_ms": 1.200, "verify_time_ms": 0.500, "sig_size_bytes": 2420},
    "Kyber-768": {"encaps_time_ms": 0.150, "decaps_time_ms": 0.180}
}

# ============= Logging Configuration =============
LOG_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "level": "INFO"
}
