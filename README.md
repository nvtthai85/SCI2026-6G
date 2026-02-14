# A Hierarchical Risk-Adaptive Framework for Ensuring AI Trustworthiness in Quantum-Resistant 6G Networks

**A Hierarchical Risk-Adaptive Framework for Ensuring AI Trustworthiness in Quantum-Resistant 6G Networks**

## Overview

This repository contains the complete experimental implementation for the CryptoTrust-6G paper. The framework demonstrates an adaptive cryptographic approach for Edge AI security in 6G networks using PPO-based risk-adaptive control.

## Key Features

- **Three-layer Security Architecture**: Perception, Network, and Service layers
- **PPO-based Adaptive Controller**: Dynamic security level selection
- **CNN-LSTM Threat Classifier**: For anomaly detection and threat assessment
- **Comprehensive Cryptographic Operations**: SHA-3, EdDSA, AES-256, Dilithium-2 (simulated)
- **Complete Ablation Study**: Evaluates individual component contributions
- **SOTA Comparison**: Compares with EdgeGuard-IoT, AIDA6G, VeriLLM, iTrust6G

## Expected Results (from Paper)

| Configuration | Accuracy (%) | TDR (%) | Latency (ms) | URLLC (%) | F1-Score |
|---------------|-------------|---------|--------------|-----------|----------|
| Baseline | 99.1±0.2 | 0.0±0.0 | 0.02±0.01 | 100.0 | 0.991 |
| Hash-only | 97.8±0.3 | 78.5±1.2 | 0.05±0.01 | 100.0 | 0.976 |
| EdDSA | 96.8±0.4 | 96.2±0.8 | 0.40±0.03 | 100.0 | 0.965 |
| AES-256 | 95.7±0.5 | 84.7±1.5 | 0.35±0.02 | 100.0 | 0.954 |
| Full PQC | 95.3±0.4 | 98.5±0.3 | 2.50±0.15 | 0.0 | 0.951 |
| **Ours (Adaptive)** | **94.6±0.5** | **97.8±0.4** | **0.12±0.02** | **94.2** | **0.944** |

## Security Level Distribution

| Level | Time (%) | L_crypto (ms) | Trigger Risk |
|-------|----------|---------------|--------------|
| L0 | 42.5% | 0.02 | < 0.1 |
| L1 | 35.8% | 0.05 | 0.1 - 0.3 |
| L2 | 13.4% | 0.40 | 0.3 - 0.6 |
| L3 | 5.8% | 0.55 | 0.6 - 0.8 |
| L4 | 2.5% | 2.50 | ≥ 0.8 |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone or extract the repository
cd crypto_trust_6g

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
crypto_trust_6g/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── utils.py               # Logging and utilities
│   ├── data_loader.py         # Dataset loading (NSL-KDD, CICIDS2017, TON_IoT)
│   ├── crypto_module.py       # Cryptographic operations
│   ├── threat_classifier.py   # CNN-LSTM classifier
│   ├── ppo_controller.py      # PPO security controller
│   ├── ablation_study.py      # Ablation experiments
│   └── main.py                # Main experiment runner
├── data/                      # Datasets (auto-downloaded)
├── logs/                      # Experiment logs
├── results/                   # Results and outputs
├── models/                    # Saved models
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Usage

### Quick Start

```bash
# Run full experiment (quick mode - 3 runs)
python src/main.py --quick

# Run full experiment (10 runs for statistical significance)
python src/main.py --runs 10

# With custom experiment name
python src/main.py --name MyExperiment --runs 10
```

### Command Line Arguments

```
--name       Experiment name (default: CryptoTrust6G)
--runs       Number of runs for statistical significance (default: 10)
--synthetic  Use synthetic data for CICIDS2017/TON_IoT (default: True)
--device     Computing device: cpu or cuda (default: cpu)
--quick      Quick mode with fewer runs
```

### Running Individual Modules

```bash
# Test data loading
python src/data_loader.py

# Test cryptographic operations
python src/crypto_module.py

# Test threat classifier
python src/threat_classifier.py

# Test PPO controller
python src/ppo_controller.py

# Test ablation study
python src/ablation_study.py
```

## Datasets

### NSL-KDD (Auto-downloaded)
- Training: 125,973 samples
- Testing: 22,544 samples
- Features: 41
- Classes: 5 (Normal + 4 attack categories)

### CICIDS2017 (Synthetic/Manual)
- Total: 2,830,743 samples
- Features: 78
- Classes: 15

### TON_IoT (Synthetic/Manual)
- Total: 461,043 samples
- Features: 44
- Classes: 10 (Normal + 9 attack types)

For real CICIDS2017 and TON_IoT datasets, download from:
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets

## Output Files

After running the experiment, you will find:

```
logs/CryptoTrust6G/<timestamp>/
├── main.log                    # Complete experiment log
├── experiment_summary.json     # Summary statistics
├── ablation_results.json       # Ablation study results
├── final_results.json          # All results
└── ...
```

## PPO Controller Details

### State Space (12 dimensions)
1. Threat level estimate (0-1)
2. Packet loss rate
3. Latency jitter variance
4. Failed authentication attempts
5. Current security level (normalized)
6. Time since last level change
7. Bandwidth utilization
8. Queue length
9-12. Entropy-based traffic features

### Action Space
- L0: PLS only
- L1: SHA-3 Hash
- L2: SHA-3 + EdDSA
- L3: SHA-3 + EdDSA + AES-256
- L4: SHA-3 + Dilithium-2 + AES-256 (Full PQC)

### Reward Function
```
R(s,a) = α·TDR(a) - β·L_crypto(a)/L_max - γ·|a - a_{t-1}| - δ·𝟙[attack ∧ level < L3]
```
Where:
- α = 1.0 (TDR weight)
- β = 0.3 (Latency penalty)
- γ = 0.1 (Stability penalty)
- δ = 2.0 (Under-protection penalty)
- L_max = 1.0 ms

## Interpreting Results

### Log Files
Each phase produces detailed logs:
- Data loading statistics
- Classifier training progress
- PPO training convergence
- Ablation study results
- SOTA comparison

### JSON Results
All numerical results are saved in JSON format with:
- Mean values
- Standard deviations
- Per-run details

## Troubleshooting

### Common Issues

1. **PyTorch not found**: Install with `pip install torch`
2. **CUDA out of memory**: Use `--device cpu`
3. **Dataset download fails**: Check internet connection or download manually

### Performance Tips

- Use GPU if available: `--device cuda`
- For quick testing: `--quick`
- For production: `--runs 10` or more

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{cryptotrust6g2026,
  title={An Adaptive Cryptographic Framework for Ensuring Trustworthy 
         Artificial Intelligence in Ultra-Reliable Low-Latency 6G Networks},
  author={[Authors]},
  booktitle={[Conference]},
  year={2026}
}
```

## License

This research code is provided for academic purposes.

## Contact

For questions or issues, please open an issue in the repository.
