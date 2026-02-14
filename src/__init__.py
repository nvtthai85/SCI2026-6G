"""
CryptoTrust-6G: Adaptive Cryptographic Framework for Trustworthy AI in 6G Networks

This package implements the complete experimental framework including:
- Data loading and preprocessing
- CNN-LSTM threat classifier
- PPO-based risk-adaptive security controller
- Cryptographic operations (SHA-3, EdDSA, AES-256, Dilithium-2)
- Ablation study and SOTA comparison
"""

__version__ = "1.0.0"
__author__ = "[Authors]"

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
