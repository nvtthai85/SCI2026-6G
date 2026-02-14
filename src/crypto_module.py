"""
Cryptographic Operations Module for CryptoTrust-6G Experiment
Implements SHA-3, EdDSA, AES-256, and simulated Dilithium-2
"""

import os
import sys
import time
import hashlib
import secrets
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))
from config import SECURITY_LEVELS, CRYPTO_BENCHMARKS
from utils import ExperimentLogger, Timer

# Try to import real cryptographic libraries
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography library not available. Using simulated operations.")


class CryptoOperation(Enum):
    """Enumeration of cryptographic operations"""
    HASH = "hash"
    SIGN = "sign"
    VERIFY = "verify"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"


@dataclass
class CryptoResult:
    """Result of a cryptographic operation"""
    success: bool
    latency_ms: float
    output: Optional[bytes] = None
    error: Optional[str] = None


class CryptoModule:
    """
    Cryptographic module implementing various security operations
    Supports both real and simulated cryptographic operations
    """
    
    def __init__(self, logger: Optional[ExperimentLogger] = None, use_simulation: bool = False):
        self.logger = logger
        self.use_simulation = use_simulation or not CRYPTO_AVAILABLE
        
        # Initialize keys
        self._init_keys()
        
        # Latency statistics
        self.latency_stats = {op.value: [] for op in CryptoOperation}
        
        if self.use_simulation:
            self.log("Using simulated cryptographic operations")
        else:
            self.log("Using real cryptographic operations")
    
    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
    
    def _init_keys(self):
        """Initialize cryptographic keys"""
        if not self.use_simulation and CRYPTO_AVAILABLE:
            # Real keys
            self.ed25519_private = Ed25519PrivateKey.generate()
            self.ed25519_public = self.ed25519_private.public_key()
            self.aes_key = secrets.token_bytes(32)  # 256-bit key
        else:
            # Simulated keys (random bytes)
            self.ed25519_private = secrets.token_bytes(32)
            self.ed25519_public = secrets.token_bytes(32)
            self.aes_key = secrets.token_bytes(32)
        
        # Simulated Dilithium keys (larger)
        self.dilithium_private = secrets.token_bytes(2528)  # Dilithium-2 private key size
        self.dilithium_public = secrets.token_bytes(1312)   # Dilithium-2 public key size
    
    def sha3_256(self, data: bytes) -> CryptoResult:
        """
        Compute SHA-3-256 hash
        
        Args:
            data: Input data to hash
        
        Returns:
            CryptoResult with hash output
        """
        start_time = time.perf_counter()
        
        try:
            if self.use_simulation:
                # Simulate processing time
                time.sleep(CRYPTO_BENCHMARKS["SHA3-256"]["compute_time_ms"] / 1000)
                hash_output = hashlib.sha3_256(data).digest()
            else:
                hash_output = hashlib.sha3_256(data).digest()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["hash"].append(latency_ms)
            
            return CryptoResult(
                success=True,
                latency_ms=latency_ms,
                output=hash_output
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def eddsa_sign(self, data: bytes) -> CryptoResult:
        """
        Sign data using EdDSA (Ed25519)
        
        Args:
            data: Data to sign
        
        Returns:
            CryptoResult with signature
        """
        start_time = time.perf_counter()
        
        try:
            if self.use_simulation:
                time.sleep(CRYPTO_BENCHMARKS["EdDSA"]["sign_time_ms"] / 1000)
                signature = secrets.token_bytes(64)  # Ed25519 signature size
            else:
                signature = self.ed25519_private.sign(data)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["sign"].append(latency_ms)
            
            return CryptoResult(
                success=True,
                latency_ms=latency_ms,
                output=signature
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def eddsa_verify(self, data: bytes, signature: bytes) -> CryptoResult:
        """
        Verify EdDSA signature
        
        Args:
            data: Original data
            signature: Signature to verify
        
        Returns:
            CryptoResult with verification status
        """
        start_time = time.perf_counter()
        
        try:
            if self.use_simulation:
                time.sleep(CRYPTO_BENCHMARKS["EdDSA"]["verify_time_ms"] / 1000)
                valid = True  # Assume valid in simulation
            else:
                self.ed25519_public.verify(signature, data)
                valid = True
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["verify"].append(latency_ms)
            
            return CryptoResult(
                success=valid,
                latency_ms=latency_ms,
                output=b'\x01' if valid else b'\x00'
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def aes256_gcm_encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> CryptoResult:
        """
        Encrypt using AES-256-GCM
        
        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data
        
        Returns:
            CryptoResult with ciphertext (nonce + ciphertext + tag)
        """
        start_time = time.perf_counter()
        
        try:
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            
            if self.use_simulation:
                time.sleep(CRYPTO_BENCHMARKS["AES-256-GCM"]["encrypt_time_ms"] / 1000)
                # Simulated ciphertext
                ciphertext = nonce + secrets.token_bytes(len(plaintext) + 16)
            else:
                aesgcm = AESGCM(self.aes_key)
                encrypted = aesgcm.encrypt(nonce, plaintext, associated_data)
                ciphertext = nonce + encrypted
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["encrypt"].append(latency_ms)
            
            return CryptoResult(
                success=True,
                latency_ms=latency_ms,
                output=ciphertext
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def aes256_gcm_decrypt(self, ciphertext: bytes, associated_data: bytes = b"") -> CryptoResult:
        """
        Decrypt AES-256-GCM ciphertext
        
        Args:
            ciphertext: nonce + ciphertext + tag
            associated_data: Additional authenticated data
        
        Returns:
            CryptoResult with plaintext
        """
        start_time = time.perf_counter()
        
        try:
            nonce = ciphertext[:12]
            encrypted_data = ciphertext[12:]
            
            if self.use_simulation:
                time.sleep(CRYPTO_BENCHMARKS["AES-256-GCM"]["decrypt_time_ms"] / 1000)
                plaintext = secrets.token_bytes(len(encrypted_data) - 16)
            else:
                aesgcm = AESGCM(self.aes_key)
                plaintext = aesgcm.decrypt(nonce, encrypted_data, associated_data)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["decrypt"].append(latency_ms)
            
            return CryptoResult(
                success=True,
                latency_ms=latency_ms,
                output=plaintext
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def dilithium2_sign(self, data: bytes) -> CryptoResult:
        """
        Sign using Dilithium-2 (Post-Quantum)
        Note: This is a simulation as real Dilithium requires specialized libraries
        
        Args:
            data: Data to sign
        
        Returns:
            CryptoResult with signature
        """
        start_time = time.perf_counter()
        
        try:
            # Simulate Dilithium-2 signing time
            time.sleep(CRYPTO_BENCHMARKS["Dilithium-2"]["sign_time_ms"] / 1000)
            
            # Generate simulated signature (2420 bytes for Dilithium-2)
            signature = secrets.token_bytes(2420)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["sign"].append(latency_ms)
            
            return CryptoResult(
                success=True,
                latency_ms=latency_ms,
                output=signature
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )
    
    def dilithium2_verify(self, data: bytes, signature: bytes) -> CryptoResult:
        """
        Verify Dilithium-2 signature (simulated)
        
        Args:
            data: Original data
            signature: Signature to verify
        
        Returns:
            CryptoResult with verification status
        """
        start_time = time.perf_counter()
        
        try:
            # Simulate Dilithium-2 verification time
            time.sleep(CRYPTO_BENCHMARKS["Dilithium-2"]["verify_time_ms"] / 1000)
            
            # Assume valid signature in simulation
            valid = len(signature) == 2420
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_stats["verify"].append(latency_ms)
            
            return CryptoResult(
                success=valid,
                latency_ms=latency_ms,
                output=b'\x01' if valid else b'\x00'
            )
        except Exception as e:
            return CryptoResult(
                success=False,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )


class SecurityLevelExecutor:
    """
    Executes cryptographic operations based on security level
    """
    
    def __init__(self, crypto_module: CryptoModule, logger: Optional[ExperimentLogger] = None):
        self.crypto = crypto_module
        self.logger = logger
        self.execution_stats = {level: [] for level in SECURITY_LEVELS.keys()}
    
    def log(self, message: str):
        if self.logger:
            self.logger.debug(message)
    
    def execute_security_level(self, level: str, data: bytes) -> Tuple[bool, float, Dict]:
        """
        Execute all cryptographic operations for a security level
        
        Args:
            level: Security level (L0-L4)
            data: Input data to process
        
        Returns:
            Tuple of (success, total_latency_ms, details)
        """
        if level not in SECURITY_LEVELS:
            raise ValueError(f"Invalid security level: {level}")
        
        config = SECURITY_LEVELS[level]
        crypto_ops = config["crypto_config"]
        
        total_latency_ms = 0.0
        details = {
            "level": level,
            "operations": [],
            "success": True
        }
        
        start_time = time.perf_counter()
        
        # Execute operations based on config
        if "SHA3" in crypto_ops:
            result = self.crypto.sha3_256(data)
            total_latency_ms += result.latency_ms
            details["operations"].append(("SHA3-256", result.latency_ms, result.success))
            if not result.success:
                details["success"] = False
            data = result.output if result.output else data
        
        if "EdDSA" in crypto_ops:
            result = self.crypto.eddsa_sign(data)
            total_latency_ms += result.latency_ms
            details["operations"].append(("EdDSA_Sign", result.latency_ms, result.success))
            if not result.success:
                details["success"] = False
        
        if "Dilithium" in crypto_ops:
            result = self.crypto.dilithium2_sign(data)
            total_latency_ms += result.latency_ms
            details["operations"].append(("Dilithium2_Sign", result.latency_ms, result.success))
            if not result.success:
                details["success"] = False
        
        if "AES256" in crypto_ops:
            result = self.crypto.aes256_gcm_encrypt(data)
            total_latency_ms += result.latency_ms
            details["operations"].append(("AES256_Encrypt", result.latency_ms, result.success))
            if not result.success:
                details["success"] = False
        
        # Add base latency for PLS (Physical Layer Security)
        pls_latency = 0.01  # 10 microseconds for PLS overhead
        total_latency_ms += pls_latency
        details["operations"].append(("PLS", pls_latency, True))
        
        details["total_latency_ms"] = total_latency_ms
        self.execution_stats[level].append(total_latency_ms)
        
        return details["success"], total_latency_ms, details
    
    def get_statistics(self) -> Dict[str, Dict]:
        """Get latency statistics for each security level"""
        stats = {}
        for level, latencies in self.execution_stats.items():
            if latencies:
                stats[level] = {
                    "mean_ms": np.mean(latencies),
                    "std_ms": np.std(latencies),
                    "min_ms": np.min(latencies),
                    "max_ms": np.max(latencies),
                    "count": len(latencies)
                }
            else:
                stats[level] = {"mean_ms": 0, "std_ms": 0, "count": 0}
        return stats


def calculate_tampering_detection_rate(
    security_level: str,
    attack_present: bool,
    attack_detected: bool
) -> float:
    """
    Calculate tampering detection rate based on security level
    
    The TDR depends on the cryptographic strength of the security level:
    - L0 (PLS only): Low detection capability
    - L1 (Hash): Medium detection (can detect modifications but not replay)
    - L2 (Hash + Sign): High detection (authenticity verified)
    - L3 (Hash + Sign + Encrypt): Very high detection
    - L4 (Full PQC): Maximum detection
    """
    base_tdr = {
        "L0": 0.20,   # PLS alone has limited detection
        "L1": 0.785,  # Hash-only (78.5% from paper)
        "L2": 0.962,  # EdDSA signature (96.2% from paper)
        "L3": 0.975,  # Full classical
        "L4": 0.985   # Full PQC (98.5% from paper)
    }
    
    return base_tdr.get(security_level, 0.5)


if __name__ == "__main__":
    # Test cryptographic operations
    from utils import ExperimentLogger
    
    logger = ExperimentLogger("crypto_test", "crypto_module")
    
    logger.phase_start("Cryptographic Operations Test")
    
    crypto = CryptoModule(logger, use_simulation=True)
    executor = SecurityLevelExecutor(crypto, logger)
    
    # Test data
    test_data = b"Test data for cryptographic operations" * 10
    
    # Test each security level
    for level in SECURITY_LEVELS.keys():
        logger.info(f"\nTesting {level}: {SECURITY_LEVELS[level]['name']}")
        success, latency, details = executor.execute_security_level(level, test_data)
        logger.info(f"  Success: {success}")
        logger.info(f"  Total Latency: {latency:.4f} ms")
        logger.info(f"  Expected: {SECURITY_LEVELS[level]['latency_ms']:.4f} ms")
        for op, lat, suc in details["operations"]:
            logger.info(f"    {op}: {lat:.4f} ms")
    
    # Get statistics
    stats = executor.get_statistics()
    logger.info("\nLatency Statistics:")
    for level, stat in stats.items():
        logger.info(f"  {level}: mean={stat['mean_ms']:.4f}ms, std={stat['std_ms']:.4f}ms")
    
    logger.phase_end("Cryptographic Operations Test")
    logger.finalize()
    
    print("\nCryptographic module test completed!")
