"""
Utilities and Logging Module for CryptoTrust-6G Experiment
"""

import os
import sys
import logging
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import LOGS_DIR, RESULTS_DIR, LOG_CONFIG


class ExperimentLogger:
    """
    Comprehensive logger for experiment phases
    Outputs to both console and file
    """
    
    def __init__(self, experiment_name: str, phase: str = "main"):
        self.experiment_name = experiment_name
        self.phase = phase
        self.start_time = datetime.now()
        
        # Create log directory for this experiment
        self.log_dir = LOGS_DIR / experiment_name / self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"{experiment_name}_{phase}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        log_file = self.log_dir / f"{phase}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_CONFIG["format"], LOG_CONFIG["date_format"])
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {}
        self.phase_times = {}
        
        self.info(f"{'='*60}")
        self.info(f"Experiment: {experiment_name}")
        self.info(f"Phase: {phase}")
        self.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Log Directory: {self.log_dir}")
        self.info(f"{'='*60}")
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def phase_start(self, phase_name: str):
        """Mark the start of a phase"""
        self.phase_times[phase_name] = {"start": time.time()}
        self.info(f"\n{'='*40}")
        self.info(f"PHASE START: {phase_name}")
        self.info(f"{'='*40}")
    
    def phase_end(self, phase_name: str, metrics: Optional[Dict] = None):
        """Mark the end of a phase"""
        if phase_name in self.phase_times:
            self.phase_times[phase_name]["end"] = time.time()
            duration = self.phase_times[phase_name]["end"] - self.phase_times[phase_name]["start"]
            self.phase_times[phase_name]["duration"] = duration
        else:
            duration = 0
        
        self.info(f"\n{'-'*40}")
        self.info(f"PHASE END: {phase_name}")
        self.info(f"Duration: {duration:.2f} seconds")
        
        if metrics:
            self.metrics[phase_name] = metrics
            self.info("Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.info(f"  {key}: {value:.4f}")
                else:
                    self.info(f"  {key}: {value}")
        
        self.info(f"{'-'*40}\n")
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log a dictionary of metrics"""
        for key, value in metrics.items():
            metric_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, (int, float)):
                self.info(f"{metric_name}: {value:.4f}" if isinstance(value, float) else f"{metric_name}: {value}")
            elif isinstance(value, np.ndarray):
                self.info(f"{metric_name}: mean={np.mean(value):.4f}, std={np.std(value):.4f}")
            else:
                self.info(f"{metric_name}: {value}")
    
    def log_table(self, headers: list, rows: list, title: str = ""):
        """Log a formatted table"""
        if title:
            self.info(f"\n{title}")
            self.info("=" * len(title))
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)
        
        self.info(header_line)
        self.info(separator)
        
        # Format rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            self.info(row_line)
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save results to JSON file"""
        results_file = self.log_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.info(f"Results saved to: {results_file}")
    
    def save_model(self, model, filename: str):
        """Save model to pickle file"""
        model_file = self.log_dir / filename
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        self.info(f"Model saved to: {model_file}")
    
    def finalize(self):
        """Finalize logging and save summary"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        self.info(f"\n{'='*60}")
        self.info("EXPERIMENT COMPLETED")
        self.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        self.info(f"{'='*60}")
        
        # Save phase timings
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration,
            "phase_times": self.phase_times,
            "metrics": self.metrics
        }
        
        self.save_results(summary, "experiment_summary.json")


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def calculate_statistics(values: list) -> Dict[str, float]:
    """Calculate mean and std for a list of values"""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }


def format_metric_with_std(mean: float, std: float, decimals: int = 2) -> str:
    """Format metric as mean±std"""
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "", logger: Optional[ExperimentLogger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if self.logger and self.name:
            self.logger.debug(f"Timer [{self.name}]: {self.duration*1000:.4f} ms")
    
    def get_duration_ms(self) -> float:
        return self.duration * 1000 if self.duration else 0


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    # Test logging
    logger = ExperimentLogger("test_experiment", "test_phase")
    logger.info("Testing logger...")
    logger.phase_start("Test Phase 1")
    time.sleep(0.5)
    logger.phase_end("Test Phase 1", {"accuracy": 0.95, "loss": 0.05})
    logger.finalize()
    print("Logger test completed!")
