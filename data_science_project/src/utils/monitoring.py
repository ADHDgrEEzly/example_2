import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import numpy as np
import pandas as pd

from .logger import setup_logger

logger = setup_logger(__name__)

class PerformanceMonitor:
    """Monitor and log model performance metrics."""
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        """
        Initialize the performance monitor.
        
        Args:
            metrics_dir: Directory to store metrics. If None, uses 'metrics'
                       in the project root
        """
        self.metrics_dir = metrics_dir or Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        model_version: str
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            model_name: Name of the model
            model_version: Version of the model
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare metrics record
        record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_version": model_version,
            "metrics": metrics
        }
        
        # Save to JSON file
        metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(record)
        
        with open(metrics_file, "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Logged metrics for {model_name} version {model_version}")
    
    def get_metrics_history(
        self,
        model_name: str,
        metric_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical metrics for a model.
        
        Args:
            model_name: Name of the model
            metric_name: Optional specific metric to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with metrics history
        """
        metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
        
        if not metrics_file.exists():
            return pd.DataFrame()
        
        with open(metrics_file, "r") as f:
            history = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if metric_name:
            df[metric_name] = df["metrics"].apply(lambda x: x.get(metric_name))
            df = df.drop("metrics", axis=1)
        
        return df
    
    def detect_drift(
        self,
        model_name: str,
        metric_name: str,
        threshold: float = 0.1,
        window: int = 5
    ) -> bool:
        """
        Detect if there's significant drift in a metric.
        
        Args:
            model_name: Name of the model
            metric_name: Metric to monitor
            threshold: Relative change threshold to consider as drift
            window: Number of recent records to consider
            
        Returns:
            bool: True if drift detected, False otherwise
        """
        df = self.get_metrics_history(model_name, metric_name)
        
        if len(df) < window:
            return False
        
        recent = df[metric_name].tail(window)
        baseline = df[metric_name].iloc[:-window].mean()
        
        if baseline == 0:
            return False
        
        relative_change = abs(recent.mean() - baseline) / baseline
        
        return relative_change > threshold

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        wrapper: Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            f"Function {func.__name__} took "
            f"{end_time - start_time:.2f} seconds"
        )
        return result
    
    return wrapper