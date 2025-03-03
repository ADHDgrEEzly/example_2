import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .logger import setup_logger

logger = setup_logger(__name__)

class ExperimentTracker:
    """Track and manage machine learning experiments."""
    
    def __init__(self, experiments_dir: Optional[Path] = None):
        """
        Initialize the experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiments
        """
        self.experiments_dir = experiments_dir or Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(
        self,
        name: str,
        params: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Name of the experiment
            params: Dictionary of experiment parameters
            tags: Optional list of tags for the experiment
            
        Returns:
            str: Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        experiment = {
            "id": experiment_id,
            "name": name,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "params": params,
            "tags": tags or [],
            "metrics": {},
            "artifacts": []
        }
        
        # Save experiment metadata
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        self.current_experiment = experiment_id
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a metric for the current experiment.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.experiments_dir / self.current_experiment
        
        with open(exp_dir / "metadata.json", "r") as f:
            experiment = json.load(f)
        
        if name not in experiment["metrics"]:
            experiment["metrics"][name] = []
        
        metric_value = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        if step is not None:
            metric_value["step"] = step
        
        experiment["metrics"][name].append(metric_value)
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        logger.debug(f"Logged metric {name}={value}")
    
    def log_artifact(
        self,
        name: str,
        artifact_path: Path,
        description: Optional[str] = None
    ) -> None:
        """
        Log an artifact for the current experiment.
        
        Args:
            name: Artifact name
            artifact_path: Path to the artifact file
            description: Optional description
        """
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.experiments_dir / self.current_experiment
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Copy artifact to experiments directory
        import shutil
        dest_path = artifacts_dir / artifact_path.name
        shutil.copy2(artifact_path, dest_path)
        
        with open(exp_dir / "metadata.json", "r") as f:
            experiment = json.load(f)
        
        artifact = {
            "name": name,
            "path": str(dest_path.relative_to(exp_dir)),
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        experiment["artifacts"].append(artifact)
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        logger.info(f"Logged artifact: {name}")
    
    def end_experiment(
        self,
        status: str = "completed"
    ) -> None:
        """
        End the current experiment.
        
        Args:
            status: Final status of the experiment
        """
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        exp_dir = self.experiments_dir / self.current_experiment
        
        with open(exp_dir / "metadata.json", "r") as f:
            experiment = json.load(f)
        
        experiment["status"] = status
        experiment["end_time"] = datetime.now().isoformat()
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        logger.info(f"Ended experiment {self.current_experiment} with status: {status}")
        self.current_experiment = None
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment details.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            dict: Experiment metadata
        """
        exp_dir = self.experiments_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(exp_dir / "metadata.json", "r") as f:
            return json.load(f)
    
    def list_experiments(
        self,
        tag: Optional[str] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List all experiments.
        
        Args:
            tag: Filter by tag
            status: Filter by status
            
        Returns:
            pd.DataFrame: DataFrame with experiment details
        """
        experiments = []
        
        for exp_dir in self.experiments_dir.glob("**/metadata.json"):
            with open(exp_dir, "r") as f:
                exp = json.load(f)
                
                if tag and tag not in exp["tags"]:
                    continue
                if status and exp["status"] != status:
                    continue
                
                experiments.append({
                    "id": exp["id"],
                    "name": exp["name"],
                    "status": exp["status"],
                    "start_time": exp["start_time"],
                    "end_time": exp["end_time"],
                    "tags": ",".join(exp["tags"])
                })
        
        return pd.DataFrame(experiments)