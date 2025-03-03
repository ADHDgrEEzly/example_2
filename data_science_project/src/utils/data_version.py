import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from .logger import setup_logger

logger = setup_logger(__name__)

class DataVersion:
    """Handle data versioning and tracking."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data versioning.
        
        Args:
            data_dir: Directory to store versioned data
        """
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.versions_file = self.data_dir / "versions.json"
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load versions metadata from file."""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions(self) -> None:
        """Save versions metadata to file."""
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash of DataFrame content.
        
        Args:
            df: Input DataFrame
            
        Returns:
            str: Hash of DataFrame content
        """
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values
        ).hexdigest()
    
    def save_version(
        self,
        df: pd.DataFrame,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a new version of the dataset.
        
        Args:
            df: DataFrame to save
            name: Name of the dataset
            description: Optional description
            tags: Optional list of tags
            
        Returns:
            str: Version ID
        """
        # Create version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{name}_{timestamp}"
        
        # Save data file
        data_path = self.data_dir / f"{version_id}.parquet"
        df.to_parquet(data_path)
        
        # Create version metadata
        version_info = {
            "id": version_id,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "tags": tags or [],
            "hash": self._compute_hash(df),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "file_path": str(data_path)
        }
        
        if name not in self.versions:
            self.versions[name] = []
        
        self.versions[name].append(version_info)
        self._save_versions()
        
        logger.info(f"Saved version: {version_id}")
        return version_id
    
    def load_version(
        self,
        version_id: Optional[str] = None,
        name: Optional[str] = None,
        tag: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load a specific version of the dataset.
        
        Args:
            version_id: Specific version ID to load
            name: Dataset name (loads latest version if version_id not provided)
            tag: Load latest version with this tag
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if version_id:
            # Find version by ID
            for versions in self.versions.values():
                for version in versions:
                    if version["id"] == version_id:
                        return pd.read_parquet(version["file_path"])
            raise ValueError(f"Version {version_id} not found")
        
        elif name:
            if name not in self.versions:
                raise ValueError(f"Dataset {name} not found")
            
            versions = self.versions[name]
            if tag:
                versions = [v for v in versions if tag in v["tags"]]
                if not versions:
                    raise ValueError(f"No versions found with tag {tag}")
            
            latest = sorted(
                versions,
                key=lambda x: x["timestamp"],
                reverse=True
            )[0]
            return pd.read_parquet(latest["file_path"])
        
        else:
            raise ValueError("Must provide either version_id or name")
    
    def list_versions(
        self,
        name: Optional[str] = None,
        tag: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List available versions.
        
        Args:
            name: Filter by dataset name
            tag: Filter by tag
            
        Returns:
            pd.DataFrame: DataFrame with version information
        """
        versions = []
        
        for dataset_name, dataset_versions in self.versions.items():
            if name and dataset_name != name:
                continue
            
            for version in dataset_versions:
                if tag and tag not in version["tags"]:
                    continue
                
                versions.append({
                    "id": version["id"],
                    "name": version["name"],
                    "timestamp": version["timestamp"],
                    "description": version["description"],
                    "tags": ",".join(version["tags"]),
                    "shape": f"{version['shape'][0]}x{version['shape'][1]}"
                })
        
        return pd.DataFrame(versions)
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            dict: Comparison results
        """
        df1 = self.load_version(version_id1)
        df2 = self.load_version(version_id2)
        
        comparison = {
            "shape_diff": {
                "rows": df2.shape[0] - df1.shape[0],
                "cols": df2.shape[1] - df1.shape[1]
            },
            "columns": {
                "added": list(set(df2.columns) - set(df1.columns)),
                "removed": list(set(df1.columns) - set(df2.columns))
            },
            "common_columns": list(set(df1.columns) & set(df2.columns))
        }
        
        # Compare statistics for common columns
        stats = {}
        for col in comparison["common_columns"]:
            if pd.api.types.is_numeric_dtype(df1[col]):
                stats[col] = {
                    "mean_diff": df2[col].mean() - df1[col].mean(),
                    "std_diff": df2[col].std() - df1[col].std(),
                    "null_diff": df2[col].isnull().sum() - df1[col].isnull().sum()
                }
        
        comparison["column_stats"] = stats
        return comparison