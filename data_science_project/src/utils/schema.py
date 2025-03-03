from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class FeatureSchema:
    """Schema definition for a single feature."""
    name: str
    dtype: str
    nullable: bool = False
    unique: bool = False
    range: Optional[tuple[float, float]] = None
    categories: Optional[List[str]] = None
    
    def validate(self, series: pd.Series) -> List[str]:
        """
        Validate a series against this schema.
        
        Args:
            series: pandas Series to validate
            
        Returns:
            list: List of validation error messages
        """
        errors = []
        
        # Check dtype
        if str(series.dtype) != self.dtype:
            errors.append(
                f"Feature '{self.name}' has wrong dtype: "
                f"expected {self.dtype}, got {series.dtype}"
            )
        
        # Check nullability
        if not self.nullable and series.isnull().any():
            errors.append(
                f"Feature '{self.name}' contains null values "
                f"but is not nullable"
            )
        
        # Check uniqueness
        if self.unique and series.nunique() != len(series):
            errors.append(
                f"Feature '{self.name}' should contain unique values"
            )
        
        # Check range for numerical features
        if self.range is not None:
            min_val, max_val = self.range
            if series.min() < min_val or series.max() > max_val:
                errors.append(
                    f"Feature '{self.name}' has values outside "
                    f"range [{min_val}, {max_val}]"
                )
        
        # Check categories for categorical features
        if self.categories is not None:
            invalid_cats = set(series.unique()) - set(self.categories)
            if invalid_cats:
                errors.append(
                    f"Feature '{self.name}' contains invalid "
                    f"categories: {invalid_cats}"
                )
        
        return errors

class DataSchema:
    """Schema definition for the entire dataset."""
    
    def __init__(self, features: Dict[str, FeatureSchema]):
        """
        Initialize schema with feature definitions.
        
        Args:
            features: Dictionary mapping feature names to their schemas
        """
        self.features = features
    
    def validate(self, df: pd.DataFrame) -> List[str]:
        """
        Validate a DataFrame against this schema.
        
        Args:
            df: pandas DataFrame to validate
            
        Returns:
            list: List of validation error messages
        """
        errors = []
        
        # Check for missing columns
        missing_cols = set(self.features.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(self.features.keys())
        if extra_cols:
            errors.append(f"Found unexpected columns: {extra_cols}")
        
        # Validate each feature
        for name, schema in self.features.items():
            if name in df.columns:
                errors.extend(schema.validate(df[name]))
        
        return errors

# Example schema definition
example_schema = DataSchema({
    "age": FeatureSchema(
        name="age",
        dtype="int64",
        nullable=False,
        range=(0, 120)
    ),
    "income": FeatureSchema(
        name="income",
        dtype="float64",
        nullable=True,
        range=(0, float("inf"))
    ),
    "category": FeatureSchema(
        name="category",
        dtype="object",
        nullable=False,
        categories=["A", "B", "C"]
    )
})