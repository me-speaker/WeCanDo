"""Data Loading Skill.

Reusable skill for loading chemical formulation data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class DataLoaderSkill:
    """Skill for loading formulation data."""

    @staticmethod
    def load_csv(path: str, delimiter: str = ",", skip_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file.
        
        Args:
            path: Path to CSV file
            delimiter: CSV delimiter
            skip_header: Whether to skip header row
            
        Returns:
            X: Decision variables
            y: Objective values
        """
        if skip_header:
            data = pd.read_csv(path, delimiter=delimiter)
        else:
            data = pd.read_csv(path, delimiter=delimiter, header=None)
        
        # Last columns are objectives
        n_objectives = 2  # Default
        X = data.iloc[:, :-n_objectives].values
        y = data.iloc[:, -n_objectives:].values
        return X, y

    @staticmethod
    def validate_data(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate loaded data.
        
        Returns:
            Validation result with 'valid', 'issues' keys
        """
        issues = []
        if X.shape[0] != y.shape[0]:
            issues.append("Sample count mismatch between X and y")
        if np.isnan(X).any():
            issues.append("NaN values in X")
        if np.isnan(y).any():
            issues.append("NaN values in y")
        return {"valid": len(issues) == 0, "issues": issues}
