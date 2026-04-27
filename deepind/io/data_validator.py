"""Data validation utilities."""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Result of a data validation check."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.is_valid = False
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class DataValidator:
    """Validator for DataFrames."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        nullable_columns: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
    ):
        """Initialize the validator.

        Args:
            required_columns: List of column names that must be present.
            nullable_columns: List of column names that can contain null values.
            dtypes: Dictionary mapping column names to expected dtypes.
            min_rows: Minimum number of rows required.
            max_rows: Maximum number of rows allowed.
        """
        self.required_columns = required_columns or []
        self.nullable_columns = nullable_columns or []
        self.dtypes = dtypes or {}
        self.min_rows = min_rows
        self.max_rows = max_rows

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with any errors or warnings.
        """
        result = ValidationResult(is_valid=True)

        self._validate_columns(df, result)
        self._validate_dtypes(df, result)
        self._validate_nulls(df, result)
        self._validate_row_count(df, result)
        self._validate_values(df, result)

        return result

    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate required columns are present."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            result.add_error(f"Missing required columns: {missing}")

    def _validate_dtypes(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate column dtypes."""
        for col, expected_dtype in self.dtypes.items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                if not self._is_dtype_compatible(actual_dtype, expected_dtype):
                    result.add_warning(
                        f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
                    )

    def _is_dtype_compatible(self, actual, expected) -> bool:
        """Check if dtypes are compatible."""
        if expected in (int, float, np.integer, np.floating):
            return np.issubdtype(actual, np.number)
        if expected in (str, object):
            return np.issubdtype(actual, np.object_) or np.issubdtype(actual, np.str_)
        return str(actual) == str(expected)

    def _validate_nulls(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate null values."""
        for col in df.columns:
            if col not in self.nullable_columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                result.add_error(
                    f"Column '{col}' contains {null_count} null values but is not nullable"
                )

    def _validate_row_count(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate row count constraints."""
        n_rows = len(df)
        if self.min_rows is not None and n_rows < self.min_rows:
            result.add_error(f"DataFrame has {n_rows} rows, minimum is {self.min_rows}")
        if self.max_rows is not None and n_rows > self.max_rows:
            result.add_warning(
                f"DataFrame has {n_rows} rows, maximum recommended is {self.max_rows}"
            )

    def _validate_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate data values."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().all():
                result.add_warning(f"Column '{col}' contains only null values")
            elif df[col].isin([np.inf, -np.inf]).any():
                result.add_error(f"Column '{col}' contains infinite values")
