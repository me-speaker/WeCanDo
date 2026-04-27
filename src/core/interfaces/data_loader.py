"""
数据导入接口
支持从CSV文件和数据库加载历史配方-性能数据
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from .base import BaseDataImporter


class CSVDataLoader(BaseDataImporter):
    """
    CSV数据加载器
    从CSV文件加载配方-性能数据
    """

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        skip_header: bool = False,
        decision_var_cols: Optional[List[int]] = None,
        objective_cols: Optional[List[int]] = None,
        col_names: Optional[List[str]] = None,
    ):
        """
        初始化CSV数据加载器

        Args:
            path: CSV文件路径
            delimiter: 分隔符，默认","
            skip_header: 是否跳过表头行
            decision_var_cols: 决策变量列索引，如 [0, 1, 2, 3]
            objective_cols: 目标变量列索引，如 [4, 5]
            col_names: 列名列表（当skip_header=True时使用）
        """
        self.path = Path(path)
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.decision_var_cols = decision_var_cols
        self.objective_cols = objective_cols
        self.col_names = col_names
        self._metadata: Dict[str, Any] = {}

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载CSV数据

        Returns:
            X: 决策变量数据, shape (n_samples, n_decision_vars)
            y: 目标变量数据, shape (n_samples, n_objectives)

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        if not self.path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.path}")

        # Read CSV
        if self.skip_header or self.col_names:
            df = pd.read_csv(
                self.path,
                delimiter=self.delimiter,
                header=0 if self.skip_header else None,
                names=self.col_names,
            )
        else:
            df = pd.read_csv(self.path, delimiter=self.delimiter, header=None)

        # Convert to numpy
        data = df.values.astype(float)

        # Split into decision variables and objectives
        if self.decision_var_cols is not None and self.objective_cols is not None:
            X = data[:, self.decision_var_cols]
            y = data[:, self.objective_cols]
        else:
            # Auto-detect: assume last columns are objectives
            n_obj = 1
            if self.objective_cols is not None:
                n_obj = len(self.objective_cols)
            elif self.col_names is not None:
                # Count non-numeric columns or assume last 1-2 columns
                n_obj = 1
            X = data[:, :-n_obj] if n_obj > 0 else data
            y = data[:, -n_obj:] if n_obj > 0 else np.zeros((len(data), 0))

        self._metadata = {
            "n_samples": len(X),
            "n_decision_vars": X.shape[1],
            "n_objectives": y.shape[1] if len(y.shape) > 1 else 1,
            "path": str(self.path),
            "col_names": self.col_names,
        }

        return X, y

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据元信息"""
        if not self._metadata:
            self.load()  # Trigger metadata extraction
        return self._metadata.copy()

    def preprocess(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理

        Args:
            X: 决策变量数据
            y: 目标变量数据

        Returns:
            预处理后的 X, y
        """
        # Remove NaN values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
            X, y = X[mask], y[mask]

        return X, y

    def load_with_bounds_filter(
        self, bounds: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据并根据决策变量边界过滤

        Args:
            bounds: 决策变量边界列表

        Returns:
            过滤后的 X, y
        """
        X, y = self.load()

        mask = np.ones(len(X), dtype=bool)
        for i, (low, high) in enumerate(bounds):
            mask &= (X[:, i] >= low) & (X[:, i] <= high)

        return X[mask], y[mask]


class DatabaseDataLoader(BaseDataImporter):
    """
    数据库数据加载器
    从SQLite数据库加载配方-性能数据
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        decision_var_cols: List[str],
        objective_cols: List[str],
        query: Optional[str] = None,
    ):
        """
        初始化数据库数据加载器

        Args:
            db_path: 数据库文件路径
            table_name: 表名
            decision_var_cols: 决策变量列名
            objective_cols: 目标变量列名
            query: 自定义SQL查询（可选）
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.decision_var_cols = decision_var_cols
        self.objective_cols = objective_cols
        self.query = query
        self._metadata: Dict[str, Any] = {}

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据库加载数据

        Returns:
            X: 决策变量数据, shape (n_samples, n_decision_vars)
            y: 目标变量数据, shape (n_samples, n_objectives)

        Raises:
            sqlite3.Error: 数据库错误
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            if self.query:
                cursor.execute(self.query)
            else:
                all_cols = self.decision_var_cols + self.objective_cols
                col_str = ", ".join(all_cols)
                cursor.execute(f"SELECT {col_str} FROM {self.table_name}")

            rows = cursor.fetchall()

            if not rows:
                raise ValueError(f"表 {self.table_name} 中没有数据")

            data = np.array(rows, dtype=float)
            X = data[:, : len(self.decision_var_cols)]
            y = data[:, len(self.decision_var_cols) :]

            self._metadata = {
                "n_samples": len(X),
                "n_decision_vars": X.shape[1],
                "n_objectives": y.shape[1] if len(y.shape) > 1 else 1,
                "table_name": self.table_name,
                "decision_var_cols": self.decision_var_cols,
                "objective_cols": self.objective_cols,
            }

        finally:
            conn.close()

        return X, y

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据元信息"""
        if not self._metadata:
            self.load()
        return self._metadata.copy()

    def load_with_filter(
        self, where_clause: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        带过滤条件加载数据

        Args:
            where_clause: SQL WHERE子句

        Returns:
            过滤后的 X, y
        """
        original_query = self.query
        all_cols = self.decision_var_cols + self.objective_cols
        col_str = ", ".join(all_cols)

        self.query = f"SELECT {col_str} FROM {self.table_name} WHERE {where_clause}"
        try:
            return self.load()
        finally:
            self.query = original_query
