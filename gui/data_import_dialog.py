"""Data Import Dialog for column selection and data preview."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QAbstractItemView,
    QGroupBox, QCheckBox, QSpinBox, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np


class DataImportDialog(QDialog):
    """
    Dialog for data import - column selection and data preview.

    Allows user to:
    - Preview first 100 rows of data
    - Select input feature columns (X)
    - Select output target columns (y)
    """

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df
        self.input_cols = []
        self.output_cols = []
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("数据导入 - 列选择")
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout()

        # Data summary section
        summary_layout = QHBoxLayout()
        summary_layout.addWidget(QLabel(f"数据形状: {self.df.shape[0]} 行 × {self.df.shape[1]} 列"))
        summary_layout.addWidget(QLabel(f"列名: {', '.join(self.df.columns.tolist())}"))
        layout.addLayout(summary_layout)

        # Column selection section
        selection_layout = QHBoxLayout()

        # Input columns selection (left)
        input_group = QGroupBox("输入特征列 (X) - 决策变量")
        input_layout = QVBoxLayout()

        self.input_list = QListWidget()
        self.input_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self.input_list.addItem(item)
        input_layout.addWidget(self.input_list)

        # Quick selection buttons for input
        input_btn_layout = QHBoxLayout()
        btn_select_all_inputs = QPushButton("全选")
        btn_select_all_inputs.clicked.connect(lambda: self._select_all(self.input_list, True))
        btn_clear_inputs = QPushButton("清空")
        btn_clear_inputs.clicked.connect(lambda: self._select_all(self.input_list, False))
        input_btn_layout.addWidget(btn_select_all_inputs)
        input_btn_layout.addWidget(btn_clear_inputs)
        input_layout.addLayout(input_btn_layout)

        input_group.setLayout(input_layout)
        selection_layout.addWidget(input_group)

        # Output columns selection (right)
        output_group = QGroupBox("输出目标列 (y) - 优化目标")
        output_layout = QVBoxLayout()

        self.output_list = QListWidget()
        self.output_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self.output_list.addItem(item)
        output_layout.addWidget(self.output_list)

        # Quick selection buttons for output
        output_btn_layout = QHBoxLayout()
        btn_select_all_outputs = QPushButton("全选")
        btn_select_all_outputs.clicked.connect(lambda: self._select_all(self.output_list, True))
        btn_clear_outputs = QPushButton("清空")
        btn_clear_outputs.clicked.connect(lambda: self._select_all(self.output_list, False))
        output_btn_layout.addWidget(btn_select_all_outputs)
        output_btn_layout.addWidget(btn_clear_outputs)
        output_layout.addLayout(output_btn_layout)

        output_group.setLayout(output_layout)
        selection_layout.addWidget(output_group)

        layout.addLayout(selection_layout)

        # Data preview section
        preview_label = QLabel("数据预览 (前100行):")
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        self._populate_preview()
        layout.addWidget(self.preview_table)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        btn_confirm = QPushButton("确认")
        btn_confirm.setDefault(True)
        btn_confirm.clicked.connect(self.on_confirm)
        button_layout.addWidget(btn_confirm)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _select_all(self, list_widget, checked):
        """Select or deselect all items in a list widget."""
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _populate_preview(self):
        """Populate the preview table with data."""
        preview_df = self.df.head(100)

        self.preview_table.setRowCount(preview_df.shape[0])
        self.preview_table.setColumnCount(preview_df.shape[1])
        self.preview_table.setHorizontalHeaderLabels(preview_df.columns.tolist())

        for i in range(preview_df.shape[0]):
            for j in range(preview_df.shape[1]):
                value = preview_df.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                else:
                    item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

        # Resize columns to fit content
        self.preview_table.resizeColumnsToContents()

    def on_confirm(self):
        """Handle confirm button click."""
        # Get selected input columns
        self.input_cols = []
        for i in range(self.input_list.count()):
            item = self.input_list.item(i)
            if item.checkState() == Qt.Checked:
                self.input_cols.append(item.text())

        # Get selected output columns
        self.output_cols = []
        for i in range(self.output_list.count()):
            item = self.output_list.item(i)
            if item.checkState() == Qt.Checked:
                self.output_cols.append(item.text())

        # Validate selection
        if not self.input_cols:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请至少选择一个输入特征列")
            return

        if not self.output_cols:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请至少选择一个输出目标列")
            return

        self.accept()

    def get_column_selection(self):
        """Get the selected input and output columns.

        Returns:
            tuple: (input_cols list, output_cols list)
        """
        return self.input_cols, self.output_cols

    def get_data_summary(self):
        """Get a string summary of the data for logging.

        Returns:
            str: Data analysis summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("【数据导入分析】")
        lines.append("=" * 60)
        lines.append(f"数据形状: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        # Column info
        lines.append("\n列信息:")
        lines.append("-" * 40)
        lines.append(f"{'列名':<30} {'类型':<10} {'非空数量':<10} {'缺失率':<10}")
        lines.append("-" * 40)

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            n_valid = self.df[col].notna().sum()
            n_missing = self.df[col].isna().sum()
            missing_pct = n_missing / len(self.df) * 100
            lines.append(f"{col:<30} {dtype:<10} {n_valid:<10} {missing_pct:<10.1f}%")

        # Numeric columns stats
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            lines.append("\n数值列统计:")
            lines.append("-" * 40)
            lines.append(f"{'列名':<30} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
            lines.append("-" * 40)

            for col in numeric_cols:
                stats = self.df[col].describe()
                lines.append(
                    f"{col:<30} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                    f"{stats['min']:<12.4f} {stats['max']:<12.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def analyze_data(df: pd.DataFrame) -> str:
    """
    Analyze a DataFrame and return a summary string.

    Args:
        df: DataFrame to analyze

    Returns:
        str: Analysis summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("【数据宏观分析】")
    lines.append("=" * 60)
    lines.append(f"样本数: {len(df)}")
    lines.append(f"特征数: {len(df.columns)}")
    lines.append(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Column details
    lines.append("\n列详情:")
    lines.append("-" * 60)

    for col in df.columns:
        col_type = df[col].dtype
        n_valid = df[col].notna().sum()
        n_missing = df[col].isna().sum()
        missing_pct = n_missing / len(df) * 100

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            lines.append(f"  {col}:")
            lines.append(f"    类型: {col_type}, 非空: {n_valid} ({100-missing_pct:.1f}%)")
            lines.append(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}], 均值: {stats['mean']:.4f}")
        else:
            n_unique = df[col].nunique()
            lines.append(f"  {col}:")
            lines.append(f"    类型: {col_type}, 非空: {n_valid} ({100-missing_pct:.1f}%), 唯一值: {n_unique}")

    # Correlation hint for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        lines.append("\n数值列相关性矩阵 (前5列):")
        corr = numeric_df.corr()
        lines.append("-" * 40)
        for i, col in enumerate(corr.columns[:5]):
            line = f"  {col:<20}"
            for j, val in enumerate(corr[col].values[:5]):
                line += f" {val:>8.3f}"
            lines.append(line)

    lines.append("=" * 60)
    return "\n".join(lines)