"""Soft Sensing widget for DeepInd GUI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QGroupBox, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QMessageBox
)
from PyQt5.QtCore import Qt
import numpy as np

try:
    from src.sensing import SoftSensingPredictor
    from src.core.registry import SURROGATE_REGISTRY
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class SoftSensingWidget(QWidget):
    """Widget for soft sensing (soft sensor) functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.trained_model = None
        self.input_columns = []
        self.output_columns = []
        self.input_dim = 0
        self.output_dim = 0
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Header
        header = QLabel("软测量功能 - 基于训练好的代理模型进行预测")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)

        if not ENGINE_AVAILABLE:
            layout.addWidget(QLabel("引擎组件不可用，请确保已正确安装 DeepInd。"))
            self.setLayout(layout)
            return

        # Model selection
        model_group = QGroupBox("模型状态")
        model_layout = QFormLayout()

        self.model_status = QLabel("未加载模型")
        self.model_status.setStyleSheet("color: red; font-weight: bold;")
        model_layout.addRow("状态:", self.model_status)

        self.column_info = QLabel("")
        model_layout.addRow("数据列:", self.column_info)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Prediction inputs
        pred_group = QGroupBox("预测输入")
        pred_layout = QFormLayout()

        self.input_dim_label = QLabel("0")
        pred_layout.addRow("输入维度:", self.input_dim_label)

        self.input_values = QLineEdit()
        self.input_values.setPlaceholderText("用逗号分隔，例如: 0.5, 0.3, 0.2, 180, 85")
        pred_layout.addRow("输入值:", self.input_values)

        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)

        # Prediction button
        self.predict_btn = QPushButton("执行预测")
        self.predict_btn.clicked.connect(self.on_predict)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # Prediction output
        output_group = QGroupBox("预测结果")
        output_layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("预测结果将显示在这里...")
        output_layout.addWidget(self.result_text)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Usage hint
        hint = QLabel("提示: 请先在「配置」页面运行优化流程完成模型训练，训练好的模型将自动传递到此处。")
        hint.setStyleSheet("color: gray; font-size: 11px; padding: 5px;")
        layout.addWidget(hint)

        layout.addStretch()
        self.setLayout(layout)

    def on_load_model(self):
        """Load a trained model."""
        self.result_text.setText("请先运行优化流程完成模型训练。\n训练好的模型将自动传递到软测量组件。")

    def on_train_model(self):
        """Train a model with data."""
        self.result_text.setText("请先在「配置」页面配置模型参数，然后在「概览」页面点击「执行优化」完成训练。")

    def on_predict(self):
        """Execute prediction with proper normalization/inverse transform."""
        if self.trained_model is None:
            QMessageBox.warning(self, "警告", "模型未加载，请先运行优化流程训练模型")
            return

        try:
            # Parse input values
            input_str = self.input_values.text().strip()
            if not input_str:
                QMessageBox.warning(self, "警告", "请输入预测值")
                return

            values = [float(x.strip()) for x in input_str.split(",")]

            # Validate input dimension
            if len(values) != self.input_dim:
                QMessageBox.warning(
                    self, "警告",
                    f"输入维度不匹配！期望 {self.input_dim} 个值，但收到了 {len(values)} 个值"
                )
                return

            X = np.array(values).reshape(1, -1)

            # Normalize input if scalers are available
            if self.X_scaler_mean is not None and self.X_scaler_std is not None:
                X_normalized = (X - self.X_scaler_mean) / self.X_scaler_std
            else:
                X_normalized = X

            # Perform prediction
            result = self.trained_model.predict(X_normalized)

            # Inverse transform output if scalers are available
            if self.y_scaler_means is not None and self.y_scaler_stds is not None:
                result_normalized = result.flatten()
                result_original = []
                for i, (mean, std) in enumerate(zip(self.y_scaler_means, self.y_scaler_stds)):
                    if i < len(result_normalized):
                        result_original.append(result_normalized[i] * std + mean)
                result_flat = np.array(result_original)
            else:
                result_flat = result.flatten() if isinstance(result, np.ndarray) else np.array(result).flatten()

            # Format output
            output_lines = []
            output_lines.append(f"输入值 (原始): {values}")
            output_lines.append(f"预测维度: {len(result_flat)}")
            output_lines.append("")

            for i, val in enumerate(result_flat):
                col_name = self.output_columns[i] if i < len(self.output_columns) else f"输出{i+1}"
                output_lines.append(f"  {col_name}: {val:.4f}")

            self.result_text.setText("\n".join(output_lines))
            self.model_status.setText("预测完成")
            self.model_status.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            self.result_text.setText(f"预测失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"预测失败:\n{str(e)}")

    def set_trained_model(self, model, input_columns=None, output_columns=None,
                       X_scaler_mean=None, X_scaler_std=None,
                       y_scaler_means=None, y_scaler_stds=None):
        """Set a trained model for prediction.

        Args:
            model: Trained surrogate model (ELM, GP, NN, XGBoost)
            input_columns: List of input column names
            output_columns: List of output column names
            X_scaler_mean: Mean values for X normalization (for inverse transform)
            X_scaler_std: Std values for X normalization (for inverse transform)
            y_scaler_means: Mean values for y normalization (for inverse transform)
            y_scaler_stds: Std values for y normalization (for inverse transform)
        """
        self.trained_model = model
        self.predict_btn.setEnabled(True)
        self.model_status.setText("模型已加载")
        self.model_status.setStyleSheet("color: green; font-weight: bold;")

        # Store scalers for inverse transform
        self.X_scaler_mean = X_scaler_mean
        self.X_scaler_std = X_scaler_std
        self.y_scaler_means = y_scaler_means
        self.y_scaler_stds = y_scaler_stds

        if input_columns is not None and len(input_columns) > 0:
            self.input_columns = input_columns
            self.input_dim = len(input_columns)
            self.input_dim_label.setText(str(self.input_dim))

        if output_columns is not None and len(output_columns) > 0:
            self.output_columns = output_columns
            self.output_dim = len(output_columns)

        if input_columns and output_columns:
            self.column_info.setText(
                f"输入: {len(input_columns)} 列 ({', '.join(input_columns[:3])}{'...' if len(input_columns) > 3 else ''})\n"
                f"输出: {len(output_columns)} 列 ({', '.join(output_columns[:3])}{'...' if len(output_columns) > 3 else ''})"
            )
        elif input_columns:
            self.column_info.setText(f"输入: {len(input_columns)} 列")

        self.result_text.setText(
            f"模型加载成功！\n"
            f"输入维度: {self.input_dim}\n"
            f"输出维度: {self.output_dim}\n\n"
            f"请在下方输入预测值进行预测。"
        )

    def get_widget(self):
        """Return the widget itself."""
        return self