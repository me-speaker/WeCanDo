"""Configuration widget for DeepInd GUI."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QPushButton, QFormLayout, QScrollArea,
    QMessageBox, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt


class ConfigWidget(QWidget):
    """Widget for configuring model and optimizer parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {
            'model_type': 'NeuralNetwork',
            'model_params': {},
            'optimizer': 'AutoOptimizer',
            'optimizer_params': {},
            'input_dim': 10,
        }
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create content widget
        content = QWidget()
        content_layout = QVBoxLayout()

        # Data configuration group
        data_group = self.create_data_group()
        content_layout.addWidget(data_group)

        # Model configuration group
        model_group = self.create_model_group()
        content_layout.addWidget(model_group)

        # Optimizer configuration group
        optimizer_group = self.create_optimizer_group()
        content_layout.addWidget(optimizer_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.on_apply)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        button_layout.addWidget(self.btn_apply)
        button_layout.addWidget(self.btn_reset)
        content_layout.addLayout(button_layout)

        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        self.setLayout(layout)

    def create_data_group(self):
        """Create data configuration group."""
        group = QGroupBox("数据配置")
        layout = QFormLayout()

        # Input dimension
        self.input_dim_spin = QSpinBox()
        self.input_dim_spin.setRange(1, 10000)
        self.input_dim_spin.setValue(10)
        self.input_dim_spin.setObjectName("input_dim")
        layout.addRow("输入维度:", self.input_dim_spin)

        # Output dimension (for multi-objective optimization)
        self.output_dim_spin = QSpinBox()
        self.output_dim_spin.setRange(1, 1000)
        self.output_dim_spin.setValue(1)
        self.output_dim_spin.setObjectName("output_dim")
        layout.addRow("输出维度:", self.output_dim_spin)

        # Column info labels
        self.input_cols_label = QLabel("未选择")
        self.input_cols_label.setStyleSheet("color: gray;")
        layout.addRow("输入列:", self.input_cols_label)

        self.output_cols_label = QLabel("未选择")
        self.output_cols_label.setStyleSheet("color: gray;")
        layout.addRow("输出列:", self.output_cols_label)

        # Data status
        self.data_status_label = QLabel("未导入数据")
        self.data_status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addRow("数据状态:", self.data_status_label)

        group.setLayout(layout)
        return group

    def create_model_group(self):
        """Create model configuration group."""
        group = QGroupBox("代理模型配置")
        layout = QFormLayout()

        # Model type selector
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['ELM', 'GaussianProcess', 'NeuralNetwork', 'XGBoost'])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        layout.addRow("模型类型:", self.model_type_combo)

        # Model parameters based on type
        self.model_params_widget = QWidget()
        self.model_params_layout = QFormLayout()
        self.model_params_widget.setLayout(self.model_params_layout)
        layout.addRow("模型参数:", self.model_params_widget)

        # Initialize with default parameters
        self.update_model_params('NeuralNetwork')

        group.setLayout(layout)
        return group

    def create_optimizer_group(self):
        """Create optimizer configuration group."""
        group = QGroupBox("优化器配置")
        layout = QFormLayout()

        # Optimizer type selector
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['AutoOptimizer', 'LBFGS', 'CG', 'Bayesian', 'Genetic'])
        self.optimizer_combo.currentTextChanged.connect(self.on_optimizer_changed)
        layout.addRow("优化器:", self.optimizer_combo)

        # Optimizer parameters
        self.opt_params_widget = QWidget()
        self.opt_params_layout = QFormLayout()
        self.opt_params_widget.setLayout(self.opt_params_layout)
        layout.addRow("优化器参数:", self.opt_params_widget)

        # Initialize with default parameters
        self.update_optimizer_params('AutoOptimizer')

        group.setLayout(layout)
        return group

    def update_model_params(self, model_type):
        """Update model parameter fields based on selected type."""
        # Clear existing params
        while self.model_params_layout.count():
            item = self.model_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.model_params = {}

        if model_type == 'ELM':
            self.add_model_param("隐藏层神经元数", "hidden_dim", 64, "spin", 1, 10000)
            self.add_model_param("激活函数", "activation", "relu", "combo", combo_options=["relu", "sigmoid", "tanh"])

        elif model_type == 'GaussianProcess':
            self.add_model_param("噪声方差", "noise_var", 1e-5, "double", 1e-10, 1.0)
            self.add_model_param("长度尺度", "length_scale", 1.0, "double", 0.01, 100.0)
            self.add_model_param("输出尺度", "output_scale", 1.0, "double", 0.01, 100.0)

        elif model_type == 'NeuralNetwork':
            self.add_model_param("隐藏层维度", "hidden_dims", "128,64,32", "line")
            self.add_model_param("学习率", "learning_rate", 0.001, "double", 1e-6, 1.0)
            self.add_model_param("训练轮数", "epochs", 1000, "spin", 1, 100000)
            self.add_model_param("批次大小", "batch_size", 32, "spin", 1, 1024)
            self.add_model_param("权重衰减", "weight_decay", 1e-5, "double", 0.0, 1.0)
            self.add_model_param("Dropout", "dropout", 0.0, "double", 0.0, 0.9)
            self.add_model_param("激活函数", "activation", "relu", "combo", combo_options=["relu", "sigmoid", "tanh"])

        elif model_type == 'XGBoost':
            self.add_model_param("树数量", "n_estimators", 100, "spin", 1, 10000)
            self.add_model_param("最大深度", "max_depth", 6, "spin", 1, 50)
            self.add_model_param("学习率", "learning_rate", 0.1, "double", 0.001, 1.0)
            self.add_model_param("子采样比例", "subsample", 1.0, "double", 0.1, 1.0)
            self.add_model_param("列采样比例", "colsample_bytree", 1.0, "double", 0.1, 1.0)
            self.add_model_param("L1正则化", "reg_alpha", 0.0, "double", 0.0, 100.0)
            self.add_model_param("L2正则化", "reg_lambda", 1.0, "double", 0.0, 100.0)
            self.add_model_param("最小子节点权重", "min_child_weight", 1, "spin", 0, 100)

    def add_model_param(self, label, key, default, widget_type, min_val=None, max_val=None, combo_options=None):
        """Add a model parameter field."""
        self.model_params[key] = {'label': label, 'default': default, 'type': widget_type}

        if widget_type == "spin":
            spin = QSpinBox()
            spin.setRange(min_val or 1, max_val or 1000)
            spin.setValue(default)
            spin.setObjectName(key)
            self.model_params_layout.addRow(f"{label}:", spin)

        elif widget_type == "double":
            dspin = QDoubleSpinBox()
            dspin.setRange(min_val or 0.0, max_val or 1.0)
            dspin.setDecimals(6)
            dspin.setValue(default)
            dspin.setObjectName(key)
            self.model_params_layout.addRow(f"{label}:", dspin)

        elif widget_type == "line":
            line = QLineEdit(str(default))
            line.setObjectName(key)
            self.model_params_layout.addRow(f"{label}:", line)

        elif widget_type == "combo":
            combo = QComboBox()
            combo.addItems(combo_options)
            combo.setCurrentText(default)
            combo.setObjectName(key)
            self.model_params_layout.addRow(f"{label}:", combo)

    def update_optimizer_params(self, optimizer_type):
        """Update optimizer parameter fields based on selected type."""
        # Clear existing params
        while self.opt_params_layout.count():
            item = self.opt_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.opt_params = {}

        if optimizer_type == 'AutoOptimizer':
            # AutoOptimizer automatically selects based on model differentiability
            label = QLabel("自动选择最适合的优化器")
            self.opt_params_layout.addRow("", label)

        elif optimizer_type == 'LBFGS':
            self.add_opt_param("最大迭代次数", "n_iter", 100, "spin", 1, 10000)
            self.add_opt_param("收敛 tolerance", "tol", 1e-6, "double", 1e-12, 1e-2)
            self.add_opt_param("历史步数 m", "m", 10, "spin", 1, 100)
            self.add_opt_param("输出日志", "verbose", True, "check")

        elif optimizer_type == 'CG':
            self.add_opt_param("最大迭代次数", "n_iter", 100, "spin", 1, 10000)
            self.add_opt_param("收敛 tolerance", "tol", 1e-6, "double", 1e-12, 1e-2)
            self.add_opt_param("CG方法", "method", "FR", "combo", combo_options=["FR", "PR", "HS"])
            self.add_opt_param("输出日志", "verbose", True, "check")

        elif optimizer_type == 'Bayesian':
            self.add_opt_param("初始采样数", "n_initial", 10, "spin", 1, 100)
            self.add_opt_param("最大迭代次数", "n_iter", 50, "spin", 1, 1000)
            self.add_opt_param("采集函数", "acquisition", "EI", "combo", combo_options=["EI", "PI", "UCB"])
            self.add_opt_param("输出日志", "verbose", True, "check")

        elif optimizer_type == 'Genetic':
            self.add_opt_param("种群大小", "pop_size", 50, "spin", 10, 1000)
            self.add_opt_param("最大代数", "n_iter", 100, "spin", 1, 10000)
            self.add_opt_param("变异概率", "mutation_rate", 0.1, "double", 0.001, 0.5)
            self.add_opt_param("交叉概率", "crossover_rate", 0.8, "double", 0.1, 1.0)
            self.add_opt_param("精英比例", "elite_ratio", 0.1, "double", 0.0, 0.5)
            self.add_opt_param("输出日志", "verbose", True, "check")

    def add_opt_param(self, label, key, default, widget_type, min_val=None, max_val=None, combo_options=None):
        """Add an optimizer parameter field."""
        self.opt_params[key] = {'label': label, 'default': default, 'type': widget_type}

        if widget_type == "spin":
            spin = QSpinBox()
            spin.setRange(min_val or 1, max_val or 1000)
            spin.setValue(default)
            spin.setObjectName(key)
            self.opt_params_layout.addRow(f"{label}:", spin)

        elif widget_type == "double":
            dspin = QDoubleSpinBox()
            dspin.setRange(min_val or 0.0, max_val or 1.0)
            dspin.setDecimals(6)
            dspin.setValue(default)
            dspin.setObjectName(key)
            self.opt_params_layout.addRow(f"{label}:", dspin)

        elif widget_type == "line":
            line = QLineEdit(str(default))
            line.setObjectName(key)
            self.opt_params_layout.addRow(f"{label}:", line)

        elif widget_type == "combo":
            combo = QComboBox()
            combo.addItems(combo_options)
            combo.setCurrentText(default)
            combo.setObjectName(key)
            self.opt_params_layout.addRow(f"{label}:", combo)

        elif widget_type == "check":
            checkbox = QCheckBox()
            checkbox.setChecked(default)
            checkbox.setObjectName(key)
            self.opt_params_layout.addRow(f"{label}:", checkbox)

    def on_model_type_changed(self, text):
        """Handle model type change."""
        self.update_model_params(text)

    def on_optimizer_changed(self, text):
        """Handle optimizer type change."""
        self.update_optimizer_params(text)

    def on_apply(self):
        """Handle apply button click."""
        self.config['model_type'] = self.model_type_combo.currentText()
        self.config['input_dim'] = self.input_dim_spin.value()

        # Collect model params
        model_params = {}
        for key, info in self.model_params.items():
            widget = self.findChild(QWidget, key)
            if widget:
                if info['type'] == 'spin':
                    model_params[key] = widget.value()
                elif info['type'] == 'double':
                    model_params[key] = widget.value()
                elif info['type'] == 'line':
                    val = widget.text()
                    # Handle hidden_dims specially - parse comma-separated integers
                    if key == 'hidden_dims':
                        model_params[key] = [int(x.strip()) for x in val.split(',')]
                    else:
                        try:
                            model_params[key] = float(val)
                        except ValueError:
                            model_params[key] = val
                elif info['type'] == 'combo':
                    model_params[key] = widget.currentText()
        self.config['model_params'] = model_params

        # Collect optimizer params
        self.config['optimizer'] = self.optimizer_combo.currentText()
        opt_params = {}
        for key, info in self.opt_params.items():
            widget = self.findChild(QWidget, key)
            if widget:
                if info['type'] == 'spin':
                    opt_params[key] = widget.value()
                elif info['type'] == 'double':
                    opt_params[key] = widget.value()
                elif info['type'] == 'line':
                    opt_params[key] = widget.text()
                elif info['type'] == 'combo':
                    opt_params[key] = widget.currentText()
                elif info['type'] == 'check':
                    opt_params[key] = widget.isChecked()
        self.config['optimizer_params'] = opt_params

        # Visual feedback - flash button green briefly
        original_style = self.btn_apply.styleSheet()
        self.btn_apply.setStyleSheet("background-color: #3FB950; color: white;")
        self.btn_apply.setText("Applied!")
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(800, lambda: self.btn_apply.setStyleSheet(original_style))
        QTimer.singleShot(800, lambda: self.btn_apply.setText("Apply"))

        return self.config

    def on_reset(self):
        """Handle reset button click."""
        self.model_type_combo.setCurrentText('NeuralNetwork')
        self.optimizer_combo.setCurrentText('AutoOptimizer')
        self.input_dim_spin.setValue(10)
        self.update_model_params('NeuralNetwork')
        self.update_optimizer_params('AutoOptimizer')

    def get_config(self):
        """Get current configuration."""
        self.on_apply()  # Apply current settings first
        return self.config

    def set_config(self, config):
        """Set configuration from external source."""
        self.config = config

        # Set model type and params
        if 'model_type' in config:
            self.model_type_combo.setCurrentText(config['model_type'])
            self.update_model_params(config['model_type'])

        if 'input_dim' in config:
            self.input_dim_spin.setValue(config['input_dim'])

        if 'model_params' in config:
            for key, value in config['model_params'].items():
                widget = self.findChild(QWidget, key)
                if widget:
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(value))
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QLineEdit):
                        if key == 'hidden_dims' and isinstance(value, list):
                            widget.setText(','.join(map(str, value)))
                        else:
                            widget.setText(str(value))
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))

        # Set optimizer type and params
        if 'optimizer' in config:
            self.optimizer_combo.setCurrentText(config['optimizer'])
            self.update_optimizer_params(config['optimizer'])

        if 'optimizer_params' in config:
            for key, value in config['optimizer_params'].items():
                widget = self.findChild(QWidget, key)
                if widget:
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(value))
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(value))
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(bool(value))

    def update_dimensions(self, input_dim, output_dim):
        """Update input and output dimensions from imported data.

        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        self.input_dim_spin.setValue(input_dim)
        self.output_dim_spin.setValue(output_dim)
        self.config['input_dim'] = input_dim
        self.config['output_dim'] = output_dim

    def set_columns_info(self, input_cols, output_cols):
        """Update column selection info display.

        Args:
            input_cols: List of input column names
            output_cols: List of output column names
        """
        self.config['input_columns'] = input_cols
        self.config['output_columns'] = output_cols

        # Update labels
        if input_cols is not None and len(input_cols) > 0:
            self.input_cols_label.setText(f"{len(input_cols)} 列: {', '.join(input_cols[:3])}{'...' if len(input_cols) > 3 else ''}")
            self.input_cols_label.setStyleSheet("color: green;")
        else:
            self.input_cols_label.setText("未选择")
            self.input_cols_label.setStyleSheet("color: gray;")

        if output_cols is not None and len(output_cols) > 0:
            self.output_cols_label.setText(f"{len(output_cols)} 列: {', '.join(output_cols[:3])}{'...' if len(output_cols) > 3 else ''}")
            self.output_cols_label.setStyleSheet("color: green;")
        else:
            self.output_cols_label.setText("未选择")
            self.output_cols_label.setStyleSheet("color: gray;")

    def set_data_loaded(self, n_samples, file_path=None):
        """Update data status after successful import.

        Args:
            n_samples: Number of samples in the loaded data
            file_path: Optional file path for display
        """
        if file_path:
            self.data_status_label.setText(f"已加载: {n_samples} 样本 ({file_path.split('/')[-1]})")
        else:
            self.data_status_label.setText(f"已加载: {n_samples} 样本")
        self.data_status_label.setStyleSheet("color: green; font-weight: bold;")