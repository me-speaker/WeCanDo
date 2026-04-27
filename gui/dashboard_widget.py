"""Dashboard widget for DeepInd GUI - Industrial Tech Style."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea,
    QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon


class InfoCard(QFrame):
    """Card widget for displaying key information."""

    clicked = pyqtSignal()

    def __init__(self, title, icon="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self._title = title
        self._icon = icon
        self.init_ui()

    def init_ui(self):
        """Initialize the card UI."""
        self.setStyleSheet("""
            QFrame#card {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 16px;
            }
            QFrame#card:hover {
                border-color: #58A6FF;
                background-color: #1C2128;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header with icon and title
        header = QHBoxLayout()
        header.setSpacing(8)

        self.icon_label = QLabel(self._icon)
        self.icon_label.setStyleSheet("font-size: 20px;")

        self.title_label = QLabel(self._title)
        self.title_label.setStyleSheet("""
            color: #8B949E;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """)

        header.addWidget(self.icon_label)
        header.addWidget(self.title_label)
        header.addStretch()

        # Value label
        self.value_label = QLabel("--")
        self.value_label.setObjectName("value")
        self.value_label.setStyleSheet("""
            color: #58A6FF;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 24px;
            font-weight: bold;
        """)

        # Subtitle/status
        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("color: #8B949E; font-size: 11px;")

        layout.addLayout(header)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)
        layout.addStretch()

        self.setLayout(layout)

    def set_value(self, value):
        """Set the main value text."""
        self.value_label.setText(str(value))

    def set_subtitle(self, subtitle):
        """Set the subtitle text."""
        self.subtitle_label.setText(subtitle)

    def set_value_color(self, color):
        """Set the value text color."""
        self.value_label.setStyleSheet(f"""
            color: {color};
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 24px;
            font-weight: bold;
        """)


class ActionButton(QPushButton):
    """Styled action button for quick actions."""

    def __init__(self, text, icon="", parent=None):
        super().__init__(parent)
        self.setText(f"  {icon}  {text}" if icon else f"  {text}")
        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #21262D;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #58A6FF;
            }
            QPushButton:pressed {
                background-color: #161B22;
            }
        """)


class DashboardWidget(QWidget):
    """Main dashboard widget with cards and quick actions."""

    # Signals for navigation
    navigate_request = pyqtSignal(str)  # 'new_project', 'import_data', 'run_optimization', etc.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the dashboard UI."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setStyleSheet("""
            color: #F0F6FC;
            font-size: 28px;
            font-weight: bold;
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.status_indicator = QLabel("Ready")
        self.status_indicator.setStyleSheet("""
            background-color: #238636;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.status_indicator)

        main_layout.addLayout(header_layout)

        # Info Cards Row
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # Data Overview Card
        self.data_card = InfoCard("数据概览", "📊")
        self.data_card.set_value("0")
        self.data_card.set_subtitle("样本数 | 0 输入 | 0 输出")
        cards_layout.addWidget(self.data_card)

        # Model Status Card
        self.model_card = InfoCard("模型状态", "🤖")
        self.model_card.set_value("未训练")
        self.model_card.set_subtitle("等待数据导入")
        self.model_card.set_value_color("#D29922")
        cards_layout.addWidget(self.model_card)

        # Optimization Progress Card
        self.progress_card = InfoCard("优化进度", "📈")
        self.progress_card.set_value("0%")
        self.progress_card.set_subtitle("未开始")
        self.progress_card.set_value_color("#8B949E")
        cards_layout.addWidget(self.progress_card)

        # Best Result Card
        self.result_card = InfoCard("最优结果", "🎯")
        self.result_card.set_value("--")
        self.result_card.set_subtitle("等待优化完成")
        self.result_card.set_value_color("#8B949E")
        cards_layout.addWidget(self.result_card)

        for card in [self.data_card, self.model_card, self.progress_card, self.result_card]:
            card.setMinimumWidth(200)

        main_layout.addLayout(cards_layout)

        # Quick Actions Section
        actions_group = QFrame()
        actions_group.setObjectName("card")
        actions_group.setStyleSheet("""
            QFrame#card {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 16px;
            }
        """)

        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(12)

        actions_title = QLabel("⚡ 快捷操作")
        actions_title.setStyleSheet("""
            color: #F0F6FC;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
        """)
        actions_layout.addWidget(actions_title)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.btn_new = ActionButton("新建项目", "📁")
        self.btn_new.clicked.connect(lambda: self.navigate_request.emit('new_project'))

        self.btn_import = ActionButton("导入数据", "📥")
        self.btn_import.clicked.connect(lambda: self.navigate_request.emit('import_data'))

        self.btn_optimize = ActionButton("执行优化", "🚀")
        self.btn_optimize.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: white;
                border: 1px solid #2EA043;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2EA043;
            }
        """)
        self.btn_optimize.clicked.connect(lambda: self.navigate_request.emit('optimize'))

        self.btn_export = ActionButton("导出结果", "📤")
        self.btn_export.clicked.connect(lambda: self.navigate_request.emit('export'))

        self.btn_view_report = ActionButton("查看报告", "📋")
        self.btn_view_report.clicked.connect(lambda: self.navigate_request.emit('report'))

        for btn in [self.btn_new, self.btn_import, self.btn_optimize, self.btn_export, self.btn_view_report]:
            btn.setMinimumWidth(130)
            buttons_layout.addWidget(btn)

        actions_layout.addLayout(buttons_layout)
        actions_group.setLayout(actions_layout)
        main_layout.addWidget(actions_group)

        # Recent Logs Section
        logs_group = QFrame()
        logs_group.setObjectName("card")
        logs_group.setStyleSheet("""
            QFrame#card {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 16px;
            }
        """)

        logs_layout = QVBoxLayout()
        logs_layout.setSpacing(12)

        logs_header = QHBoxLayout()
        logs_title = QLabel("📋 最近日志")
        logs_title.setStyleSheet("""
            color: #F0F6FC;
            font-size: 16px;
            font-weight: bold;
        """)
        logs_header.addWidget(logs_title)
        logs_header.addStretch()

        self.clear_logs_btn = QPushButton("清空")
        self.clear_logs_btn.setMaximumWidth(60)
        self.clear_logs_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #8B949E;
                border: 1px solid #30363D;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #21262D;
                color: #F0F6FC;
            }
        """)
        logs_header.addWidget(self.clear_logs_btn)

        logs_layout.addLayout(logs_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0D1117;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        self.log_text.setPlaceholderText("日志输出将显示在这里...")
        logs_layout.addWidget(self.log_text)

        logs_group.setLayout(logs_layout)
        main_layout.addWidget(logs_group)

        main_layout.addStretch()

        self.setLayout(main_layout)

    def update_data_info(self, n_samples, n_inputs, n_outputs):
        """Update data overview card."""
        self.data_card.set_value(f"{n_samples:,}")
        self.data_card.set_subtitle(f"{n_inputs} 输入 | {n_outputs} 输出")

    def update_model_status(self, model_type, status):
        """Update model status card."""
        self.model_card.set_value(model_type)
        if status == "trained":
            self.model_card.set_subtitle("✓ 已训练完成")
            self.model_card.set_value_color("#3FB950")
        elif status == "training":
            self.model_card.set_subtitle("训练中...")
            self.model_card.set_value_color("#D29922")
        else:
            self.model_card.set_subtitle("等待数据导入")
            self.model_card.set_value_color("#D29922")

    def update_optimization_progress(self, current, total, best_value):
        """Update optimization progress card."""
        if total > 0:
            pct = int(current / total * 100)
            self.progress_card.set_value(f"{pct}%")
            best_val_str = f"{float(best_value):.6f}" if best_value is not None and (not hasattr(best_value, '__len__') or len(best_value) == 1) else str(best_value)
            self.progress_card.set_subtitle(f"第 {current}/{total} 代 | 最优: {best_val_str}")
            self.progress_card.set_value_color("#58A6FF")
        else:
            self.progress_card.set_value("0%")
            self.progress_card.set_subtitle("未开始")

    def update_best_result(self, value):
        """Update best result card."""
        if value is not None:
            self.result_card.set_value(f"{value:.6f}")
            self.result_card.set_subtitle("优化完成")
            self.result_card.set_value_color("#3FB950")
        else:
            self.result_card.set_value("--")
            self.result_card.set_subtitle("等待优化完成")
            self.result_card.set_value_color("#8B949E")

    def append_log(self, message):
        """Append a message to the log display."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def clear_logs(self):
        """Clear the log display."""
        self.log_text.clear()
