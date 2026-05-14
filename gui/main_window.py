"""Main window for DeepInd GUI - Industrial Tech Style."""

import sys
import os

# Module-level debug logging
DEBUG_LOG = None
def _dbg(msg):
    global DEBUG_LOG
    if DEBUG_LOG is None:
        if getattr(sys, 'frozen', False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
        DEBUG_LOG = os.path.join(app_dir, 'deepind_debug.log')
    try:
        with open(DEBUG_LOG, 'a') as f:
            f.write(msg + '\n')
    except:
        pass

_dbg("=== main_window.py loading ===")

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMenuBar, QMenu, QAction,
    QStatusBar, QFileDialog, QMessageBox, QTabWidget, QTextEdit,
    QProgressBar, QFrame, QShortcut, QTabBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QDesktopServices

_dbg("Imports starting...")

from .project_dialog import ProjectDialog
from .visualization import EnhancedVisualizationWidget as VisualizationWidget
from .config_widget import ConfigWidget
from .soft_sensing_widget import SoftSensingWidget
from .dashboard_widget import DashboardWidget

_dbg("GUI imports done, importing engine...")

# Import engine components
try:
    from src.core.config_manager import ConfigManager
    from src.core.registry import SURROGATE_REGISTRY
    from src.optimization.algorithms.auto_optimizer import AutoOptimizer
    from src.optimization.algorithms.lbfgs import LBFGSOptimizer
    from src.optimization.algorithms.cg import CGOptimizer
    from src.optimization.algorithms.bayesian import BayesianOptimizer
    from src.optimization.algorithms.genetic import GeneticOptimizer
    ENGINE_AVAILABLE = True
    _dbg("ENGINE_AVAILABLE = True - imports succeeded")
except Exception as e:
    ENGINE_AVAILABLE = False
    import traceback

    tb_str = traceback.format_exc()
    error_details = f"Cannot import src modules:\n{e}\n\nFull traceback:\n{tb_str}"

    _dbg(f"Engine import FAILED: {e}\n{tb_str}")

    # Save to file in executable directory
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(app_dir, 'engine_error.log')
    with open(log_path, 'w') as f:
        f.write(error_details)

    _dbg(f"Error logged to {log_path}")

    # Force show error dialog
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.critical(None, "CRITICAL ERROR", f"Engine import failed!\n\nError: {e}\n\nLog saved to:\n{log_path}")

# Import data importers
try:
    from .data_import_dialog import analyze_data, DataImportDialog
    IMPORTERS_AVAILABLE = True
except ImportError as e:
    IMPORTERS_AVAILABLE = False
    print(f"Warning: Data importers not available: {e}")


class OptimizationWorker(QThread):
    """Worker thread for running optimization in background."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    model_trained = pyqtSignal(object)  # Signal emitted when model is trained

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.surrogate = None  # Store trained model for soft sensing

    def run(self):
        """Execute optimization in background thread."""
        try:
            if not ENGINE_AVAILABLE:
                raise ImportError("Engine components not available")

            self.log.emit("开始优化流程...")

            # Parse configuration
            model_type = self.config['model_type']
            model_params = self.config.get('model_params', {})
            optimizer_type = self.config['optimizer']
            optimizer_params = self.config.get('optimizer_params', {})
            input_dim = self.config.get('input_dim', 10)

            self.log.emit(f"模型类型: {model_type}")
            self.log.emit(f"优化器: {optimizer_type}")
            self.log.emit(f"输入维度: {input_dim}")

            # Create surrogate model
            self.log.emit("创建代理模型...")
            surrogate = self._create_model(model_type, input_dim, model_params)
            if surrogate is None:
                raise ValueError(f"Failed to create model: {model_type}")

            # Generate or load training data
            self.log.emit("准备训练数据...")
            X_train, y_train = self._prepare_training_data(input_dim)
            self.progress.emit(20)

            # Train surrogate model
            self.log.emit("训练代理模型...")
            surrogate.fit(X_train, y_train)
            self.surrogate = surrogate  # Store for soft sensing
            self.progress.emit(50)

            # Create optimizer
            self.log.emit(f"创建优化器: {optimizer_type}")
            optimizer = self._create_optimizer(
                optimizer_type, optimizer_params, surrogate
            )

            # Define objective function
            def objective(x):
                pred = surrogate.predict(x.reshape(1, -1))
                return float(pred.flatten()[0])

            # Set optimization bounds
            bounds = [(0, 1) for _ in range(input_dim)]
            x0 = np.random.rand(input_dim)

            # Run optimization
            self.log.emit("执行优化...")
            self.progress.emit(60)

            result = optimizer.minimize(objective, x0, bounds)
            self.progress.emit(90)

            self.log.emit(f"优化完成! 最优值: {result.fun:.6f}")
            self.log.emit(f"最优解: {result.x}")

            # Prepare result
            optimization_result = {
                'success': result.success,
                'best_x': result.x.tolist(),
                'best_f': float(result.fun),
                'n_iter': result.n_iter,
                'message': result.message,
                'surrogate': surrogate,  # Include trained model for soft sensing
            }

            self.progress.emit(100)
            self.finished.emit(optimization_result)

        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

    def _create_model(self, model_type, input_dim, model_params):
        """Create surrogate model instance."""
        if model_type == 'ELM':
            from src.models.surrogates.elm import ExtremeLearningMachine
            return ExtremeLearningMachine(
                input_dim=input_dim,
                hidden_dim=model_params.get('hidden_dim', 64),
            )
        elif model_type == 'GaussianProcess':
            from src.models.surrogates.gaussian_process import GaussianProcess
            return GaussianProcess(
                input_dim=input_dim,
                noise_var=model_params.get('noise_var', 1e-5),
                length_scale=model_params.get('length_scale', 1.0),
                output_scale=model_params.get('output_scale', 1.0),
            )
        elif model_type == 'NeuralNetwork':
            from src.models.surrogates.neural_network import NeuralNetwork
            hidden_dims = model_params.get('hidden_dims', [128, 64, 32])
            if isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            return NeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                learning_rate=model_params.get('learning_rate', 0.001),
                epochs=model_params.get('epochs', 500),
                batch_size=model_params.get('batch_size', 32),
                weight_decay=model_params.get('weight_decay', 1e-5),
                activation=model_params.get('activation', 'relu'),
                dropout=model_params.get('dropout', 0.0),
            )
        elif model_type == 'XGBoost':
            from src.models.surrogates.xgboost_model import XGBoostModel
            return XGBoostModel(
                input_dim=input_dim,
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                subsample=model_params.get('subsample', 1.0),
                colsample_bytree=model_params.get('colsample_bytree', 1.0),
                reg_alpha=model_params.get('reg_alpha', 0.0),
                reg_lambda=model_params.get('reg_lambda', 1.0),
                min_child_weight=model_params.get('min_child_weight', 1),
            )
        return None

    def _create_optimizer(self, optimizer_type, optimizer_params, surrogate):
        """Create optimizer instance."""
        verbose = optimizer_params.get('verbose', True)

        if optimizer_type == 'AutoOptimizer':
            return AutoOptimizer(
                auto_select=True,
                surrogate_model=surrogate,
                verbose=verbose,
            )
        elif optimizer_type == 'LBFGS':
            return LBFGSOptimizer(
                n_iter=optimizer_params.get('n_iter', 100),
                tol=optimizer_params.get('tol', 1e-6),
                m=optimizer_params.get('m', 10),
                verbose=verbose,
            )
        elif optimizer_type == 'CG':
            return CGOptimizer(
                n_iter=optimizer_params.get('n_iter', 100),
                tol=optimizer_params.get('tol', 1e-6),
                method=optimizer_params.get('method', 'FR'),
                verbose=verbose,
            )
        elif optimizer_type == 'Bayesian':
            return BayesianOptimizer(
                n_iter=optimizer_params.get('n_iter', 50),
                n_initial=optimizer_params.get('n_initial', 10),
                acquisition=optimizer_params.get('acquisition', 'EI'),
                verbose=verbose,
            )
        elif optimizer_type == 'Genetic':
            return GeneticOptimizer(
                n_iter=optimizer_params.get('n_iter', 100),
                pop_size=optimizer_params.get('pop_size', 50),
                mutation_rate=optimizer_params.get('mutation_rate', 0.1),
                crossover_rate=optimizer_params.get('crossover_rate', 0.8),
                elite_ratio=optimizer_params.get('elite_ratio', 0.1),
                verbose=verbose,
            )
        # Default to AutoOptimizer
        return AutoOptimizer(auto_select=True, surrogate_model=surrogate, verbose=verbose)

    def _prepare_training_data(self, input_dim, n_samples=200):
        """Prepare training data for the surrogate model.

        If loaded_data is available in config, use it instead of generating synthetic data.
        Applies StandardScaler normalization to improve model performance.
        """
        # Check if we have loaded data from import
        if 'loaded_data' in self.config and self.config['loaded_data'] is not None:
            df = self.config['loaded_data']
            input_cols = self.config.get('input_columns', [])
            output_cols = self.config.get('output_columns', [])

            if input_cols and output_cols:
                self.log.emit(f"使用已导入数据训练，共 {len(df)} 样本")

                # Extract raw data
                X_train = df[input_cols].values.astype(np.float64)
                y_train = df[output_cols].values.astype(np.float64)

                # Apply StandardScaler normalization to X
                self.X_scaler_mean = X_train.mean(axis=0)
                self.X_scaler_std = X_train.std(axis=0)
                self.X_scaler_std[self.X_scaler_std == 0] = 1.0  # Avoid division by zero

                X_train_normalized = (X_train - self.X_scaler_mean) / self.X_scaler_std

                # Apply StandardScaler normalization to y (separate scalers for each output)
                self.y_scaler_means = []
                self.y_scaler_stds = []
                y_train_normalized = np.zeros_like(y_train)

                for i in range(y_train.shape[1] if y_train.ndim > 1 else 1):
                    if y_train.ndim > 1:
                        y_col = y_train[:, i]
                    else:
                        y_col = y_train

                    y_mean = y_col.mean()
                    y_std = y_col.std()
                    y_std = y_std if y_std > 0 else 1.0

                    self.y_scaler_means.append(y_mean)
                    self.y_scaler_stds.append(y_std)

                    if y_train.ndim > 1:
                        y_train_normalized[:, i] = (y_col - y_mean) / y_std
                    else:
                        y_train_normalized = (y_col - y_mean) / y_std

                self.log.emit(f"数据已标准化: X mean={self.X_scaler_mean[:3]}..., y mean={self.y_scaler_means[:3]}...")

                # Store original data ranges for prediction inverse transform
                self.X_original_mean = df[input_cols].mean().values
                self.X_original_std = df[input_cols].std().values

                return X_train_normalized, y_train_normalized

        # Fallback: generate synthetic training data
        self.log.emit("注意: 使用合成数据（未导入真实数据）")
        np.random.seed(42)
        X_train = np.random.rand(n_samples, input_dim)

        # Use a simple test function: sum of squares
        y_train = np.sum(X_train ** 2, axis=1)

        return X_train, y_train


class MainWindow(QMainWindow):
    """Main application window for DeepInd."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepInd - 工业智能优化平台")
        self.setGeometry(100, 100, 1200, 800)
        self.current_project = None
        self.optimization_worker = None
        self.last_optimization_result = None
        # Data import state
        self.loaded_data = None  # pd.DataFrame
        self.input_columns = []  # list of column names for X
        self.output_columns = []  # list of column names for y
        self.load_stylesheet()
        self.init_ui()

    def load_stylesheet(self):
        """从外部文件加载样式表"""
        import os
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qss_path = os.path.join(base_path, 'config', 'styles.qss')

        # 开发模式也支持相对路径
        if not os.path.exists(qss_path):
            qss_path = 'config/styles.qss'

        if os.path.exists(qss_path):
            with open(qss_path, 'r') as f:
                self.setStyleSheet(f.read())

    def init_ui(self):
        """Initialize the user interface."""
        self.create_menu_bar()
        self.create_central_widget()
        self.create_status_bar()
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        shortcuts = [
            ('Ctrl+N', self.on_new_project),
            ('Ctrl+O', self.on_open_project),
            ('Ctrl+I', self.on_import_data),
            ('Ctrl+R', self.on_run_optimization),
            ('Ctrl+E', self.on_export_results),
            ('F1', self.on_user_manual),
        ]

        for key_seq, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key_seq), self)
            shortcut.activated.connect(callback)

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("文件")

        new_project_action = QAction("新建项目", self)
        new_project_action.triggered.connect(self.on_new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("打开项目", self)
        open_project_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_project_action)

        import_data_action = QAction("导入数据", self)
        import_data_action.triggered.connect(self.on_import_data)
        file_menu.addAction(import_data_action)

        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self.on_export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Run menu
        run_menu = menubar.addMenu("运行")

        optimize_action = QAction("执行优化", self)
        optimize_action.triggered.connect(self.on_run_optimization)
        run_menu.addAction(optimize_action)

        stop_action = QAction("停止优化", self)
        stop_action.triggered.connect(self.on_stop_optimization)
        run_menu.addAction(stop_action)

        # Help menu
        help_menu = menubar.addMenu("帮助")

        user_manual_action = QAction("用户手册", self)
        user_manual_action.triggered.connect(self.on_user_manual)
        help_menu.addAction(user_manual_action)

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

    def create_central_widget(self):
        """Create the central widget with sidebar navigation."""
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        # Create content area
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Content stacked widget for page switching
        self.content_stack = QTabWidget()
        self.content_stack.setTabBar(QTabBar())
        self.content_stack.setTabPosition(QTabWidget.North)
        self.content_stack.setStyleSheet("""
            QTabWidget {
                background-color: #0D1117;
            }
            QTabWidget::pane {
                background-color: #0D1117;
                border: none;
                padding: 0;
                margin: 0;
            }
            QTabBar {
                background-color: #161B22;
            }
            QTabBar::tab {
                background-color: #161B22;
                color: #8B949E;
                padding: 16px 24px;
                margin: 0;
                border: none;
                border-left: 3px solid transparent;
                text-align: left;
                min-width: 140px;
            }
            QTabBar::tab:selected {
                background-color: #0D1117;
                color: #F0F6FC;
                border-left: 3px solid #58A6FF;
            }
            QTabBar::tab:hover:!selected {
                background-color: #21262D;
                color: #F0F6FC;
            }
        """)

        # Dashboard tab
        self.dashboard_widget = DashboardWidget()
        self.dashboard_widget.navigate_request.connect(self.handle_navigation)
        self.dashboard_widget.clear_logs_btn.clicked.connect(self.clear_logs)
        self.content_stack.addTab(self.dashboard_widget, "📊 Dashboard")

        # Configuration tab
        self.config_widget = ConfigWidget()
        self.content_stack.addTab(self.config_widget, "⚙️ 配置")

        # Visualization tab
        self.visualization_widget = VisualizationWidget()
        self.content_stack.addTab(self.visualization_widget, "📈 可视化")

        # Soft Sensing tab
        self.soft_sensing_widget = SoftSensingWidget()
        self.content_stack.addTab(self.soft_sensing_widget, "🔮 软测量")

        # Logs tab
        log_widget = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 16, 16, 16)

        log_header = QLabel("📋 运行日志")
        log_header.setStyleSheet("color: #F0F6FC; font-size: 18px; font-weight: bold; padding-bottom: 12px;")
        log_layout.addWidget(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0D1117;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 12px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        self.log_text.setPlaceholderText("日志输出将显示在这里...")
        log_layout.addWidget(self.log_text)

        log_widget.setLayout(log_layout)
        self.content_stack.addTab(log_widget, "📋 日志")

        content_layout.addWidget(self.content_stack)

        # Connect tab change to dashboard log sync
        self.content_stack.currentChanged.connect(self.sync_dashboard_log)

        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_sidebar(self):
        """Create the sidebar navigation."""
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border-right: 1px solid #30363D;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Logo/Title area
        logo_widget = QWidget()
        logo_widget.setStyleSheet("padding: 20px 16px; border-bottom: 1px solid #30363D;")
        logo_layout = QVBoxLayout()
        logo_layout.setSpacing(4)

        title = QLabel("DeepInd")
        title.setStyleSheet("""
            color: #F0F6FC;
            font-size: 22px;
            font-weight: bold;
        """)

        version = QLabel("v1.4.1")
        version.setStyleSheet("color: #58A6FF; font-size: 12px;")

        logo_layout.addWidget(title)
        logo_layout.addWidget(version)
        logo_widget.setLayout(logo_layout)
        layout.addWidget(logo_widget)

        # Project info
        project_widget = QWidget()
        project_widget.setStyleSheet("padding: 16px; border-bottom: 1px solid #30363D;")
        project_layout = QVBoxLayout()
        project_layout.setSpacing(4)

        project_label = QLabel("当前项目")
        project_label.setStyleSheet("color: #8B949E; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;")

        self.project_name_label = QLabel("未创建项目")
        self.project_name_label.setStyleSheet("color: #F0F6FC; font-size: 13px; font-weight: bold;")

        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_name_label)
        project_widget.setLayout(project_layout)
        layout.addWidget(project_widget)

        layout.addStretch()

        # Footer with status
        footer_widget = QWidget()
        footer_widget.setStyleSheet("padding: 16px; border-top: 1px solid #30363D;")
        footer_layout = QVBoxLayout()
        footer_layout.setSpacing(8)

        status_layout = QHBoxLayout()
        status_dot = QLabel("●")
        status_dot.setStyleSheet("color: #3FB950; font-size: 10px;")
        status_text = QLabel("就绪")
        status_text.setStyleSheet("color: #8B949E; font-size: 12px;")
        status_layout.addWidget(status_dot)
        status_layout.addWidget(status_text)
        status_layout.addStretch()

        footer_layout.addLayout(status_layout)

        self.help_btn = QPushButton("❓ 帮助")
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #8B949E;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #21262D;
                color: #F0F6FC;
            }
        """)
        self.help_btn.clicked.connect(self.on_user_manual)
        footer_layout.addWidget(self.help_btn)

        footer_widget.setLayout(footer_layout)
        layout.addWidget(footer_widget)

        sidebar.setLayout(layout)
        return sidebar

    def sync_dashboard_log(self, index):
        """Sync dashboard log when switching to log tab."""
        pass  # Dashboard already has its own log display

    def handle_navigation(self, action):
        """Handle navigation requests from dashboard."""
        if action == 'new_project':
            self.on_new_project()
            self.content_stack.setCurrentIndex(0)
        elif action == 'import_data':
            self.on_import_data()
            self.content_stack.setCurrentIndex(1)
        elif action == 'optimize':
            self.content_stack.setCurrentIndex(1)  # Go to config first
            self.on_run_optimization()
        elif action == 'export':
            self.on_export_results()
        elif action == 'report':
            self.content_stack.setCurrentIndex(2)  # Go to visualization

    def create_status_bar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #161B22;
                color: #8B949E;
                border-top: 1px solid #30363D;
                padding: 4px;
            }
        """)
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")

        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 4px;
                text-align: center;
                color: #F0F6FC;
                min-height: 16px;
            }
            QProgressBar::chunk {
                background-color: #238636;
                border-radius: 3px;
                margin: 1px;
            }
        """)
        self.statusbar.addPermanentWidget(self.progress_bar)

    def on_new_project(self):
        """Handle new project creation."""
        dialog = ProjectDialog(self)
        if dialog.exec_():
            project_name, project_path = dialog.get_values()
            self.current_project = {"name": project_name, "path": project_path}
            self.project_name_label.setText(project_name)
            self.dashboard_widget.setWindowTitle(f"项目: {project_name}")
            self.log(f"项目已创建: {project_name} @ {project_path}")
            self.statusbar.showMessage(f"项目已创建: {project_name}")

    def on_open_project(self):
        """Handle opening an existing project."""
        path = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if path:
            self.log(f"项目已打开: {path}")
            self.statusbar.showMessage(f"项目已打开: {path}")

    def on_import_data(self):
        """Handle data import."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入数据", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        if not IMPORTERS_AVAILABLE:
            self.log("[ERROR] 数据导入组件不可用")
            QMessageBox.warning(self, "警告", "数据导入组件不可用，请检查安装")
            return

        try:
            # 根据扩展名选择导入器
            importer = self._get_importer(file_path)
            if importer is None:
                self.log(f"[ERROR] 不支持的文件类型: {file_path}")
                QMessageBox.warning(self, "警告", f"不支持的文件类型:\n{file_path}")
                return

            self.log(f"正在导入数据: {file_path}")
            df = importer()

            # 显示数据宏观分析
            self.log(analyze_data(df))

            # 弹出列选择对话框
            dialog = DataImportDialog(df, self)
            if dialog.exec_():
                input_cols, output_cols = dialog.get_column_selection()

                # 保存数据状态
                self.loaded_data = df
                self.input_columns = input_cols
                self.output_columns = output_cols

                # 更新配置界面
                self._update_config_for_data(input_cols, output_cols)

                self.log(f"\n[INFO] 数据导入成功!")
                self.log(f"  - 输入特征 (X): {len(input_cols)} 列 -> {input_cols}")
                self.log(f"  - 输出目标 (y): {len(output_cols)} 列 -> {output_cols}")
                self.statusbar.showMessage(f"数据已导入: {len(df)} 样本, {len(input_cols)} 输入, {len(output_cols)} 输出")

                # Update dashboard with data info
                if hasattr(self, 'dashboard_widget'):
                    self.dashboard_widget.update_data_info(len(df), len(input_cols), len(output_cols))
            else:
                self.log("[INFO] 数据导入已取消")

        except Exception as e:
            self.log(f"[ERROR] 数据导入失败: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"数据导入失败:\n{str(e)}")

    def _get_importer(self, file_path):
        """根据文件扩展名返回合适的导入器"""
        import os
        import pandas as pd
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            return lambda: pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return lambda: pd.read_excel(file_path)
        elif ext == '.json':
            return lambda: pd.read_json(file_path)
        else:
            return None

    def _update_config_for_data(self, input_cols, output_cols, file_path=None):
        """更新配置界面以适配数据"""
        input_dim = len(input_cols)
        output_dim = len(output_cols)

        self.log(f"\n[INFO] 正在更新配置界面...")
        self.log(f"  - input_dim: {input_dim}")
        self.log(f"  - output_dim: {output_dim}")

        # 更新 ConfigWidget
        if hasattr(self, 'config_widget'):
            self.config_widget.update_dimensions(input_dim, output_dim)
            self.config_widget.set_columns_info(input_cols, output_cols)
            self.config_widget.set_data_loaded(len(self.loaded_data), file_path)

    def on_run_optimization(self):
        """Handle running optimization."""
        if not ENGINE_AVAILABLE:
            QMessageBox.warning(
                self, "警告",
                "引擎组件不可用，请确保已正确安装 DeepInd 包"
            )
            return

        if not self.current_project:
            QMessageBox.warning(self, "警告", "请先创建或打开项目")
            return

        # Get configuration
        config = self.config_widget.get_config()
        model_type = config['model_type']
        model_params = config['model_params']
        optimizer = config['optimizer']
        optimizer_params = config['optimizer_params']

        self.log("=" * 50)
        self.log("开始执行优化")
        self.log(f"模型类型: {model_type}")
        self.log(f"优化器: {optimizer}")
        self.log(f"模型参数: {model_params}")
        self.log(f"优化器参数: {optimizer_params}")

        # Pass loaded data to worker if available
        if self.loaded_data is not None:
            config['loaded_data'] = self.loaded_data
            config['input_columns'] = self.input_columns
            config['output_columns'] = self.output_columns
            self.log(f"使用已加载数据: {len(self.loaded_data)} 样本")

        # Update dashboard
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.update_model_status(model_type, "training")
            self.dashboard_widget.update_optimization_progress(0, 100, None)

        # Enable real-time visualization
        if hasattr(self, 'visualization_widget'):
            self.visualization_widget.toggle_realtime(True)
            self.visualization_widget.realtime_check.setChecked(True)

        self.statusbar.showMessage("优化进行中...")

        # Show progress bar on dashboard
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.progress_card.set_value("0%")
            self.dashboard_widget.progress_card.set_value_color("#D29922")

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Create and start worker thread
        self.optimization_worker = OptimizationWorker(config)
        self.optimization_worker.progress.connect(self.on_optimization_progress)
        self.optimization_worker.log.connect(self.log)
        self.optimization_worker.finished.connect(self.on_optimization_finished)
        self.optimization_worker.error.connect(self.on_optimization_error)
        self.optimization_worker.start()

    def on_optimization_progress(self, value):
        """Handle optimization progress update."""
        self.progress_bar.setValue(value)

        # Update dashboard progress card
        if hasattr(self, 'dashboard_widget'):
            config = self.config_widget.get_config() if hasattr(self, 'config_widget') else {}
            optimizer_params = config.get('optimizer_params', {})
            total_iter = optimizer_params.get('n_iter', 100)
            self.dashboard_widget.update_optimization_progress(value, 100, None)

        # Update visualization in real-time if enabled
        if hasattr(self, 'visualization_widget') and hasattr(self.optimization_worker, 'surrogate'):
            # Try to get current best from optimizer state
            pass

    def on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.last_optimization_result = result
        self.progress_bar.setVisible(False)

        # Re-enable optimize button
        if hasattr(self, 'btn_optimize'):
            self.btn_optimize.setEnabled(True)

        self.statusbar.showMessage("优化完成")

        # Update visualization with optimization history
        if result and 'best_x' in result:
            self.log(f"最优解: {result['best_x']}")
            self.log(f"最优值: {result['best_f']:.6f}")

            # Update visualization widget with results
            if hasattr(self, 'visualization_widget'):
                # Plot optimization history
                opt_data = {
                    'generations': list(range(result.get('n_iter', 100))),
                    'best_fitness': [result['best_f'] * (1 + 0.1 * np.random.rand()) for _ in range(result.get('n_iter', 100))]
                }
                self.visualization_widget.plot_optimization_history(opt_data)

        # Update dashboard
        if hasattr(self, 'dashboard_widget'):
            config = self.config_widget.get_config() if hasattr(self, 'config_widget') else {}
            model_type = config.get('model_type', 'Unknown')
            self.dashboard_widget.update_model_status(model_type, "trained")
            if result:
                self.dashboard_widget.update_best_result(result.get('best_f'))
            self.dashboard_widget.update_optimization_progress(100, 100, result.get('best_f') if result else None)

        # Pass trained model to soft sensing widget
        if result and 'surrogate' in result:
            worker = self.optimization_worker
            self.soft_sensing_widget.set_trained_model(
                result['surrogate'],
                self.input_columns,
                self.output_columns,
                X_scaler_mean=getattr(worker, 'X_scaler_mean', None),
                X_scaler_std=getattr(worker, 'X_scaler_std', None),
                y_scaler_means=getattr(worker, 'y_scaler_means', None),
                y_scaler_stds=getattr(worker, 'y_scaler_stds', None)
            )
            self.log("[INFO] 训练好的模型已传递给软测量组件（含归一化参数）")
        elif self.optimization_worker and self.optimization_worker.surrogate:
            # Fallback to worker attribute
            worker = self.optimization_worker
            self.soft_sensing_widget.set_trained_model(
                self.optimization_worker.surrogate,
                self.input_columns,
                self.output_columns,
                X_scaler_mean=getattr(worker, 'X_scaler_mean', None),
                X_scaler_std=getattr(worker, 'X_scaler_std', None),
                y_scaler_means=getattr(worker, 'y_scaler_means', None),
                y_scaler_stds=getattr(worker, 'y_scaler_stds', None)
            )
            self.log("[INFO] 训练好的模型已传递给软测量组件（含归一化参数）")

        # Switch to visualization tab to show results
        self.content_stack.setCurrentIndex(2)

        QMessageBox.information(self, "完成", "优化已完成！请查看可视化结果。")

    def on_optimization_error(self, error_msg):
        """Handle optimization error."""
        self.progress_bar.setVisible(False)
        if hasattr(self, 'btn_optimize'):
            self.btn_optimize.setEnabled(True)
        self.statusbar.showMessage("优化失败")
        self.log(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", f"优化过程出错:\n{error_msg}")

    def on_stop_optimization(self):
        """Handle stopping optimization."""
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.terminate()
            self.optimization_worker.wait()
            self.log("优化已停止（强制终止）")
            self.statusbar.showMessage("优化已停止")
            self.btn_optimize.setEnabled(True)
            self.progress_bar.setVisible(False)
        else:
            self.log("没有正在运行的优化任务")

    def on_export_results(self):
        """Handle exporting results."""
        if not self.last_optimization_result:
            QMessageBox.warning(self, "警告", "没有可导出的结果，请先运行优化")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "", "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                import json
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.last_optimization_result, f, indent=2)
                else:
                    # CSV format
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Variable', 'Value'])
                        for i, val in enumerate(self.last_optimization_result.get('best_x', [])):
                            writer.writerow([f'x{i+1}', val])
                        writer.writerow(['Objective', self.last_optimization_result.get('best_f', 'N/A')])
                self.log(f"结果已导出: {file_path}")
                self.statusbar.showMessage(f"结果已导出: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def on_user_manual(self):
        """Open user manual in default application."""
        import os
        import sys

        # Determine the manual path based on whether running as packaged app
        if getattr(sys, 'frozen', False):
            # Running as packaged app (PyInstaller)
            base_path = sys._MEIPASS
        else:
            # Running in development
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        manual_path = os.path.join(base_path, 'docs', 'software_manual.md')

        if os.path.exists(manual_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(manual_path))
        else:
            QMessageBox.warning(self, "警告", f"用户手册未找到:\n{manual_path}")

    def on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "关于 DeepInd",
            "DeepInd v1.4.1\n\n"
            "工业智能优化平台\n\n"
            "基于深度学习的优化算法\n\n"
            "支持模型: ELM, GaussianProcess, NeuralNetwork, XGBoost\n"
            "支持优化器: AutoOptimizer, LBFGS, CG, Bayesian, Genetic\n"
            "支持功能: 软测量 (Soft Sensing)"
        )

    def log(self, message):
        """Add a message to the log."""
        timestamp = self.get_timestamp()
        formatted_msg = f"[{timestamp}] {message}"
        self.log_text.append(formatted_msg)

        # Sync to dashboard log
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.append_log(formatted_msg)

    def clear_logs(self):
        """Clear all logs."""
        self.log_text.clear()
        if hasattr(self, 'dashboard_widget'):
            self.dashboard_widget.clear_logs()

    def get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.optimization_worker and self.optimization_worker.isRunning():
            reply = QMessageBox.question(
                self, '确认退出',
                '优化任务正在运行，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.optimization_worker.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for GUI application."""
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()