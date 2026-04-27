"""Enhanced visualization module for DeepInd GUI - Industrial Tech Style."""

import warnings
import logging

# Suppress all matplotlib warnings (including C-level font warnings)
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QFrame, QFileDialog, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch

# Configure matplotlib for dark theme
# Only use fonts that are likely to exist to avoid findfont warnings
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.font_manager._findfont = matplotlib.font_manager.fontManager.findfont
def _safe_findfont(*args, **kwargs):
    try:
        return matplotlib.font_manager._findfont(*args, **kwargs)
    except Exception:
        return matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family='DejaVu Sans'))
matplotlib.font_manager.findfont = _safe_findfont

matplotlib.rcParams['axes.facecolor'] = '#0D1117'
matplotlib.rcParams['figure.facecolor'] = '#161B22'
matplotlib.rcParams['axes.edgecolor'] = '#30363D'
matplotlib.rcParams['axes.labelcolor'] = '#F0F6FC'
matplotlib.rcParams['xtick.color'] = '#8B949E'
matplotlib.rcParams['ytick.color'] = '#8B949E'
matplotlib.rcParams['text.color'] = '#F0F6FC'
matplotlib.rcParams['grid.color'] = '#21262D'
matplotlib.rcParams['grid.alpha'] = 0.8


class EnhancedVisualizationWidget(QWidget):
    """Enhanced widget for displaying optimization visualizations with real-time updates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._realtime_enabled = False
        self._realtime_data = {'generations': [], 'best_fitness': []}
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_realtime_chart)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header with controls
        header_layout = QHBoxLayout()

        # Chart type selector
        type_layout = QHBoxLayout()
        type_layout.setSpacing(8)
        type_label = QLabel("Chart Type:")
        type_label.setStyleSheet("color: #8B949E; font-size: 13px;")

        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Optimization History",
            "Pareto Front",
            "Predicted vs Actual",
            "Feature Importance",
            "Residual Analysis"
        ])
        self.chart_combo.setMinimumWidth(180)
        self.chart_combo.setStyleSheet("""
            QComboBox {
                background-color: #21262D;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QComboBox:hover {
                border-color: #58A6FF;
            }
        """)
        self.chart_combo.currentTextChanged.connect(self.on_chart_type_changed)

        type_layout.addWidget(type_label)
        type_layout.addWidget(self.chart_combo)
        header_layout.addLayout(type_layout)

        # Realtime toggle
        self.realtime_check = QCheckBox("Real-time Update")
        self.realtime_check.setStyleSheet("""
            QCheckBox {
                color: #F0F6FC;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #30363D;
                border-radius: 4px;
                background-color: #0D1117;
            }
            QCheckBox::indicator:checked {
                background-color: #58A6FF;
                border-color: #58A6FF;
            }
        """)
        self.realtime_check.stateChanged.connect(self.toggle_realtime)
        header_layout.addWidget(self.realtime_check)

        header_layout.addStretch()

        # Action buttons
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262D;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #58A6FF;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_chart)

        self.export_btn = QPushButton("Export")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262D;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #58A6FF;
            }
        """)
        self.export_btn.clicked.connect(self.export_chart)

        header_layout.addWidget(self.refresh_btn)
        header_layout.addWidget(self.export_btn)

        main_layout.addLayout(header_layout)

        # Chart container
        self.chart_container = QFrame()
        self.chart_container.setStyleSheet("""
            QFrame {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
            }
        """)

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(8, 8, 8, 8)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.patch.set_facecolor('#161B22')
        self.canvas = FigureCanvas(self.figure)
        container_layout.addWidget(self.canvas)

        self.chart_container.setLayout(container_layout)
        main_layout.addWidget(self.chart_container)

        # Stats bar
        self.stats_layout = QHBoxLayout()
        self.stats_layout.setSpacing(24)

        self.stat_labels = {}

        stats = [
            ("current_gen", "Current Gen"),
            ("best_fitness", "Best Fitness"),
            ("avg_fitness", "Avg Fitness"),
            ("data_points", "Data Points")
        ]

        for key, label_text in stats:
            stat_layout = QVBoxLayout()
            stat_layout.setSpacing(2)

            label = QLabel(label_text)
            label.setStyleSheet("color: #8B949E; font-size: 11px;")

            value = QLabel("--")
            value.setObjectName(f"stat_{key}")
            value.setStyleSheet("""
                color: #58A6FF;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 16px;
                font-weight: bold;
            """)

            self.stat_labels[key] = value
            stat_layout.addWidget(value)
            stat_layout.addWidget(label)
            self.stats_layout.addLayout(stat_layout)

        self.stats_layout.addStretch()
        main_layout.addLayout(self.stats_layout)

        self.setLayout(main_layout)

        # Initialize with empty plot
        self.plot_empty()

    def on_chart_type_changed(self, chart_type):
        """Handle chart type change."""
        self._realtime_enabled = False
        self.realtime_check.setChecked(False)
        self.refresh_chart()

    def toggle_realtime(self, enabled):
        """Toggle real-time update mode."""
        self._realtime_enabled = enabled
        if enabled:
            self._realtime_data = {'generations': [], 'best_fitness': []}
            self._update_timer.start(500)  # Update every 500ms
        else:
            self._update_timer.stop()
        self.refresh_chart()

    def add_realtime_point(self, generation, best_fitness):
        """Add a data point for real-time plotting."""
        if self._realtime_enabled:
            self._realtime_data['generations'].append(generation)
            self._realtime_data['best_fitness'].append(best_fitness)
            # Keep only last 200 points
            if len(self._realtime_data['generations']) > 200:
                self._realtime_data['generations'] = self._realtime_data['generations'][-200:]
                self._realtime_data['best_fitness'] = self._realtime_data['best_fitness'][-200:]

    def _update_realtime_chart(self):
        """Update chart in real-time mode."""
        if self._realtime_enabled and len(self._realtime_data.get('generations', [])) > 0:
            self.plot_optimization_history(self._realtime_data)

    def plot_empty(self):
        """Plot an empty chart with placeholder text."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        # Add styled placeholder
        ax.text(0.5, 0.5, "No Data\n\nRun optimization to generate data",
                ha='center', va='center', fontsize=14,
                color='#8B949E', transform=ax.transAxes)

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        self.canvas.draw()

    def plot_optimization_history(self, data=None):
        """Plot optimization history (convergence curve)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        generations = []
        best_fitness = []

        if data is None or not data.get('generations'):
            # Generate sample data
            generations = list(range(100))
            best_fitness = [100 * np.exp(-0.03 * g) + np.random.normal(0, 2) for g in generations]
            best_fitness = np.maximum.accumulate(best_fitness)
        else:
            generations = data.get('generations', list(range(len(data.get('best_fitness', [])))))
            best_fitness = data.get('best_fitness', [])

        if generations is not None and len(generations) > 0 and best_fitness is not None and len(best_fitness) > 0:
            # Main line with gradient fill
            ax.plot(generations, best_fitness, color='#58A6FF', linewidth=2, label='Best Fitness')

            # Fill area under curve
            ax.fill_between(generations, best_fitness, alpha=0.2, color='#58A6FF')

            # Current best marker
            current_best = best_fitness[-1] if best_fitness is not None and len(best_fitness) > 0 else 0
            ax.scatter([generations[-1]], [current_best], color='#3FB950', s=80, zorder=5, marker='*')
            ax.annotate(f'{current_best:.4f}',
                        xy=(generations[-1], current_best),
                        xytext=(10, 10), textcoords='offset points',
                        color='#3FB950', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#161B22', edgecolor='#3FB950', alpha=0.9))

            # Update stats
            self.stat_labels['current_gen'].setText(str(generations[-1]))
            self.stat_labels['best_fitness'].setText(f"{current_best:.6f}")
            self.stat_labels['avg_fitness'].setText(f"{np.mean(best_fitness):.6f}")
            self.stat_labels['data_points'].setText(str(len(generations)))

        ax.set_xlabel('Iterations', fontsize=12, color='#8B949E')
        ax.set_ylabel('Best Fitness', fontsize=12, color='#8B949E')
        ax.set_title('Optimization History', fontsize=14, fontweight='bold', color='#F0F6FC', pad=20)

        ax.grid(True, alpha=0.3, color='#21262D')
        ax.tick_params(colors='#8B949E')

        # Style spines
        for spine in ax.spines.values():
            spine.set_color('#30363D')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_pareto_front(self, data=None):
        """Plot Pareto front for multi-objective optimization."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        if data is None:
            n_points = 50
            obj1 = np.linspace(0, 1, n_points)
            obj2 = 1 - np.sqrt(obj1) + np.random.normal(0, 0.05, n_points)
            obj2 = np.maximum(obj2, 0)
        else:
            obj1 = data.get('objectives1', [])
            obj2 = data.get('objectives2', [])

        if len(obj1) > 0:
            sorted_indices = np.argsort(obj1)
            obj1 = np.array(obj1)[sorted_indices]
            obj2 = np.array(obj2)[sorted_indices]

            # Pareto front line
            ax.plot(obj1, obj2, color='#F85149', linewidth=2, label='Pareto Front', zorder=2)

            # Scatter points
            scatter = ax.scatter(obj1, obj2, c=obj2, cmap='RdYlGn', s=60, alpha=0.8, zorder=3)

            # Colorbar
            self.figure.colorbar(scatter, ax=ax, label='Objective 2', pad=0.02)

            # Update stats
            self.stat_labels['data_points'].setText(str(len(obj1)))
            self.stat_labels['best_fitness'].setText(f"{obj1[0]:.4f}")

        ax.set_xlabel('Objective 1 (Cost)', fontsize=12, color='#8B949E')
        ax.set_ylabel('Objective 2 (Performance)', fontsize=12, color='#8B949E')
        ax.set_title('Pareto Front Analysis', fontsize=14, fontweight='bold', color='#F0F6FC', pad=20)
        ax.legend(loc='upper right', framealpha=0.9, facecolor='#161B22', edgecolor='#30363D')
        ax.grid(True, alpha=0.3, color='#21262D')

        for spine in ax.spines.values():
            spine.set_color('#30363D')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_predicted_vs_actual(self, data=None):
        """Plot predicted vs actual values."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        if data is None:
            n_samples = 100
            actual = np.linspace(0, 10, n_samples) + np.random.normal(0, 0.5, n_samples)
            predicted = actual + np.random.normal(0, 0.8, n_samples)
        else:
            actual = data.get('actual', [])
            predicted = data.get('predicted', [])

        if len(actual) > 0:
            actual = np.array(actual)
            predicted = np.array(predicted)

            # Scatter plot with density coloring
            scatter = ax.scatter(actual, predicted, c='#58A6FF', alpha=0.6, s=50, label='Data Points')

            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Calculate and display R²
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # R² box
            r2_box = FancyBboxPatch((0.02, 0.88), 0.15, 0.08,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#21262D', edgecolor='#30363D',
                                     transform=ax.transAxes, zorder=5)
            ax.add_patch(r2_box)
            ax.text(0.095, 0.92, f'R² = {r2:.4f}', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12,
                   bbox=dict(facecolor='none', edgecolor='none'),
                   color='#3FB950' if r2 > 0.8 else '#D29922')

            # Update stats
            self.stat_labels['data_points'].setText(str(len(actual)))
            self.stat_labels['best_fitness'].setText(f"{r2:.4f}")
            self.stat_labels['avg_fitness'].setText(f"{np.mean(np.abs(actual - predicted)):.4f}")

        ax.set_xlabel('Actual', fontsize=12, color='#8B949E')
        ax.set_ylabel('Predicted', fontsize=12, color='#8B949E')
        ax.set_title('Predicted vs Actual', fontsize=14, fontweight='bold', color='#F0F6FC', pad=20)
        ax.legend(loc='lower right', framealpha=0.9, facecolor='#161B22', edgecolor='#30363D')
        ax.grid(True, alpha=0.3, color='#21262D')

        for spine in ax.spines.values():
            spine.set_color('#30363D')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_feature_importance(self, data=None):
        """Plot feature importance for the model."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        if data is None:
            # Sample feature importance
            features = [f'Feature {i+1}' for i in range(10)]
            importance = np.random.rand(10)
            importance = importance / importance.sum()
        else:
            features = data.get('features', [f'F{i}' for i in range(len(data.get('importance', [])))])
            importance = data.get('importance', [])

        if len(importance) > 0:
            # Sort by importance
            sorted_idx = np.argsort(importance)
            features = np.array(features)[sorted_idx]
            importance = np.array(importance)[sorted_idx]

            # Horizontal bar chart
            colors = ['#58A6FF' if i > 0.1 else '#3FB950' for i in importance]
            bars = ax.barh(features, importance, color=colors, height=0.6, edgecolor='#30363D')

            # Add value labels
            for bar, val in zip(bars, importance):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10, color='#8B949E')

            # Update stats
            self.stat_labels['data_points'].setText(str(len(features)))
            self.stat_labels['best_fitness'].setText(f"{importance.max():.4f}")

        ax.set_xlabel('Importance', fontsize=12, color='#8B949E')
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold', color='#F0F6FC', pad=20)
        ax.grid(True, alpha=0.3, color='#21262D', axis='x')

        for spine in ax.spines.values():
            spine.set_color('#30363D')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_residual_analysis(self, data=None):
        """Plot residual analysis."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#0D1117')

        if data is None:
            n_samples = 100
            actual = np.linspace(0, 10, n_samples)
            residuals = np.random.normal(0, 0.5, n_samples)
        else:
            actual = data.get('actual', [])
            residuals = data.get('residuals', [])

        if len(residuals) > 0:
            actual = np.array(actual)
            residuals = np.array(residuals)

            # Residuals vs predicted
            ax.scatter(actual, residuals, c='#58A6FF', alpha=0.6, s=50)

            # Zero line
            ax.axhline(y=0, color='#F85149', linestyle='--', linewidth=2, label='Zero Line')

            # Confidence interval
            std = np.std(residuals)
            ax.axhline(y=2*std, color='#D29922', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axhline(y=-2*std, color='#D29922', linestyle=':', linewidth=1.5, alpha=0.7)

            # Update stats
            self.stat_labels['data_points'].setText(str(len(residuals)))
            self.stat_labels['best_fitness'].setText(f"{np.mean(np.abs(residuals)):.4f}")
            self.stat_labels['avg_fitness'].setText(f"{std:.4f}")

        ax.set_xlabel('Actual', fontsize=12, color='#8B949E')
        ax.set_ylabel('Residual', fontsize=12, color='#8B949E')
        ax.set_title('Residual Analysis', fontsize=14, fontweight='bold', color='#F0F6FC', pad=20)
        ax.legend(loc='upper right', framealpha=0.9, facecolor='#161B22', edgecolor='#30363D')
        ax.grid(True, alpha=0.3, color='#21262D')

        for spine in ax.spines.values():
            spine.set_color('#30363D')

        self.figure.tight_layout()
        self.canvas.draw()

    def refresh_chart(self):
        """Refresh the chart based on selected type."""
        chart_type = self.chart_combo.currentText()

        if "Optimization" in chart_type:
            self.plot_optimization_history(self._realtime_data if self._realtime_enabled else None)
        elif "Pareto" in chart_type:
            self.plot_pareto_front()
        elif "Predicted" in chart_type:
            self.plot_predicted_vs_actual()
        elif "Feature" in chart_type:
            self.plot_feature_importance()
        elif "Residual" in chart_type:
            self.plot_residual_analysis()

    def export_chart(self):
        """Export the current chart as an image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart", "",
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, facecolor='#161B22',
                                    edgecolor='none', bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Chart exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")


# Backward compatibility alias
VisualizationWidget = EnhancedVisualizationWidget
