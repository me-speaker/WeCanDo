"""Project dialog for creating new projects."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt


class ProjectDialog(QDialog):
    """Dialog for creating or editing a project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Project name
        name_layout = QHBoxLayout()
        name_label = QLabel("项目名称:")
        name_label.setMinimumWidth(100)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入项目名称")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Save path
        path_layout = QHBoxLayout()
        path_label = QLabel("保存路径:")
        path_label.setMinimumWidth(100)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择项目保存位置")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.on_browse)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        # Description
        desc_layout = QHBoxLayout()
        desc_label = QLabel("描述:")
        desc_label.setMinimumWidth(100)
        self.desc_input = QLineEdit()
        self.desc_input.setPlaceholderText("项目描述(可选)")
        desc_layout.addWidget(desc_label)
        desc_layout.addWidget(self.desc_input)
        layout.addLayout(desc_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def on_browse(self):
        """Handle browse button click."""
        path = QFileDialog.getExistingDirectory(
            self, "选择文件夹", "", QFileDialog.ShowDirsOnly
        )
        if path:
            self.path_input.setText(path)

    def on_ok(self):
        """Handle OK button click."""
        name = self.name_input.text().strip()
        path = self.path_input.text().strip()

        if not name:
            QMessageBox.warning(self, "警告", "请输入项目名称")
            return

        if not path:
            QMessageBox.warning(self, "警告", "请选择保存路径")
            return

        self.accept()

    def get_values(self):
        """Get the project name and path."""
        return self.name_input.text().strip(), self.path_input.text().strip()

    def get_description(self):
        """Get the project description."""
        return self.desc_input.text().strip()
