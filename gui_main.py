"""DAE-ELM GUI Entry Point."""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Main entry point for DAE-ELM GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("DAE-ELM")
    app.setOrganizationName("DeepInd")

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    main()