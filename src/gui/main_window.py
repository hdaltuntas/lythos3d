import sys

class DummyApplication:
    """A dummy class to represent QApplication if PyQt/PySide is not installed."""
    def __init__(self, sys_argv):
        pass
    def exec(self):
        return 0

class DummyWidget:
    """A dummy class to represent QWidget."""
    def __init__(self):
        pass
    def show(self):
        print("GUI Window Shown (Dummy)")

try:
    # Try to import PyQt6 or PyQt5, fallback to Dummy if not found (for CI/headless testing)
    from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton
    HAS_QT = True
except ImportError:
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton
        HAS_QT = True
    except ImportError:
        HAS_QT = False

if not HAS_QT:
    QApplication = DummyApplication
    QMainWindow = DummyWidget
    QTabWidget = DummyWidget
    QWidget = DummyWidget
    QVBoxLayout = DummyWidget
    QLabel = DummyWidget
    QPushButton = DummyWidget

class MainWindow(QMainWindow if HAS_QT else DummyWidget):
    def __init__(self):
        super().__init__()

        if HAS_QT:
            self.setWindowTitle("Lythos3D - Geotechnical FEM")
            self.resize(800, 600)

            self.tabs = QTabWidget()
            self.setCentralWidget(self.tabs)

            self.setup_materials_tab()
            self.setup_analysis_tab()

    def setup_materials_tab(self):
        if not HAS_QT: return
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Define Soil and Pile properties here."))
        # In reality, add tables/forms for Mohr-Coulomb parameters and Pile specs
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Materials")

    def setup_analysis_tab(self):
        if not HAS_QT: return
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Configure and Run Analysis."))

        self.run_btn = QPushButton("Run SSR Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Analysis")

    def run_analysis(self):
        print("Run button clicked! Starting analysis pipeline...")
        # Hook into src.fem.ssr here
