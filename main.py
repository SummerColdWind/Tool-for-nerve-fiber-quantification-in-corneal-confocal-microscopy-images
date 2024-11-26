import sys
from ui.ui import AiCCMWindow
from PyQt6.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AiCCMWindow()
    window.show()
    sys.exit(app.exec())
