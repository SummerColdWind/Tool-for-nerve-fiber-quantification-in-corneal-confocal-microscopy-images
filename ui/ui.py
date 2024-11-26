import os
import sys
import cv2

from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog, QListWidget, QMainWindow
from PyQt6.QtGui import QIcon, QPixmap, QImage

from process.processor import Processor

from .utils.output import OutputStream
from .utils.drop import label_drop, label_drag_enter
from .functions.io import open_image
from .functions.analysis import perform_analysis
from .functions.draw import draw_bone, draw_body, draw_binary, show_origin


class AiCCMWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'main.ui'), self)
        sys.stdout = OutputStream(self.LogWindow)

        self.p = Processor()
        self.raw_image = None
        self.cur_image = None

        self.init()

    def init(self):
        self.p.load_model(os.path.join(os.path.dirname(__file__), '../models/nerve.onnx'))
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'icon.ico')))
        self.actionOpen.triggered.connect(lambda: open_image(self))
        # self.ImageWindow.triggered.connect(lambda: open_image(self))
        self.ImageWindow.dragEnterEvent = lambda e: label_drag_enter(self.ImageWindow, e)
        self.ImageWindow.dropEvent = lambda e: label_drop(self, e)
        self.actionProcess.triggered.connect(lambda: perform_analysis(self))
        self.actionDraw_bone.triggered.connect(lambda: draw_bone(self))
        self.actionDraw_body.triggered.connect(lambda: draw_body(self))
        self.actionDraw_binary.triggered.connect(lambda: draw_binary(self))
        self.actionShow_origin.triggered.connect(lambda: show_origin(self))

        print('Initialization complete.')

    def show_image(self):
        image = self.cur_image if self.cur_image is not None else self.raw_image

        try:
            height, width, channel = image.shape
        except Exception as e:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            height, width, channel = image.shape


        bytes_per_line = channel * width
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)
        self.ImageWindow.setPixmap(pixmap)


