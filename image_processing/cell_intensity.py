
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv
from image_processing.canvas import ImageGraphicsView
import ui.app

class CellIntensity:
    def __init__(self):
        self.params = {
        'cell_channel': 'Channel 1',
        'intensity_channel': '2D_versatile_fluo',
        'max_size' : 1.00,
        'percentile_high': 99.80,
        'prob_threshold': 0.48,
        'nms_threshold': 0.3,
        'n_tiles': 0,
        'kernel_size': 3,
        'iterations': 1
        }

