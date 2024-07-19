
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv
from image_processing.canvas import ImageGraphicsView
import ui.app

class CellIntensity:
    def __init__(self):
        self.params = {
        'alignment_layer': 'Channel 1',
        'cell_layer': 'Channel 3',
        'protein_detection_layer' : 'Channel 4',
        'max_size': 23000,
        'num_tiles': 5,
        'overlap': 500,
        'num_decoding_cycles': 3,
        'num_decoding_colors': 3,
        'radius_fg': 2,
        'radius_bg': 6
        }

    def setAlignmentLayer(self, channel):
        self.params['alignment_layer'] = channel

    def setCellLayer(self, model):
        self.params['cell_layer'] = model

    def setProteinDetectionLayer(self, value):
        self.params['protein_detection_layer'] = value

    def setMaxSize(self, value):
        self.params['max_size'] = value

    def setNumTiles(self, value):
        self.params['num_tiles'] = value

    def setOverlap(self, value):
        self.params['overlap'] = value

    def setNumDecodingCycles(self, value):
        self.params['num_decoding_cycles'] = value

    def setNumDecodingColors(self, value):
        self.params['num_decoding_colors'] = value

    def setRadiusFG(self, value):
        self.params['radius_fg'] = value

    def setRadiusBG(self, value):
        self.params['radius_bg'] = value
        print(value)

