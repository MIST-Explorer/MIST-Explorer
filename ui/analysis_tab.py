import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stardist.models import StarDist2D
from PIL import Image
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import cv2
import os
from numba import njit
from tqdm import tqdm

from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, 
                             QGroupBox, QFormLayout, QScrollArea, QSizePolicy, QPushButton, 
                             QListWidget, QListWidgetItem, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

import ui.graphing.Test as test


        
class AnalysisTab(QWidget):
    def __init__(self, pixmap_label):
        super().__init__()

        self.initUI()


    def initUI(self):
        main_layout = QVBoxLayout()        
        sc = test.Window()
        
        main_layout.addWidget(sc)
        
        self.setLayout(main_layout)

    
   