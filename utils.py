from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import  QImage
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot
import Dialogs, tifffile as tiff, numpy as np
from PIL import Image, ImageSequence
import cv2
import time



def numpy_to_qimage(array:np.ndarray) -> QImage:
    if len(array.shape) == 2:
        # Grayscale image
        height, width = array.shape
        qimage =  QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:
            # RGB image
            qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGB888)
        elif channels == 4:
            # RGBA image
            qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGBA8888)
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    return qimage