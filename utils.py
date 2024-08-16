from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import  QImage, QPixmap
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


def qimage_to_numpy(qimage:QImage):
    # Ensure the QImage format is suitable for conversion
    print(qimage.format())
    if qimage.format() == QImage.Format.Format_Grayscale8:
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.constBits()
        ptr.setsize(qimage.sizeInBytes())  # Ensure the pointer size matches the image byte count
        
        # Convert QImage to a 2D numpy array
        arr = np.ndarray(shape=(height, width), buffer=ptr, dtype=np.uint8)
        return arr
    else:
        raise ValueError("Unsupported QImage format for conversion to NumPy array")
    

# this and qimage to numpy seems repetitive, delete one of them
def pixmap_to_image(pixmap: QPixmap):
    
    if pixmap == None:
        return None
    # Convert QPixmap to QImage
    qimage = pixmap.toImage()

    # Convert QImage to numpy array
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

    # Save numpy array as an image file using OpenCV
    return arr


def is_grayscale(image: np.ndarray) -> bool:

    if len(image.shape) == 3 and image.shape[2] == 3:
        return False
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return True
    else:
        raise ValueError("Image format not recognized")
    