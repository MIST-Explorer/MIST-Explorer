from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import  QImage, QPixmap
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot
import Dialogs, tifffile as tiff, numpy as np
from PIL import Image, ImageSequence
import cv2
import time



def numpy_to_qimage(array:np.ndarray) -> QImage:

    if not array.data.contiguous:
        array = np.ascontiguousarray(array)

    if len(array.shape) == 2:
        # Grayscale image
        height, width = array.shape
        format = QImage.Format.Format_Grayscale16 if array.dtype == np.uint16 else QImage.Format.Format_Grayscale8
        bytes_per_pixel = 2 if array.dtype == np.uint16 else 1
        bytes_per_line = width * bytes_per_pixel#uint8
        print("converting to grayscale")
        print("is array contiguous?: ", array.data.contiguous)
        qimage =  QImage(array.data, width, height, bytes_per_line, format)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:
            # RGB image
            qimage = QImage(array.data, width, height, width*channels, QImage.Format.Format_RGB888)
        elif channels == 4:
            # RGBA image
            qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGBA8888)
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    return qimage.copy()


def qimage_to_numpy(qimage:QImage):
    # Ensure the QImage format is suitable for conversion
    print("checking format: ", qimage.format())

    width = qimage.width()
    height = qimage.height()

    if qimage.format() == QImage.Format.Format_Grayscale8:
        ptr = qimage.bits()
        ptr.setsize(width * height)
        strides = [qimage.bytesPerLine(), 1]
        dtype = np.uint8
    elif qimage.format() == QImage.Format.Format_Grayscale16:
        ptr = qimage.bits()
        ptr.setsize(width * height)
        strides = [qimage.bytesPerLine(), 2]
        dtype = np.uint16

    else:
        raise ValueError("Image type not supported")
    
    arr = np.ndarray((height, width), buffer=ptr, strides=strides, dtype=dtype)

    return arr    
 
    

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


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data = normalized_data.astype(np.uint8)
    return normalized_data


def adjustContrast(img, alpha=5, beta=15):  
    
    # alpha = 5 # Contrast control
    # beta = 15 # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# uint16 to uint8
def scale_adjust(arr:np.ndarray):
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0/65535.0))
    elif arr.dtype == np.uint8:
        return arr
    else:
        print("unsupported array type")


def recolor(arr):
    pass



class Pixmap(QPixmap):
    def __init__(self, pixmap_type, qimage=None, *args):
        super().__init__(*args)  # Call QPixmap's constructor
        self.type = pixmap_type  # Store the custom type
        
        if qimage is not None:
            # Convert QImage to QPixmap if provided
            self.convertFromImage(qimage)
