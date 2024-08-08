
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv, matplotlib as mpl, time, pyclesperanto_prototype as cle
from image_processing.canvas import ImageGraphicsView
import ui.app
# STARDIST
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stardist.models import StarDist2D
from csbdeep.utils import normalize

class StarDist(QObject):
    stardistDone = pyqtSignal(QPixmap)
    sendGrayScale = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.np_channels = None
        self.np_image = None
        self.params = {
        'channel': 'Channel 1',
        'model': '2D_versatile_fluo',
        'percentile_low' : 1.00,
        'percentile_high': 99.80,
        'prob_threshold': 0.48,
        'nms_threshold': 0.3,
        'n_tiles': 0,
        'radius': 5,
        }   

    def adjust_contrast(self, img, min=2, max = 98):
            # pixvals = np.array(img)

            minval = np.percentile(img, min) # room for experimentation 
            maxval = np.percentile(img, max) # room for experimentation 
            img = np.clip(img, minval, maxval)
            img = ((img - minval) / (maxval - minval)) * 255
            return (img.astype(np.uint8))
            
    def runStarDist(self):
        
        model = StarDist2D.from_pretrained(str(self.params['model']))

        try:
            # case: image has only one channel
            if not self.np_channels:
                arr = self.np_image
            # case: image has multiple channels
            else:
                arr = self.np_channels[self.params['channel']] 

            # scale down image if it's a large image
            scaleDown = arr.shape[0] > 10000

            if scaleDown:
                scale_factor = 1
                cell_image = cv.resize(arr, (0, 0), fx = 1 / scale_factor , fy = 1 / scale_factor)
            else:
                cell_image = arr
                            

            if self.params['n_tiles'] == 0:
                self.params['n_tiles'] = 10
                stardist_labels, _ = model.predict_instances(normalize(self.adjust_contrast(cell_image), self.params['percentile_low'], self.params['percentile_high']), 
                                                             prob_thresh=self.params['prob_threshold'], 
                                                             nms_thresh=self.params['nms_threshold'],
                                                             n_tiles=(self.params['n_tiles'], self.params['n_tiles']))
            else:
                stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), 
                                                             prob_thresh=self.params['prob_threshold'], 
                                                             nms_thresh=self.params['nms_threshold'], 
                                                             n_tiles =self.params['n_tiles'])

            # size it back to original
            if scaleDown:
                # cv resize takes uint8 or uint16, can't do uint32

                normalized_image = cv.normalize(stardist_labels, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
                stardist_labels = cv.resize(normalized_image, (0, 0), fx = scale_factor , fy = scale_factor, interpolation=cv.INTER_NEAREST)

            # dilate
            radius = self.params['radius']

            start_time = time.time()  

            print("dilating...")
            stardist_labels_grayscale = np.array(cle.dilate_labels(stardist_labels, radius=radius), dtype=np.uint8)

            print("generating lut...")
            lut = self.generate_lut("viridis")

            print("converting label to rgb...")
            # stardist_labels_rgb = self.label2rgb(stardist_labels_grayscale, lut).astype(np.uint8)
            stardist_labels_rgb = stardist_labels_grayscale
            end_time = time.time()  
            print(start_time - end_time)
            # convert to pixmap
            stardist_qimage = self.numpy_to_qimage(stardist_labels_rgb)
            stardist_pixmap = QPixmap(stardist_qimage)
            self.stardistDone.emit(stardist_pixmap)
            self.sendGrayScale.emit(stardist_labels_grayscale)

        except AttributeError: # should probably start defining custom exceptions
            QMessageBox.critical(ui.app.Ui_MainWindow(), "Error", "Empty canvas, please an load image first")
    
    # only uint8

    def change_cmap(self):
        pass
    
    def generate_lut(self, cmap:str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv.LUT(cv.merge((labels, labels, labels)), lut)

    def normalize_to_uint8(self, data: np.ndarray) -> QImage:
        normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        normalized_data = normalized_data.astype(np.uint8)
        return normalized_data

    def updateChannels(self,_, channels):
        self.np_image = None
        self.np_channels = channels

    def numpy_to_qimage(self, array:np.ndarray) -> QImage:
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

    # def qimage_to_numpy(self, qimage:QImage):
    #     # Ensure the QImage format is suitable for conversion
    #     if qimage.format() == QImage.Format.Format_Grayscale8:
    #         width = qimage.width()
    #         height = qimage.height()
    #         ptr = qimage.constBits()
    #         ptr.setsize(qimage.sizeInBytes())  # Ensure the pointer size matches the image byte count
            
    #         # Convert QImage to a 2D numpy array
    #         arr = np.ndarray(shape=(height, width), buffer=ptr, dtype=np.uint8)
    #         return arr
    #     else:
    #         raise ValueError("Unsupported QImage format for conversion to NumPy array")
        
    def setImageToProcess(self, np_image):
        self.np_channels = None
        self.np_image = np_image

    def setChannel(self, channel):
        self.params['channel'] = channel

    def setModel(self, model):
        self.params['model'] = model

    def setPercentileLow(self, value):
        self.params['percentile_low'] = value

    def setPercentileHigh(self, value):
        self.params['percentile_high'] = value

    def setProbThresh(self, value):
        self.params['prob_threshold'] = value

    def setNMSThresh(self, value):
        self.params['nms_threshold'] = value

    def setNumberTiles(self, value):
        self.params['n_tiles'] = value

    def setDilationRadius(self, value):
        self.params['radius'] = value
