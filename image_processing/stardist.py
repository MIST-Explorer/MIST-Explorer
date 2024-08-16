
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv, matplotlib as mpl, time, pyclesperanto_prototype as cle
from image_processing.canvas import ImageGraphicsView
import ui.app
from utils import numpy_to_qimage
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

    def runStarDist(self):

        model = StarDist2D.from_pretrained(str(self.params['model']))

        try:
            # case: image has only one channel
            if not self.np_channels:
                arr = self.np_image
            # case: image has multiple channels
            else:
                arr = self.qimage_to_numpy(self.channels[self.params['channel']])
                # arr = self.np_channels[self.params['channel']] 

            # scale down image if it's a large image
            scaleDown = arr.shape[0] > 10000

            if scaleDown:
                scale_factor = 10
                cell_image = cv.resize(arr, (0, 0), fx = 1 / scale_factor , fy = 1 / scale_factor)
            else:
                cell_image = arr
                            

            if self.params['n_tiles'] == 0:
                stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), 
                                                             prob_thresh=self.params['prob_threshold'], 
                                                             nms_thresh=self.params['nms_threshold'])
            else:
                stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), 
                                                             prob_thresh=self.params['prob_threshold'], 
                                                             nms_thresh=self.params['nms_threshold'], 
                                                             n_tiles =(self.params['n_tiles'], (self.params['n_tiles'])))

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
            stardist_labels_rgb = self.label2rgb(stardist_labels_grayscale, lut).astype(np.uint8)
            end_time = time.time()  
            print(start_time - end_time)
            # convert to pixmap
            stardist_qimage = numpy_to_qimage(stardist_labels_rgb)
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

    def updateChannels(self, channels, np_channels):
        self.np_image = None
        self.np_channels = np_channels
        self.channels = channels
        
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
