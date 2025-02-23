
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal, QObject, pyqtSlot
from PyQt6.QtWidgets import QMessageBox
import numpy as np, cv2 as cv, matplotlib as mpl, time
from pyclesperanto_prototype import dilate_labels
from core.canvas import ImageGraphicsView, ImageType
import ui.app
from utils import numpy_to_qimage, qimage_to_numpy
from skimage.segmentation import expand_labels
from Worker import Worker
# STARDIST
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stardist.models import StarDist2D
from csbdeep.utils import normalize

class StarDist(QObject):
    stardistDone = pyqtSignal(ImageType)
    # sendGrayScale = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, str)
    errorSignal = pyqtSignal(str)
    

    def __init__(self):
        super().__init__()
        self.np_channels = None
        self.np_image = None
        self.params = {
        'channel': 'Channel 1',
        'model': '2D_versatile_fluo',
        'percentile_low' : 3,
        'percentile_high': 99.80,
        'prob_threshold': 0.48,
        'nms_threshold': 0.3,
        'n_tiles': 0,
        'radius': 5,
        }
        self.aligned = False

    def loadCellImage(self, arr):
        self.cell_image = arr
        self.aligned = True

    def runStarDist(self):
        print(self.np_channels, self.np_image)
        # case: image not loaded
        if self.np_channels is None and self.np_image is None:
            self.errorSignal.emit("please load image first")  # emit error message
            print("debug here")
            return
        elif self.np_channels and self.np_image:
            self.errorSignal.emit("unknown error, canvas has both single channel image and multi-channel image initiated")  # emit error message
            return
        
        import tensorflow as tf
        gpu = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu:
            device_name= tf.test.gpu_device_name()
            print("gpu name: ", device_name)
        else:
            device_name = '/CPU:0'

        with tf.device(device_name):
            self.stardist_worker = Worker(self.stardistTask)
            self.stardist_worker.start()
            
            self.stardist_worker.signal.connect(self.onStarDistCompleted)
    def getCellImage(self):
        if self.aligned:
            return self.cell_image
        # case: image has one channel
        elif self.np_channels is None and self.np_image:
            return self.np_image
        #
        elif self.np_channels and self.np_image is None:
            return self.np_channels[self.params['channel']] 
        
    def stardistTask(self):
        cell_image = self.getCellImage()

        # if self.aligned:
        #     self.setChannel()

        adjusted = cv.convertScaleAbs(cell_image, alpha=(255.0/65535.0))
    
        alpha = 5 # Contrast control
        beta = 15 # Brightness control
        adjusted = cv.convertScaleAbs(adjusted, alpha=alpha, beta=beta)
        cv.imshow('Image Window',adjusted)

        cv.waitKey(0)

        cv.destroyAllWindows()
        
        self.progress.emit(0, "Starting StarDist")
        model = StarDist2D.from_pretrained(str(self.params['model']))
                        
        self.progress.emit(25, "Training model")

        if self.params['n_tiles'] == 0:
            guess_tiles= model._guess_n_tiles(cell_image)
            # total_tiles = int(guess_tiles[0] * guess_tiles[1])
            # self.setNumberTiles(n_tiles)
            stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), 
                                                            prob_thresh=self.params['prob_threshold'], 
                                                            nms_thresh=self.params['nms_threshold'], n_tiles = guess_tiles)
            
        else:

            stardist_labels, _ = model.predict_instances(normalize(cell_image, self.params['percentile_low'], self.params['percentile_high']), 
                                                            prob_thresh=self.params['prob_threshold'], 
                                                            nms_thresh=self.params['nms_threshold'], 
                                                            n_tiles =(self.params['n_tiles'], (self.params['n_tiles'])))
            
        # dilate
        radius = self.params['radius']

        start_time = time.time()  

        print("dilating...")
        self.progress.emit(95, "Dilating")


        self.stardist_labels_grayscale = np.array(dilate_labels(stardist_labels, radius=radius), dtype=np.uint16)
        print("stardist type is", self.stardist_labels_grayscale.dtype)


        end_time = time.time()  
        print(start_time - end_time)

        self.progress.emit(100, "Stardist Done")
        return ImageType("stardist", self.stardist_labels_grayscale)
    
    def cancel(self):
        self.stardist_worker.terminate()


    def saveImage(self):
        from PIL import Image
        from PyQt6.QtWidgets import QFileDialog
        file_name, _ = QFileDialog.getSaveFileName(None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)")
        if not self.stardist_labels_grayscale is None:
            Image.fromarray(self.stardist_labels_grayscale).save(file_name)
        else:
            self.errorSignal.emit("Cannot save. No stardist labels available")
    # @pyqtSlot(int)
    # def updateProgress(self, num):
    #     self.progress.emit(num, f"Generating Tile {num}")
    
    # only uint8
    @pyqtSlot(ImageType)
    def onStarDistCompleted(self, stardist_result):
        self.stardistDone.emit(stardist_result)

    def change_cmap(self):
        pass
    
    def generate_lut(self, cmap:str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv.LUT(cv.merge((labels, labels, labels)), lut)

    def updateChannels(self, np_channels, _):
        self.np_image = None
        self.np_channels = np_channels
        
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
