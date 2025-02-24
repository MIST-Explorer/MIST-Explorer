from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap,  QCursor, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QTimer
import tifffile as tiff, numpy as np
import cv2, matplotlib as mpl
import time
from core.Worker import Worker
from utils import numpy_to_qimage, normalize_to_uint8, scale_adjust, adjustContrast, qimage_to_numpy


class ImageType:
    def __init__(self, name: str, arr):
        self.name = name
        self.arr = arr

class __BaseGraphicsView(QWidget):
    '''base class for graphics view'''
    channelLoaded = pyqtSignal(dict, bool)
    channelNotLoaded = pyqtSignal(np.ndarray)
    updateProgress = pyqtSignal(int, str)
    errorSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.setMinimumSize(QSize(300, 300))
        self.scene = QGraphicsView()
        self.scene.setScene(QGraphicsScene(self))
        # self.channels= None
        self.reset_pixmap =  None
        self.reset_pixmapItem = None
        self.pixmap = None
        self.pixmapItem=None
        self.np_channels = {}

    # drag and drog has issue with some tiff images, need to fix
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        event.acceptProposedAction()

    def read_tiff_pages(self, file_path):
        pages = []
        with tiff.TiffFile(file_path) as tif:
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # Check if the image is blank
                    if np.all(image == image.flat[0]):
                        print(f"Page {i} is blank, skipping.")
                        continue
                    pages.append(image)
                    self.updateProgress.emit(int((i+1)/len(tif.pages)*100), "Loading image")
                except Exception as e:
                    print(f"Error reading page {i}: {e}")
                    continue
        return pages
    
    def filename_to_image(self, file_name:str, adjust_contrast=False) -> np.ndarray:  

            t = time.time()

            if file_name.endswith((".tiff", ".tif")):
                pages = self.read_tiff_pages(file_name)
                num_channels = len(pages)

                if (num_channels > 1):

                    for channel_num, image in enumerate(pages):

                        channel_name = f'Channel {channel_num + 1}'
                        height, width = image.shape
                        image_adjusted =image

                        if adjust_contrast:
                            __scaled = scale_adjust(image) 
                            image_adjusted = adjustContrast(__scaled) # uint8

                        bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                        format = QImage.Format.Format_Grayscale16 if image_adjusted.dtype == np.uint16 else QImage.Format.Format_Grayscale8

                        print("my dtype is", image_adjusted.dtype)
                        self.np_channels[channel_name] = image_adjusted # for stardist and other image processing, maybe consider keeping it as uint16

                        # qimage_channel = QImage(image_adjusted, width, height, width*bytesPerPixel, format)
                        # self.channels[channel_name] = qimage_channel # for displaying on canvas

                        self.reset_np_channels = {key: img.copy() for key, img in self.np_channels.items()} #deep copy
                        # self.reset_channels = {key: img.copy() for key, img in self.channels.items()} #deep copy

                    channel_one_image = next(iter(self.np_channels.values()))
                    self.channelLoaded.emit(self.np_channels, True)
                else: # num of channels is 1, single page
                    print("not multilayer")
                    print(pages[0].shape)
                    channel_one_image = pages[0]
                    self.channelNotLoaded.emit(channel_one_image)
            else: # not a .tif image
                from PIL import Image
                print("not a tif")
                channel_one_image = np.array(Image.open(file_name))
                
                print(channel_one_image.shape)

            print(f"image loading time: {time.time() - t}")

            self.updateProgress.emit(100, "Image Loaded")
            return channel_one_image
    
    def deleteImage(self):
        self.scene.scene().clear()


class ReferenceGraphicsView(__BaseGraphicsView):

    referenceViewAdded = pyqtSignal(QGraphicsPixmapItem)
    referenceLoaded = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)


    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path:
                    self.addImage(file_path)
            event.acceptProposedAction()
    

    def addImage(self, file_path:str):

        self.reference_worker = Worker(self.filename_to_image, file_path, True)
        self.reference_worker.start()
        self.reference_worker.signal.connect(self.filename_to_image_complete)
        self.reference_worker.finished.connect(self.reference_worker.quit)
        self.reference_worker.finished.connect(self.reference_worker.deleteLater)

    def filename_to_image_complete(self):
        arr = self.np_channels["Channel 1"]
        qimage = numpy_to_qimage(arr)
        self.pixmap =QPixmap(qimage)
        self.pixmapItem =  QGraphicsPixmapItem(self.pixmap)
        self.referenceViewAdded.emit(self.pixmapItem)

##########################################################
class ImageGraphicsView(__BaseGraphicsView):
    
    canvasUpdated = pyqtSignal(QGraphicsPixmapItem)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    # need some dead code analysis
    saveImage = pyqtSignal(QGraphicsPixmapItem)  
    changeSlider = pyqtSignal(tuple)

    def __init__(self, parent=None):

        super().__init__(parent)
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(Qt.CursorShape.CrossCursor)
        # self.contrast_worker_running = False 
        # self.timer = QTimer()
        self.contrast_worker = None
        self.qimage_channels = {}

    def toPixmapItem(self, data:QPixmap|np.ndarray|QImage):
        #convert pixmap to pixmapItem
        if type(data) == QPixmap:
            self.pixmap = data
        elif type(data) == QImage:
            self.pixmap = QPixmap(data)
        else:
            self.pixmap = QPixmap(numpy_to_qimage(data))
            
        if hasattr(self, 'pixmapItem') and self.pixmapItem:
            self.pixmapItem.setPixmap(self.pixmap)  # update the pixmap of the existing item
        else:
            self.pixmapItem = QGraphicsPixmapItem(self.pixmap)  # create a new item if it doesn't exist
        
        print("changing slider")
        if not self.image is None and not self.image.dtype == np.uint8:
            print("debug here")
            image_uint8 = scale_adjust(self.image)
        else:
            raise ValueError("there was an error")

        self.changeSlider.emit((image_uint8.min(), image_uint8.max()))
        
        self.canvasUpdated.emit(self.pixmapItem)
        
    def change_cmap(self, cmap_text: str):
        print("generating lut...")
        lut = self.generate_lut(cmap_text)
        print("converting label to rgb...")
        adjusted_uint8 = scale_adjust(self.image)
        # if adjusted_uint8.shape[2] >= 3:
        #     r = adjusted_uint8[:,:,0]
        #     g = adjusted_uint8[:,:,1]
        #     b = adjusted_uint8[:,:,2]

        rgb = self.label2rgb(adjusted_uint8, lut).astype(np.uint8)
        # self.progress.emit(99, "converting to rgb")
        # convert to pixmap
        self.toPixmapItem(rgb)
    
    def generate_lut(self, cmap:str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):

        if len(labels) == 3:
            r,g,b = labels
            return cv2.LUT(cv2.merge((r, g, b)), lut)
        else:
            return cv2.LUT(cv2.merge((labels, labels, labels)), lut) # gray to color
    

    def loadStardistLabels(self, stardist: ImageType):
        self.stardist_labels = stardist.arr
        self.image = self.stardist_labels.copy()
        self.toPixmapItem(self.image)
    
    def addImage(self, file:str):
        '''add a new image'''

        self.qimage_channels.clear()
        self.np_channels.clear()
        self.scene.resetTransform()

        if isinstance(file, str):

           self.image_worker = Worker(self.filename_to_image, file)
           self.image_worker.signal.connect(self.onFileNameToPixmapCompleted)
           self.image_worker.error.connect(self.onError)
           self.image_worker.finished.connect(self.image_worker.quit)
           self.image_worker.start()


    @pyqtSlot(object)
    def onFileNameToPixmapCompleted(self, image):
        if image.dtype != np.uint8:
            image = scale_adjust(image)

        qimage = numpy_to_qimage(image)
        print("file", qimage.format())
        pixmap = QPixmap(qimage)
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem) # emit uint16, change to uint8 in canvas_ui

    
    def pixmap_to_numpy_array(self):
        # Convert the QPixmap to a QImage
        image = self.pixmap.toImage()

        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to byte array
        ptr = image.bits()
        ptr.setsize(height * width * 4)

        # Convert byte array to numpy array
        array = np.array(ptr).reshape((height, width, 4))

        return array

    def resetImage(self):
        if self.pixmapItem: 
            # self.channels = self.reset_channels
            self.np_channels = self.reset_np_channels
            self.channelLoaded.emit(self.np_channels, True)
            self.toPixmapItem(self.reset_pixmap)

    def rotate_image_task(self, channels:dict, angle):
        t = time.time()
        rotated_arrays = []
        print("rotation channel dtype", list(channels.values())[0].dtype)

        for channel in channels.values():
            try:
                if not channel.data.contiguous:
                    print("converting to contiguous array")
                    channel = np.ascontiguousarray(channel, dtype='uint8')
            except Exception as e:
                print("error: ", str(e))

            # rotate image
            h,w = channel.shape
            center = (w/2, h/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            cos = np.abs(rotation_matrix[0,0])
            sin = np.abs(rotation_matrix[0,1])
            updated_w = int((h*sin) + (w*cos))
            updated_h = int((h*cos) + (w*sin))
            rotation_matrix[0,2] += (updated_w/2) - w/2
            rotation_matrix[1,2] += (updated_h/2) - h/2
            rotated_arr = cv2.warpAffine(channel, rotation_matrix, (updated_h, updated_h))

            #append to rotated array list
            rotated_arrays.append(rotated_arr)
        # convert to qimage
        rotated_images = [numpy_to_qimage(array) for array in rotated_arrays]
        print(time.time()-t)
        return dict(zip(channels.keys(), rotated_images)), dict(zip(channels.keys(), rotated_arrays))

    def rotateImage(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.")
            return

        if self.pixmap and angle is not None:
            self.rotation_worker = Worker(self.rotate_image_task, self.np_channels, angle)
            self.rotation_worker.signal.connect(self.onRotationCompleted) # result is rotated_channels
            self.rotation_worker.error.connect(self.onError)
            self.rotation_worker.finished.connect(self.rotation_worker.quit)
            self.rotation_worker.finished.connect(self.rotation_worker.deleteLater)
            self.rotation_worker.start()

    @pyqtSlot(object)
    def onRotationCompleted(self, rotated_channels:dict):
        self.np_channels = rotated_channels[1]
        print("rotation curr channel num: ", self.currentChannelNum)

        channel_image = list(self.np_channels.values())[self.currentChannelNum]
        channel_image = scale_adjust(channel_image)
        channel_qimage = numpy_to_qimage(channel_image)
        print(type(channel_qimage))
        channel_pixmap = QPixmap(channel_qimage)
        rotated_pixmapItem = QGraphicsPixmapItem(channel_pixmap)
        self.canvasUpdated.emit(rotated_pixmapItem)
        self.channelLoaded.emit(self.np_channels, False)

    @pyqtSlot(str)
    def onError(self, error_message):
        print(f"Error: {error_message}")

    def updateChannels(self, channels:dict, clear:bool) -> None: #cropsignal will update this

        print("updated channels in canvas.py after cropsignal")
        self.np_channels = channels # replace channels with new, cropped/rotated, etc
        self.qimage_channels = {k: numpy_to_qimage(v) for k, v in channels.items()}
  
        self.channelLoaded.emit(self.np_channels, clear)

    def swapChannel(self, index):
        
        channel_num = f'Channel {index+1}'
        self.image = self.np_channels[channel_num]
        if channel_num in self.qimage_channels.keys():
            qimage = self.qimage_channels[channel_num]
            channel_pixmap = QPixmap.fromImage(qimage)
        else:
            qimage = numpy_to_qimage(self.image)
            self.qimage_channels[channel_num] = qimage
            channel_pixmap = QPixmap.fromImage(qimage)

        self.toPixmapItem(channel_pixmap)

    def setCurrentChannel(self, index):
        self.currentChannelNum = index


    def update_contrast(self, values):

        if self.pixmap is None:
            self.errorSignal.emit("Canvas empty")
            return
        
        min_val, max_val = values
        # min_val = int((self.image.min()/65535) *255)
        # print(min_val)
        # max_val = int((self.image.max()/65535) *255)
        # print(max_val)

        # self.contrast_worker = Worker(self.apply_contrast, min_val, max_val)
        # self.contrast_worker.start()
        # self.contrast_worker.signal.connect(self.contrast_complete)

        contrast_image = self.apply_contrast(min_val, max_val)
        
        contrastPix = QGraphicsPixmapItem(QPixmap(numpy_to_qimage(contrast_image)))

        self.canvasUpdated.emit(contrastPix)
    # def contrast_complete(self, data):
    #     self.contrast_worker_running = False
    #     contrastPix = QGraphicsPixmapItem(QPixmap(numpy_to_qimage(data)))
    #     self.canvasUpdated.emit(contrastPix)
    #     self.contrast_worker.finished.connect(self.contrast_worker.wait)
    #     self.contrast_worker.finished.connect(self.contrast_worker.deleteLater)


    # def start_timer(self):
    #     # Stop the timer if it's already running to reset it
    #     if self.timer.isActive():
    #         print("stopping timer")
    #         self.timer.stop()
    #     # Start the timer with a delay
    #     self.timer.start(100)  # Delay in milliseconds

    def auto_contrast(self, lower = 0.1, upper=.9):
        num = self.currentChannelNum + 1
        channel = scale_adjust(self.np_channels[f"Channel {num}"])


        flat_channel = channel.flatten()
    
        hist, bin_edges = np.histogram(flat_channel, bins=256, range=(0, 255))
        total_pixels = flat_channel.size
        cumulative_hist = np.cumsum(hist) / total_pixels
        new_min= np.argmax(cumulative_hist > lower)  
        new_max = np.argmax(cumulative_hist > upper)  


        self.update_contrast((new_min, new_max))

    def apply_contrast(self, new_min, new_max):

        qimage = self.pixmap.toImage()
        image = qimage_to_numpy(qimage) # returns uint8
        lut = self.create_lut(new_min, new_max)
        # apply the look up table
        return cv2.LUT(image, lut)

    def create_lut(self, new_min, new_max):

        lut = np.zeros(256, dtype=np.uint8) 


        lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
        lut[:new_min] = 0 # clip between 0 and 255
        lut[new_max+1:] = 255

        return lut
    

    def set_blur_layer(self, layer):
        self._blur_layer = layer

    def set_blur_percentage(self, blur_percentage):
        self._blur_percentage = blur_percentage

    def blur_layer(self):
        """
        Applies Gaussian blur to the 4th layer (index 3) of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """
        self.updateProgress.emit(0, "Starting to blur layer")
        blur_percentage = self._blur_percentage
        layer = self._blur_layer
        layer_key = f'Channel {layer}'
        layer_4 = self.np_channels[layer_key]
        self.updateProgress.emit(50, "blurring layer")
        blurred_mask = cv2.GaussianBlur(layer_4, (101, 101), 0)
        blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
        corrected_layer_4 = cv2.subtract(layer_4, blurred_mask_adjusted)
        corrected_layer_4 = np.clip(corrected_layer_4, 0, 65535).astype(np.uint16)
        self.np_channels[layer_key] = corrected_layer_4 # Replace the 4th layer with the corrected version
        self.channelLoaded.emit(self.np_channels, False)
        self.updateProgress.emit(100, "Done")
        

