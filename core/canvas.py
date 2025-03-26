from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap,  QCursor, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QTimer
import tifffile as tiff, numpy as np
import cv2, matplotlib as mpl
import time
from core.Worker import Worker
from utils import numpy_to_qimage, normalize_to_uint8, scale_adjust, adjustContrast, qimage_to_numpy
from ui.Dialogs import ImageDialog

class ImageWrapper:
    def __init__(self, data, name="", cmap="gray"):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")
        
        self.name = name
        self.cmap = cmap
        self.data = data
        self.contrast_min = 0
        self.contrast_max = 255

## this needs to be deleted later
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
        self.reset_np_channels = {}

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
                        image_adjusted = image

                        if adjust_contrast:
                            __scaled = scale_adjust(image) 
                            image_adjusted = adjustContrast(__scaled) # uint8

                        bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                        format = QImage.Format.Format_Grayscale16 if image_adjusted.dtype == np.uint16 else QImage.Format.Format_Grayscale8

                        print("my dtype is", image_adjusted.dtype)
                        self.np_channels[channel_name] = ImageWrapper(image_adjusted) # for stardist and other image processing, maybe consider keeping it as uint16
                        self.reset_np_channels[channel_name] = ImageWrapper(image_adjusted.copy())  # keep another copy for resetting



                    channel_one_image = next(iter(self.np_channels.values())).data
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
        arr = self.np_channels["Channel 1"].data
        qimage = numpy_to_qimage(arr)
        self.pixmap =QPixmap(qimage)
        self.pixmapItem =  QGraphicsPixmapItem(self.pixmap)
        self.referenceViewAdded.emit(self.pixmapItem)

##########################################################
class ImageGraphicsView(__BaseGraphicsView):
    
    canvasUpdated = pyqtSignal(QPixmap)
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
        self.currentChannelNum = None
        # self.qimage_channels = {}

    def toPixmapItem(self, data:QPixmap|np.ndarray|QImage):
        #convert pixmap to pixmapItem
        if type(data) == QPixmap:
            self.pixmap = data
            print("data is a pixmap")
        elif type(data) == QImage:
            print("data is a QImage")
            self.pixmap = QPixmap(data)
        else:
            print("data is a numpy array")
            self.pixmap = QPixmap(numpy_to_qimage(data))
    
        self.canvasUpdated.emit(self.pixmap)
    
    update_cmap = pyqtSignal(str)
    def change_cmap(self, cmap_text: str):
        '''updates the current image using the current colormap and contrast settings'''
        lut = self.generate_lut(cmap_text)
        adjusted_uint8 = scale_adjust(self.image)
        channel_num = f"Channel {self.currentChannelNum + 1}"
        self.np_channels[channel_num].cmap = cmap_text
        rgb = self.label2rgb(adjusted_uint8, lut).astype(np.uint8)
        self.toPixmapItem(rgb)

        min, max = self.np_channels[channel_num].contrast_min, self.np_channels[channel_num].contrast_max
        self.update_contrast((min, max))
        self.update_cmap.emit(cmap_text)

    def generate_lut(self, cmap:str):
        '''generate a 8 bit look-up table and converts to rgb space'''
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        '''applys the look-up table and merges r, g, b channels to form colored image '''
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
        '''handles operation after the file is loaded into the canvas'''
        if image.dtype != np.uint8:
            image = scale_adjust(image)

        qimage = numpy_to_qimage(image)
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
        '''resets the image to original state'''
        if self.pixmapItem: 

            import copy
            self.np_channels = copy.deepcopy(self.reset_np_channels)
            self.channelLoaded.emit(self.np_channels, False)

            channel_num = f"Channel {self.currentChannelNum + 1}"
            self.image = self.np_channels.get(channel_num).data

            layered_data = self.reset_np_channels.get(channel_num).data
            self.reset_pixmap = QPixmap(numpy_to_qimage(layered_data))
            self.toPixmapItem(self.reset_pixmap)

    def rotate_image_task(self, channels:dict, angle):
        t = time.time()
        rotated_arrays = []

        for wrapper in channels.values():
            try:
                arr = wrapper.data
                cmap = wrapper.cmap
                if not arr.data.contiguous:
                    arr = np.ascontiguousarray(arr, dtype='uint16')
            except Exception as e:
                print("error: ", str(e)) # should be a QMessageBox

            # rotate image
            h,w = arr.shape
            center = (w/2, h/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            cos = np.abs(rotation_matrix[0,0])
            sin = np.abs(rotation_matrix[0,1])
            updated_w = int((h*sin) + (w*cos))
            updated_h = int((h*cos) + (w*sin))
            rotation_matrix[0,2] += (updated_w/2) - w/2
            rotation_matrix[1,2] += (updated_h/2) - h/2
            rotated_arr = cv2.warpAffine(arr, rotation_matrix, (updated_h, updated_h))

            self.rotated_wrapper = ImageWrapper(rotated_arr, cmap = cmap)

            #append to rotated array list
            rotated_arrays.append(self.rotated_wrapper)
        # convert to qimage
        print(time.time()-t)
        return  dict(zip(channels.keys(), rotated_arrays))
    def rotateImage(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.") # this should be a QMessageBox
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
        self.np_channels = rotated_channels

        channel_image = list(self.np_channels.values())[self.currentChannelNum].data
        channel_cmap = list(self.np_channels.values())[self.currentChannelNum].cmap
        channel_image = scale_adjust(channel_image)
        self.image = channel_image
        # channel_qimage = numpy_to_qimage(channel_image)
        # channel_pixmap = QPixmap(channel_qimage)
        # self.canvasUpdated.emit(channel_pixmap)
        self.change_cmap(channel_cmap) # this also updates the contrast
        self.channelLoaded.emit(self.np_channels, False)

    @pyqtSlot(str)
    def onError(self, error_message):
        print(f"Error: {error_message}")

    def updateChannels(self, channels:dict, clear:bool) -> None: #cropsignal will update this
        self.np_channels = channels # replace channels with new, cropped/rotated, etc
        self.channelLoaded.emit(self.np_channels, clear)


    def updateCurrentImage(self, data_dict):
        self.image = data_dict[f"Channel {self.currentChannelNum + 1}"].data


    def swapChannel(self, index):
        '''swaps between channels of a multi-layered tiff image'''
        channel_num = f'Channel {index+1}'
        self.currentChannelNum = index

        self.image = self.np_channels.get(channel_num).data # self.image always needs to be updated. this is the current image that is being operated on
        
        if channel_num in self.np_channels.keys():
            self.change_cmap(self.np_channels.get(channel_num).cmap) # this also updates the contrast



    def update_contrast(self, values):

        if self.pixmap is None:
            self.errorSignal.emit("Canvas empty")
            return
        
        min_val, max_val = values
        channel_num = f"Channel {self.currentChannelNum + 1}"
        self.np_channels.get(channel_num).contrast_min = min_val
        self.np_channels.get(channel_num).contrast_max = max_val


        contrast_image = self.apply_contrast(min_val, max_val)
        
        contrastPixmap = QPixmap(numpy_to_qimage(contrast_image))

        self.canvasUpdated.emit(contrastPixmap)

        self.changeSlider.emit((min_val, max_val))


    def auto_contrast(self, lower = 0.1, upper=.9):
        channel_num = f"Channel {self.currentChannelNum + 1}"
        channel = scale_adjust(self.np_channels[channel_num].data)


        flat_channel = channel.flatten()
    
        hist, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
        total_pixels = flat_channel.size
        cumulative_hist = np.cumsum(hist) / total_pixels
        new_min= np.argmax(cumulative_hist > lower)  
        new_max = np.argmax(cumulative_hist > upper)  

        self.update_contrast((new_min, new_max))

    def apply_contrast(self, new_min, new_max):

        qimage = self.pixmap.toImage() # get current image
        image = qimage_to_numpy(qimage) # returns uint8
        lut = self.create_lut(new_min, new_max)
        return cv2.LUT(image, lut)

    def create_lut(self, new_min, new_max):

        lut = np.zeros(256, dtype=np.uint8) # uint8 for display
        lut[new_min:new_max+1] = np.linspace(start=0, stop=255, num=(new_max - new_min + 1), endpoint=True, dtype=np.uint8)
        lut[:new_min] = 0 # clip between 0 and 255
        lut[new_max+1:] = 255

        return lut
    
    # def set_blur_layer(self, layer):
    #     self._blur_layer = layer

    # def set_blur_percentage(self, blur_percentage):
    #     self._blur_percentage = blur_percentage
    #     print("blur percentage: ", self._blur_percentage)

    def blur_layer(self, blur_percentage:float, confirm=False):
        """
        Applies Gaussian blur chosen of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """
        self._blur_layer = f'Channel {self.currentChannelNum+ 1}'
        if not confirm:
            
            # blur_percentage = self._blur_percentage
            layer_to_blur = self.np_channels[self._blur_layer].data

            blurred_mask = cv2.GaussianBlur(layer_to_blur, (101, 101), 0)
            blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
            self.corrected_layer = cv2.subtract(layer_to_blur, blurred_mask_adjusted)
            self.corrected_layer = np.clip(self.corrected_layer, 0, 65535).astype(np.uint16)
            cmap = self.np_channels[self._blur_layer].cmap
            self.image = self.corrected_layer
            self.change_cmap(cmap)

        else:
            print("Error from gaussian blur")


        if confirm and hasattr(self, "corrected_layer") and (not self.np_channels.get(self._blur_layer) == None):
            self.np_channels[self._blur_layer].data = self.corrected_layer # Replace with the corrected version
            self.channelLoaded.emit(self.np_channels, False)
            self.updateProgress.emit(100, f"Replaced {self._blur_layer}")
            

    def showCroppedImage(self, image_rect):
        """Show dialog with cropped image preview"""
        pixmap = self.pixmap 
        cropped = pixmap.copy(image_rect).toImage()
        cropped_pixmap = QPixmap(cropped)
        self.crop_dialog = ImageDialog(self, cropped_pixmap)
        self.crop_dialog.exec()

        if self.crop_dialog.confirm_crop:
            self.crop_worker = Worker(self.cropImageTask, image_rect)
            self.crop_worker.signal.connect(self.onCropCompleted) 
            self.crop_worker.finished.connect(self.crop_worker.quit)
            self.crop_worker.finished.connect(self.crop_worker.deleteLater)
            self.crop_worker.start()
        else:
            self.cropSignal.emit(False)

    cropSignal = pyqtSignal(bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_wrappers: dict):
        """Handle completed crop operation"""
        self.np_channels = cropped_wrappers
        channel_num = f"Channel {self.currentChannelNum + 1}"
        self.image = self.np_channels.get(channel_num).data
        self.cropSignal.emit(False)

    def cropImageTask(self, image_rect) -> dict:
        """Process crop in background thread"""
        left = image_rect.x()
        top = image_rect.y()
        right = image_rect.right()
        bottom = image_rect.bottom()        

        for channel_name, image_arr in self.np_channels.items():
            arr = image_arr.data
            cmap = image_arr.cmap
            cropped_array = arr[top:bottom+1, left:right+1]
            if not cropped_array.data.contiguous:
                cropped_array = np.ascontiguousarray(cropped_array, dtype="uint16")

            if not hasattr(self, "cropped_wrappers"):
                self.cropped_wrappers = {}

            self.crop_wrapper = ImageWrapper(cropped_array, cmap=cmap)
            min, max = self.np_channels[channel_name].contrast_min, self.np_channels[channel_name].contrast_max
            self.crop_wrapper.contrast_min = min
            self.crop_wrapper.contrast_max = max
            self.update_contrast((min, max))
            self.cropped_wrappers[channel_name] = self.crop_wrapper

        return self.cropped_wrappers
    
    


    



