from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap,  QCursor, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
import tifffile as tiff, numpy as np
# from PIL import Image, ImageSequence
import cv2, matplotlib as mpl
import time
from qt_threading import Worker
from utils import numpy_to_qimage, normalize_to_uint8, scale_adjust, adjustContrast, qimage_to_numpy


class ImageType:
    def __init__(self, name: str, arr):
        self.name = name
        self.arr = arr

class ReferenceGraphicsView(QGraphicsView):
    referenceViewAdded = pyqtSignal(QGraphicsPixmapItem)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.reset_referencepixmap=None
        self.reset_reference_pixmapItem=None
        self.reference_pixmap=None
        self.reference_pixmapItem=None

        self.setMinimumSize(QSize(300, 300))
        self.setObjectName("reference_canvas")
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 200, 150)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")


# drag and drog has issue with some tiff images, need to fix
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path:
                    self.addImage(file_path)
            event.acceptProposedAction()

    def addImage(self, file_path):
        # check if canvas already has an image
        self.resetTransform()
        if self.reference_pixmapItem:
            self.deleteImage() 
        with tiff.TiffFile(file_path) as tif:
            image = tif.pages[0].asarray()

        arr = scale_adjust(np.array(image).astype(np.uint16))
        arr = adjustContrast(arr, alpha=20, beta=35)

        if arr.shape[0] > 20000:
            thumbnail_size = (int(arr.shape[1]/50), int(arr.shape[0]/50))
            arr = cv2.resize(arr, thumbnail_size, interpolation=cv2.INTER_AREA)
        else: pass
        qimage = numpy_to_qimage(arr)
        self.reference_pixmapItem = QGraphicsPixmapItem(QPixmap(qimage))
        self.referenceViewAdded.emit(self.reference_pixmapItem)

        
    def deleteImage(self):
        self.scene().clear()


##########################################################
class ImageGraphicsView(QGraphicsView):
    
    canvasUpdated = pyqtSignal(QGraphicsPixmapItem)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    channelLoaded = pyqtSignal(dict, bool)
    channelNotLoaded = pyqtSignal(np.ndarray)
  
    # need some dead code analysis
    saveImage = pyqtSignal(QGraphicsPixmapItem)  

    updateProgress = pyqtSignal(int, str)


    def __init__(self):

        super().__init__()
        self.channels= None
        # self.channel_selector_combobox = self.view.toolBar.channelSelector
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(QPixmap("icons/clicks.png").scaled(25,25, Qt.AspectRatioMode.KeepAspectRatio), 0,0)


    def toPixmapItem(self, data:QPixmap|np.ndarray):
        #convert pixmap to pixmapItem
        if type(data) == QPixmap:
            self.pixmap = data
        else:
            self.pixmap = QPixmap(numpy_to_qimage(data))
            
        self.pixmapItem = QGraphicsPixmapItem(self.pixmap)
        self.canvasUpdated.emit(self.pixmapItem)
        
    def change_cmap(self, cmap_text: str):
        print("generating lut...")
        lut = self.generate_lut(cmap_text)
        print("converting label to rgb...")
        stardist_labels_rgb = self.label2rgb(self.stardist_labels, lut).astype(np.uint8)
        # self.progress.emit(99, "converting to rgb")
        # convert to pixmap
        self.toPixmapItem(stardist_labels_rgb)
    
    def generate_lut(self, cmap:str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(mpl.colormaps[cmap](label_range)[:,2::-1]*256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv2.LUT(cv2.merge((labels, labels, labels)), lut)
    

    def loadStardistLabels(self, stardist: ImageType):
        self.stardist_labels = stardist.arr
        self.toPixmapItem(self.stardist_labels)
    
    def addImage(self, file:str):
        '''add a new image'''

        self.channels = None
        self.resetTransform()

        if isinstance(file, str):

           self.image_worker = Worker(self.__filename_to_pixmap, file)
           self.image_worker.signal.connect(self.onFileNameToPixmapCompleted)
           self.image_worker.error.connect(self.onError)
           self.image_worker.start()


    @pyqtSlot(object)
    def onFileNameToPixmapCompleted(self, pixmap):
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem)



    def __filename_to_pixmap(self, file_name:str):  

        t = time.time()

        if file_name.endswith((".tiff", ".tif")):
            
            pages = self.read_tiff_pages(file_name)
            num_channels = len(pages)
            format = QImage.Format.Format_Grayscale8

            if (num_channels > 1):

                # resultSize = QSize(width, height)
                # self.resultImage = QImage(resultSize, QImage.Format.Format_ARGB32_Premultiplied)

                self.channels = {}
                self.np_channels = {}
                for channel_num, image in enumerate(pages):

                    channel_name = f'Channel {channel_num + 1}'
                    height, width = image.shape
                
                    __scaled = scale_adjust(image) 
                    image_adjusted = adjustContrast(__scaled) # uint8
               

                    bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                    self.np_channels[channel_name] = image_adjusted # for stardist and other image processing, maybe consider keeping it as uint16

                    qimage_channel = QImage(image_adjusted, width, height, width*bytesPerPixel, format)
                    self.channels[channel_name] = qimage_channel # for displaying on canvas

                    self.reset_np_channels = {key: img.copy() for key, img in self.np_channels.items()} #deep copy
                    self.reset_channels = {key: img.copy() for key, img in self.channels.items()} #deep copy

                    # self.updateProgress.emit(int((channel_num+1)/num_channels*100))

                channel_one_qimage = next(iter(self.channels.values()))
                self.channelLoaded.emit(self.np_channels, True)
            else:
                # determine if image is grayscale or rgb
                if pages[0].ndim == 2:
                    h, w = pages[0].shape
                elif pages[0].ndim == 3:
                    h, w, ch = pages[0].shape
                elif pages[0].ndim == 4:
                    h, w, ch, alpha = pages[0].shape
                else:
                    raise ValueError("image type not supported")

                __scaled = scale_adjust(pages[0]) 
                image_adjusted = adjustContrast(__scaled) # uint8
                print(image_adjusted.dtype)
                print(image_adjusted.shape)
                bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                if pages[0].ndim == 2:
                    channel_one_qimage = QImage(image_adjusted, w, h, w*bytesPerPixel, format)
                elif pages[0].ndim == 3:
                    channel_one_qimage = QImage(image_adjusted.tobytes(), w, h, QImage.Format.Format_RGB888)
                elif pages[0].ndim == 4:
                    channel_one_qimage = QImage(image_adjusted.tobytes(), w, h, QImage.Format.Format_RGBA8888)
                else:
                    raise ValueError("image type not supported")

                self.channelNotLoaded.emit(image_adjusted)

            pixmap = QPixmap(channel_one_qimage)

        else:
            pixmap = QPixmap(file_name)

        print(f"image loading time: {time.time() - t}")

        return pixmap
    
    def __numpy2QImageDict(self, numpy_channels_dict: dict) -> dict:
        return {key: numpy_to_qimage(arr) for key, arr in numpy_channels_dict.items()} 
    
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


    # def adjustContrast(self,img):  
        
    #     alpha = 5 # Contrast control
    #     beta = 15 # Brightness control
    #     return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


    # # uint16 to uint8
    # def scale_adjust(self, arr:np.ndarray):
    #     if arr.dtype == np.uint16:
    #         return cv2.convertScaleAbs(arr, alpha=(255.0/65535.0))
    #     elif arr.dtype == np.uint8:
    #         return arr
    #     else:
    #         print("unsupported array type")

        
    def deleteImage(self):
        self.scene().clear()

    def resetImage(self):
        if self.pixmapItem: 
            self.channels = self.reset_channels
            self.channelLoaded.emit(self.reset_np_channels, True)
            self.toPixmapItem(self.reset_pixmap)


    def rotate_image_task(self, channels:dict, angle):
        t = time.time()
        rotated_arrays = []
        print("rotation channel dtype", list(channels.values())[0].dtype)
        # import matplotlib.pyplot as plt
        # cv2.imshow("test", list(channels.values())[2])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # channels_arr = []
        # for i, channel in enumerate(channels.values()):
        #     print(f"Channel {i}: shape={channel.shape}, dtype={channel.dtype}, contiguous={channel.flags['C_CONTIGUOUS']}")
        #     channels_arr.append(np.ascontiguousarray(channel))
        # print("here")

        for channel in channels.values():
            print(channel.dtype)
            print(channel.shape)
            print(channel.data.contiguous)
            try:
                if not channel.data.contiguous:
                    print("converting to contiguous array")
                    channel = np.ascontiguousarray(channel, dtype='uint8')
            except Exception as e:
                print("error: ", str(e))
                
            height, width = channel.shape
            center = (int(width/2), int(height/2))
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_arr= cv2.warpAffine(channel, rotation_matrix, (width, height))
            rotated_arrays.append(rotated_arr)

        # transform = QTransform()
        # transform.rotate(angle)
        # rotated_images = [(image.transformed(transform, Qt.TransformationMode.SmoothTransformation)).convertToFormat(image.format()) for image in channels.values()]
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
            self.rotation_worker.start()

    @pyqtSlot(object)
    def onRotationCompleted(self, rotated_channels:dict):
        self.channels = rotated_channels[0]
        self.np_channels = rotated_channels[1]
        print("rotation curr channel num: ", self.currentChannelNum)
        channel_qimage = list(self.channels.values())[self.currentChannelNum]
        print(type(channel_qimage))
        channel_pixmap = QPixmap(channel_qimage)
        rotated_pixmapItem = QGraphicsPixmapItem(channel_pixmap)
        self.canvasUpdated.emit(rotated_pixmapItem)
        # self.np_channels = {key: qimage_to_numpy(img) for key, img in self.channels.items()} 
        self.channelLoaded.emit(self.np_channels, False)

    @pyqtSlot(str)
    def onError(self, error_message):
        print(f"Error: {error_message}")

    def updateChannels(self, channels:dict, clear:bool) -> None: #cropsignal will update this
        self.np_channels = channels #np arrays

        print("in updateChannels method of canvas.py")
        self.channels = self.__numpy2QImageDict(self.np_channels)
        print("clear channel?", clear)
        self.channelLoaded.emit(self.np_channels, clear)

    def swapChannel(self, index):
        channel_pixmap = QPixmap.fromImage(self.channels[f'Channel {index+1}'])
        self.toPixmapItem(channel_pixmap)

    def setCurrentChannel(self, index):
        self.currentChannelNum = index
