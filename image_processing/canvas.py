from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage, QTransform
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal
import Dialogs, tifffile as tiff, numpy as np
from PIL import Image, ImageSequence
import cv2
import time

class ReferenceGraphicsView(QGraphicsView):
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
                pixmap = QPixmap(file_path) 
                if not pixmap.isNull():
                    self.addImage(pixmap)
            event.acceptProposedAction()

    def addImage(self, pixmap: QPixmap):
        # check if canvas already has an image
        self.resetTransform()
        if self.reference_pixmapItem:
            self.deleteImage() 
        # else 
        self.reset_reference_pixmap=pixmap
        self.reference_pixmap = pixmap
        self.reset_reference_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.reference_pixmapItem = QGraphicsPixmapItem(pixmap)

        # center the image
        self.scene().addItem(self.reference_pixmapItem)

        item_rect = self.reference_pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.reference_pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.reference_pixmapItem)

        
    def deleteImage(self):
        self.scene().clear()

    def resetImage(self):
        if self.reference_pixmapItem:
            self.reference_pixmapItem.setPixmap(self.reset_reference_pixmap)
            self.scene().update()


##########################################################
class ImageGraphicsView(QGraphicsView):
    
    canvasUpdated = pyqtSignal(QGraphicsPixmapItem)
    newImageAdded = pyqtSignal(QGraphicsPixmapItem)
    channelLoaded = pyqtSignal(dict, dict)
    channelNotLoaded = pyqtSignal(np.ndarray)
  
    # need some dead code analysis
    saveImage = pyqtSignal(QGraphicsPixmapItem)  

    updateProgress = pyqtSignal(int)


    def __init__(self):

        super().__init__()
        self.channels= None
        # self.channel_selector_combobox = self.view.toolBar.channelSelector
        self.reset_pixmap=None
        self.reset_pixmapItem=None
        self.pixmap=None
        self.pixmapItem=None
        self.begin_crop = False
        self.crop_cursor =  QCursor(QPixmap("icons/clicks.png").scaled(30,30, Qt.AspectRatioMode.KeepAspectRatio), 0,0)


    def toPixmapItem(self, pixmap):
        #convert pixmap to pixmapItem
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.canvasUpdated.emit(self.pixmapItem)

    def addImage(self, file:str):
        '''add a new image'''

        self.channels = None
        self.resetTransform()

        if isinstance(file, str):
            pixmap = self.__filename_to_pixmap(file)
        
        #     raise ValueError
        
        self.reset_pixmap=pixmap
        self.reset_pixmapItem = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.newImageAdded.emit(self.pixmapItem)


    def __filename_to_pixmap(self, file_name:str):  
        t_f = time.time()
        if file_name.endswith((".tiff", ".tif")):
            
            t = time.time()
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
                
                    __scaled = self.scale_adjust(image) 
                    image_adjusted = self.adjustContrast(__scaled) # uint8
               

                    bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                    self.np_channels[channel_name] = image_adjusted # for stardist and other image processing, maybe consider keeping it as uint16

                    qimage_channel = QImage(image_adjusted, width, height, width*bytesPerPixel, format)
                    self.channels[channel_name] = qimage_channel # for displaying on canvas
                    
                    # self.updateProgress.emit(int((channel_num+1)/num_channels*100))

                channel_one_qimage = next(iter(self.channels.values()))
                self.channelLoaded.emit(self.channels, self.np_channels)
                print(f"image loading time: {time.time() - t}")
            else:
                height, width = pages[0].shape
                __scaled = self.scale_adjust(pages[0]) 
                image_adjusted = self.adjustContrast(__scaled) # uint8

                bytesPerPixel = 2 if image_adjusted.dtype == np.uint16 else 1
                channel_one_qimage = QImage(image_adjusted, height, width, width*bytesPerPixel, format)
                self.channelNotLoaded.emit(image_adjusted)

            pixmap = QPixmap(channel_one_qimage)

        else:
            pixmap = QPixmap(file_name)

        print(time.time() - t_f)

        return pixmap
    
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
                    self.updateProgress.emit(int((i+1)/len(tif.pages)*100))
                except Exception as e:
                    print(f"Error reading page {i}: {e}")
                    continue


        return pages



    def adjustContrast(self,img):  
        
        alpha = 5 # Contrast control
        beta = 15 # Brightness control
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


    # uint16 to uint8
    def scale_adjust(self, arr:np.ndarray):
        if arr.dtype == np.uint16:
            return cv2.convertScaleAbs(arr, alpha=(255.0/65535.0))
        elif arr.dtype == np.uint8:
            return arr
        else:
            print("unsupported array type")

        
    def deleteImage(self):
        self.scene().clear()

    def resetImage(self):
        if self.pixmapItem: 
            self.toPixmapItem(self.reset_pixmap)

    
    # rotate
    def rotateImage(self, angle_text:str):
        angle = None
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.")

        if self.pixmap and angle != None:
            image = self.pixmap.toImage()
            transform = QTransform()
            
            transform.rotate(angle)
            rotated_image = image.transformed(transform, Qt.TransformationMode.SmoothTransformation)
            rotated_pixmapItem = QGraphicsPixmapItem(QPixmap.fromImage(rotated_image))
            self.canvasUpdated.emit(rotated_pixmapItem)
