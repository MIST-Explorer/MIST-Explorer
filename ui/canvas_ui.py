from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot, QPointF
import Dialogs, numpy as np, matplotlib as mpl, cv2
from qt_threading import Worker
import utils


class ReferenceGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(str)  

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(QSize(300, 300))
        self.setObjectName("reference_canvas")
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 200, 150)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if not file_path == None:
                    self.imageDropped.emit(file_path)
            event.acceptProposedAction()


    def addNewImage(self, pixmapItem):
        # center the image
        self.scene().addItem(pixmapItem)

        item_rect = pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmapItem)


class ImageGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(str)  

    imageCropped = pyqtSignal(dict)


    def __init__(self, parent=None, enc=None):

        super().__init__(parent)
        self.enc = enc
        self.setupUI()
        self.pixmapItem = None
        self.rubberBand = None
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(QPixmap("icons/clicks.png").scaled(30,30, Qt.AspectRatioMode.KeepAspectRatio), 0,0)
        self.scale_factor = 1.25
        
        self.select = False
        # self.currentChannelNum = 0

    def setupUI(self):
        self.setMinimumSize(QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True)
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 800, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse) 

    def updateCanvas(self, pixmapItem: QGraphicsPixmapItem):
        '''updates canvas when current image is operated on'''
        if self.pixmapItem:
            print("updating canvas")
            self.pixmapItem.setPixmap(pixmapItem.pixmap())
            self.__centerImage(self.pixmapItem)
            
    def saveImage(self):
        print("hello")
        
        
            
    def addNewImage(self, pixmapItem: QGraphicsPixmapItem):
        '''add a new image, deletes the older one'''
        # clear
        print("addNewImage: entered")
        self.scene().clear()
        self.pixmapItem = pixmapItem

        if not self.pixmapItem.pixmap().isNull():
            print("addNewImage: there is a pixmapItem")
        else:
            print("addNewImage; there is no pixmapItem")

        self.__centerImage(self.pixmapItem)
        #make item movable
        self.pixmapItem.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

        # add the item to the scene
        self.scene().addItem(self.pixmapItem)
        print("addNewImage: adding to scene")

    def __centerImage(self, pixmapItem):
        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmapItem)

    def isEmpty(self) -> bool:
        return self.pixmapItem == None
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if not file_path == None:
                    self.imageDropped.emit(file_path)
            event.acceptProposedAction()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1/self.scale_factor, 1/self.scale_factor)


    def mousePressEvent(self, event: QMouseEvent):
        print("mousePressEvent: entered", self.isEmpty(), self.begin_crop, self.select)
        # self.select = False
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
                
        elif self.select:
            print("mousePressEvent: with select mouse click detected")
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
        
        else: super().mousePressEvent(event)


    def mouseMoveEvent(self, event:QMouseEvent):
        
        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            # print(self.parent())
            
            x = int(image_pos.x())
            y = int(image_pos.y())
            if x <= self.pixmapItem.pixmap().width() and y <= self.pixmapItem.pixmap().height() and x >= 0 and y >= 0:
                self.enc.updateMousePositionLabel(f"X: {x}, Y: {y}")
            else:
                self.enc.updateMousePositionLabel(f"")
                
        if not self.isEmpty() and self.begin_crop and self.rubberBand:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
                
        if self.select and self.rubberBand and self.origin != None:
            print("okay, all good to create band")
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.rubberBand.hide()
                selectedRect = self.rubberBand.geometry()
                print(f"Selected rectangle: {selectedRect}")
                
                if not selectedRect.isEmpty():
                    view_rect = self.viewport().rect()
                    # scene_rect = self.mapToScene(view_rect).boundingRect()
                    print("view_rect:", view_rect)
                    # print("scene_rect:", scene_rect)
                    
                    x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
                    y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()
                    print(x_ratio, y_ratio)

                    #need to scale current rect up to the size of actual image 
                    self.__qt_image_rect = QRect(
                        int(selectedRect.left() * x_ratio),
                        int(selectedRect.top() * y_ratio),
                        int(selectedRect.width() * x_ratio),
                        int(selectedRect.height() * y_ratio)
                    )

                    left = int(selectedRect.left() * x_ratio)
                    top = int(selectedRect.top() * y_ratio)
                    right = int(selectedRect.right() * x_ratio)  
                    bottom = int(selectedRect.bottom() * y_ratio)  
                    image_rect = (left, top, right, bottom)
                    print(image_rect)
                    
                    # Show the cropped image in a new window
                    self.showCroppedImage(image_rect)
                    
        elif self.select:
            self.select = False
            self.origin = None
            
            if not self.isEmpty():
                selectedRect = self.rubberBand.geometry()
                print(f"Selected rectangle: {selectedRect}")
                
                if not selectedRect.isEmpty():
                    view_rect = self.viewport().rect()
                    print("view_rect:", view_rect)
                    
                    x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
                    y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()
                    print(x_ratio, y_ratio)

                    # Scale the selected rectangle to the size of the actual image
                    self.__qt_image_rect = QRect(
                        int(selectedRect.left() * x_ratio),
                        int(selectedRect.top() * y_ratio),
                        int(selectedRect.width() * x_ratio),
                        int(selectedRect.height() * y_ratio)
                    )

                    left = int(selectedRect.left() * x_ratio)
                    top = int(selectedRect.top() * y_ratio)
                    right = int(selectedRect.right() * x_ratio)  
                    bottom = int(selectedRect.bottom() * y_ratio)  
                    image_rect = (left, top, right, bottom)
                    
                    print(image_rect)
                    
                    # Adjust the rubber band to follow the image around on the canvas
                    if self.pixmapItem and 0 <= left < self.pixmapItem.pixmap().width() and 0 <= top < self.pixmapItem.pixmap().height():
                        self.rubberBand.setGeometry(QRect(
                            self.mapFromScene(self.pixmapItem.mapToScene(QPointF(left, top))),
                            self.mapFromScene(self.pixmapItem.mapToScene(QPointF(right, bottom)))
                        ).normalized())
                    else:
                        print("Invalid coordinates or pixmapItem is None")
            # self.rubberBand.hide()
            if not self.isEmpty():
                selectedRect = self.rubberBand.geometry()
                print(f"Selected rectangle: {selectedRect}")
                
                if not selectedRect.isEmpty():
                    view_rect = self.viewport().rect()
                    # scene_rect = self.mapToScene(view_rect).boundingRect()
                    print("view_rect:", view_rect)
                    # print("scene_rect:", scene_rect)
                    
                    x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
                    y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()
                    print(x_ratio, y_ratio)

                    #need to scale current rect up to the size of actual image 
                    self.__qt_image_rect = QRect(
                        int(selectedRect.left() * x_ratio),
                        int(selectedRect.top() * y_ratio),
                        int(selectedRect.width() * x_ratio),
                        int(selectedRect.height() * y_ratio)
                    )

                    left = int(selectedRect.left() * x_ratio)
                    top = int(selectedRect.top() * y_ratio)
                    right = int(selectedRect.right() * x_ratio)  
                    bottom = int(selectedRect.bottom() * y_ratio)  
                    image_rect = (left, top, right, bottom)
                    
                    print(image_rect)
                    # pass for analysis
                    # image_arr[top:bottom+1, left:right+1]
        else: 
            super().mouseReleaseEvent(event)


    def showCroppedImage(self, image_rect):
            
            print("in view.canvas: ", self.currentChannelNum)
            q_im = list(self.channels.values())[self.currentChannelNum]
            pix = QPixmap(q_im)
            cropped = pix.copy(self.__qt_image_rect).toImage()
            print("converting to pixmap") 
            cropped_pix = QPixmap(cropped)
            self.dialog = Dialogs.ImageDialog(self, cropped_pix)
            self.dialog.exec()

            print("confirmed crop?", self.dialog.confirm_crop)
            if self.dialog.confirm_crop:
                self.crop_worker = Worker(self.cropImageTask, image_rect)
                self.crop_worker.signal.connect(self.onCropCompleted) # result is cropped images dict
                self.crop_worker.finished.connect(self.crop_worker.quit)
                self.crop_worker.start()
            else:
                self.endCrop()
                print("rejected")
            #somewhere here should be a signal indicating that the cropped image was rejected and don't follow through with the rest of the code
            # crop_worker.error.connect(self.onError)

    def cropImageTask(self, image_rect) -> dict:
        from utils import qimage_to_numpy
        # cropped_images = {}
        cropped_arrays = {}
        left, top, right, bottom = image_rect
        
        for channel_name, image_arr in self.np_channels.items():

            # arr = qimage_to_numpy(image)
            cropped_array = image_arr[top:bottom+1, left:right+1]
            cropped_arrays[channel_name] = cropped_array

            # pixmap = QPixmap(image)
            # cropped = pixmap.copy(image_rect).toImage()
            # cropped.convertToFormat(QImage.Format.Format_Grayscale8)
            # print("type of cropped:", type(cropped))
            # cropped_images[channel_name] = cropped
        cropped_arrays_cont = {key: np.ascontiguousarray(a, dtype="uint8") for key, a in cropped_arrays.items() if not a.data.contiguous}
        # print("debuggingggg")
        # print(cropped_arrays_cont['Channel 3'].shape)
        # import cv2
        # cv2.imshow("test cropping", cropped_arrays_cont['Channel 3'])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return cropped_arrays_cont
    
    cropSignal = pyqtSignal(dict, bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_images:dict):
        import cv2



        # current_channel = list(cropped_images.values())[self.currentChannelNum]
        # cv2.imshow("test cropping", current)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # channel_qimage = numpy_to_qimage(current_channel)
        # channel_pixmap = QPixmap(channel_qimage)
        # self.dialog = Dialogs.ImageDialog(self, channel_pixmap)

        self.endCrop()

        self.cropSignal.emit(cropped_images, False)
        
        print("emitting cropped images")
        print("reached")

    def startCrop(self):
        self.begin_crop = True
        self.setCursor(self.crop_cursor)
        
    def endCrop(self):
        self.begin_crop = False
        self.unsetCursor()
        


    def loadChannels(self, np_channels):
        self.np_channels = np_channels
        self.channels = self.__numpy2QImageDict(self.np_channels)

    def __numpy2QImageDict(self, numpy_channels_dict: dict) -> dict:
        return {key: utils.numpy_to_qimage(arr) for key, arr in numpy_channels_dict.items()} 
    
    
    def setCurrentChannel(self, channel_num:int) -> None:
        self.currentChannelNum = channel_num
        print("in view.canvas: ", self.currentChannelNum)

    def _confirmCrop(self, confirmed:bool):
        self.confirmCrop = confirmed


    