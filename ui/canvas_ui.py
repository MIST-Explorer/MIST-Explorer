from PyQt6.QtWidgets import QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal
import Dialogs



class ReferenceGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(QPixmap)  

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
                pixmap = QPixmap(file_path) 
                if not pixmap.isNull():
                    self.imageDropped.emit(pixmap)
            event.acceptProposedAction()


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
        self.origin = QPoint()
        self.crop_cursor = QCursor(QPixmap("icons/clicks.png").scaled(30,30, Qt.AspectRatioMode.KeepAspectRatio), 0,0)
        self.scale_factor = 1.25

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
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
        else: super().mousePressEvent(event)


    def mouseMoveEvent(self, event:QMouseEvent):
        if not self.isEmpty() and self.begin_crop and self.rubberBand:
                self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
        else:
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

                    # need to scale current rect up to the size of actual image 
                    image_rect = QRect(
                        int(selectedRect.left() * x_ratio),
                        int(selectedRect.top() * y_ratio),
                        int(selectedRect.width() * x_ratio),
                        int(selectedRect.height() * y_ratio)
                    )
                    print(image_rect)
                    
                    # Show the cropped image in a new window
                    self.showCroppedImage(image_rect)
        else: super().mouseReleaseEvent(event)

    def showCroppedImage(self, image_rect):

        cropped_images = {}
        
        for channel_name, image in self.channels.items():

            pixmapItem = QGraphicsPixmapItem(QPixmap(image))
            cropped = pixmapItem.pixmap().copy(image_rect)
            cropped_images[channel_name] = cropped

        channel_one = next(iter(cropped_images.values()))
        self.dialog = Dialogs.ImageDialog(self, channel_one)

        
        self.endCrop()
        self.dialog.exec()

    def startCrop(self):
        self.begin_crop = True
        self.setCursor(self.crop_cursor)

    def endCrop(self):
        self.begin_crop = False
        self.unsetCursor()

    def loadChannels(self, value):
        self.channels = value
    