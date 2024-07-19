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

    def __init__(self, parent=None):

        super().__init__(parent)
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
        print("if statement canvas")
        print("self.pixmapItem:", self.pixmapItem)
        if self.pixmapItem:
            print("updating canvas")
            self.pixmapItem.setPixmap(pixmapItem.pixmap())
            self.__centerImage(self.pixmapItem)
            
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
        else: super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.rubberBand.hide()
                selectedRect = self.rubberBand.geometry()
                print(f"Selected rectangle: {selectedRect}")
                
                # Extract the selected rectangle from the original image
                if not selectedRect.isEmpty():
                    cropped_image = self.pixmapItem.pixmap().copy(selectedRect).toImage()

                    # Show the cropped image in a new window
                    self.showCroppedImage(cropped_image)
        else: super().mouseReleaseEvent(event)

    def showCroppedImage(self, cropped_image):
        self.dialog = Dialogs.ImageDialog(self, cropped_image)
        self.endCrop()
        self.dialog.exec()

    def startCrop(self):
        self.begin_crop = True
        self.setCursor(self.crop_cursor)

    def endCrop(self):
        self.begin_crop = False
        self.unsetCursor()
    