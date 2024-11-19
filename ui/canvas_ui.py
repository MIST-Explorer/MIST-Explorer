from PyQt6.QtWidgets import QToolTip, QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage, QPalette, QPainter, QBrush, QColor, QPen
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot, QPointF
import Dialogs, numpy as np, matplotlib as mpl, cv2
from qt_threading import Worker
import utils
from PyQt6.QtGui import QColor
import random
import pandas as pd


def getRandomColor():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)
        
        
class CustomRubberBand(QRubberBand):
    def __init__(self, shape, parent=None):
        super(CustomRubberBand, self).__init__(shape, parent)
        self.fill = getRandomColor()
        self.color = QColor(*self.fill[0:3])
        self.f = QColor(*self.fill)
        self.filled = False
        
        

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)  # Set color to red
        pen.setWidth(3)  # Set pen width
        painter.setPen(pen)
        painter.drawRect(self.rect())

        if self.filled:
            color_trans = QColor(self.color)
            color_trans.setAlpha(75)  
            painter.fillRect(self.rect(), color_trans)
            

    def setFilled(self, fill):
        self.filled = fill
        self.update()  # Trigger a repaint to apply the change
        
        # brush = QBrush(QColor(255, 0, 0, 50))  # 50 is the alpha channel (transparency)
        # painter.setBrush(brush)

class ReferenceGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(str)  

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(QSize(300, 300))
        self.setObjectName("reference_canvas")
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 200, 150)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")
        
        self.parent = parent

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
        self.rubberBands = []  # List to store multiple rubber bands
        self.rubberBandColors = []  # List to store colors of the rubber bands
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(QPixmap("icons/clicks.png").scaled(30,30, Qt.AspectRatioMode.KeepAspectRatio), 0,0)
        self.scale_factor = 1.25
        self.select = False

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
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = CustomRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
                
        elif self.select:
            print("mousePressEvent: with select mouse click detected")
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                rubberBand = CustomRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBands.append(rubberBand)
                self.rubberBandColors.append(rubberBand.color)  # Store the color of the rubber band
                rubberBand.setGeometry(QRect(self.origin, QSize()))
                rubberBand.show()
        
        else: super().mousePressEvent(event)

    def mouseMoveEvent(self, event:QMouseEvent):
        
        combined_layers=""
        
        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            x = int(image_pos.x())
            y = int(image_pos.y())
            # Get the image pixel RGB value
            img = self.pixmapItem.pixmap().toImage()
            if 0 <= x < img.width() and 0 <= y < img.height():
                global_pos = self.mapToGlobal(event.pos())
                QToolTip.showText(global_pos, f"", self)
                layers = self.enc.view_tab.get_layer_values_at(x, y)
                if layers:
                    layers = [f"{layer}: {value[0]}\n" for layer, value in layers]
                    combined_layers = ''.join(layers)[:-1]
                    
                    
                    # QToolTip.showText(global_pos, "not", self)
                    QToolTip.showText(global_pos, combined_layers, self)
                else:
                    color = QColor(img.pixel(x, y))
                    r, g, b = color.red(), color.green(), color.blue()
                    
                    QToolTip.showText(global_pos, f"R: {r}, G: {g}, B: {b}", self)

        
        
        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            x = int(image_pos.x())
            y = int(image_pos.y())
            
            img = self.pixmapItem.pixmap().toImage()
                
            if x <= self.pixmapItem.pixmap().width() and y <= self.pixmapItem.pixmap().height() and x >= 0 and y >= 0:
                color = QColor(img.pixel(x, y))
                r, g, b = color.red(), color.green(), color.blue()
                if combined_layers != "": 
                    combined_layers = combined_layers.replace("\n", ", ")
                    # print("combined_layers", combined_layers)
                    combined_layers += ";"
                    self.enc.updateMousePositionLabel(f"{combined_layers} X: {x}, Y: {y}")
                else:
                    self.enc.updateMousePositionLabel(f"R: {r}, G: {g}, B: {b} X: {x}, Y: {y}")
            else:
                self.enc.updateMousePositionLabel(f"")
                
        if not self.isEmpty() and self.begin_crop and self.rubberBand:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
                
        if self.select and self.rubberBands and self.origin != None:
            self.rubberBands[-1].setGeometry(QRect(self.origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):

        if self.isEmpty(): # exit if there is no image
            return
        
        if not self.begin_crop and not self.select:
            return

        if event.button() == Qt.MouseButton.LeftButton:  
            rubberband = self.rubberBand if self.begin_crop else self.rubberBands[-1]
            selectedRect = rubberband.geometry() 
            if selectedRect.isEmpty():
                return

            view_rect = self.viewport().rect()            
            x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
            y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()

            left = int(selectedRect.left() * x_ratio)
            top = int(selectedRect.top() * y_ratio)
            right = int(selectedRect.right() * x_ratio)  
            bottom = int(selectedRect.bottom() * y_ratio)  

            height = int(selectedRect.height() * y_ratio)
            width = int(selectedRect.width() * x_ratio)
                        
            image_rect = (left, top, right, bottom)
            print(image_rect)

                
            if self.begin_crop:
                rubberband.hide() 
                self.__qt_image_rect = QRect(left, top, width, height)
                self.showCroppedImage(image_rect)

            if self.select:
                self.select = False
                self.origin = None
                data = self.enc.view_tab.get_layer_values_from_to(*image_rect)

                # this is stupid but i have to do this weird manipulation
                # for the data to be displayed in the stats tab                
                names = [i[0] for i in data]
                expr = [i[1] for i in data]

                random_data = pd.DataFrame({
                'Protein': names,
                'Expression': expr
                })

                self.enc.analysis_tab.add_stats(random_data, rubberband.color, rubberband)

                return
            
        #     if not self.isEmpty():
                
        #         selectedRect = self.rubberBands[-1].geometry()
        #         print(f"Selected rectangle: {selectedRect}")
                
        #         if not selectedRect.isEmpty():
        #             view_rect = self.viewport().rect()
        #             print("view_rect:", view_rect)
                    
        #             x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
        #             y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()
        #             print(x_ratio, y_ratio)

        #             self.__qt_image_rect = QRect(
        #                 int(selectedRect.left() * x_ratio),
        #                 int(selectedRect.top() * y_ratio),
        #                 int(selectedRect.width() * x_ratio),
        #                 int(selectedRect.height() * y_ratio)
        #             )

        #             left = int(selectedRect.left() * x_ratio)
        #             top = int(selectedRect.top() * y_ratio)
        #             right = int(selectedRect.right() * x_ratio)  
        #             bottom = int(selectedRect.bottom() * y_ratio)  
        #             image_rect = (left, top, right, bottom)
                    
        #             print(image_rect)
                    
        #             if self.pixmapItem and 0 <= left < self.pixmapItem.pixmap().width() and 0 <= top < self.pixmapItem.pixmap().height():
        #                 self.rubberBands[-1].setGeometry(QRect(
        #                     self.mapFromScene(self.pixmapItem.mapToScene(QPointF(left, top))),
        #                     self.mapFromScene(self.pixmapItem.mapToScene(QPointF(right, bottom)))
        #                 ).normalized())
        #             else:
        #                 print("Invalid coordinates or pixmapItem is None")
                    
        #             # Call the method with the color of the rubber band
        #             # self.parent.analysis_tab.add_line_to_current_graph(self.rubberBandColors[-1])
        # else: 
        #     super().mouseReleaseEvent(event)

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
            self.crop_worker.signal.connect(self.onCropCompleted) 
            self.crop_worker.finished.connect(self.crop_worker.quit)
            self.crop_worker.start()
        else:
            self.endCrop()
            print("rejected")

    def cropImageTask(self, image_rect) -> dict:
        from utils import qimage_to_numpy
        cropped_arrays = {}
        left, top, right, bottom = image_rect
        
        for channel_name, image_arr in self.np_channels.items():
            cropped_array = image_arr[top:bottom+1, left:right+1]
            cropped_arrays[channel_name] = cropped_array

        cropped_arrays_cont = {key: np.ascontiguousarray(a, dtype="uint8") for key, a in cropped_arrays.items() if not a.data.contiguous}

        return cropped_arrays_cont
    
    cropSignal = pyqtSignal(dict, bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_images:dict):
        import cv2

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
