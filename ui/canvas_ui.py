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
        self.draggable = True
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.hello = False
        
        

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

    # def zoom(self, scale_factor):
    #     self.scale(scale_factor, scale_factor)

    def mousePressEvent(self, event):
        # if self.mousePressPos is not None:
            # print("from  customrubberband mouse press event")
            self.mousePressPos = event.pos()                # global
            self.mouseMovePos = event.pos() - self.pos()    # local
            # super(CustomRubberBand, self).mousePressEvent(event)
            self.hello = True

    def mouseMoveEvent(self, event):
        if self.mousePressPos is not None and self.hello:
            # print("from  customrubberband mouseMoveEvent")
            pos = event.pos()
            moved = pos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = pos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = pos - self.pos()
            super(CustomRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # print("from  customrubberband mouseReleaseEvent")
        self.hello = False
        if self.mousePressPos is not None:
            moved = event.pos() - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Do not call click event or so on
                event.ignore()
            self.mousePressPos = None
            self.mouseMovePos = None
        # super(CustomRubberBand, self).mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        current_rect = self.geometry()
        
        # Calculate new size
        new_width = current_rect.width() * zoom_factor
        new_height = current_rect.height() * zoom_factor
        
        # Calculate new position to keep it centered
        new_x = current_rect.x() - (new_width - current_rect.width()) / 2
        new_y = current_rect.y() - (new_height - current_rect.height()) / 2
        
        # Set new geometry
        self.setGeometry(int(new_x), int(new_y), int(new_width), int(new_height))
        self.update()


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
        # self.fitInView(pixmapItem, Qt.AspectRatioMode.KeepAspectRatio)
        # self.centerOn(pixmapItem)

class ImageGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(str)  
    imageCropped = pyqtSignal(dict)
    imageChanged = pyqtSignal()
    
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
            # self.__centerImage(self.pixmapItem)
            self.pixmapItem.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)

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
        # Determine the zoom factor
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9

        # Store the scene positions of all rubber bands
        rubber_band_positions = []
        for rubber_band in self.rubberBands:  # Assuming self.rubber_bands is your list of QRubberBand objects
            rubber_band_geometry = rubber_band.geometry()
            top_left_scene = self.mapToScene(rubber_band_geometry.topLeft())
            bottom_right_scene = self.mapToScene(rubber_band_geometry.bottomRight())
            rubber_band_positions.append((rubber_band, top_left_scene, bottom_right_scene))

        # Perform the zoom
        self.scale(zoom_factor, zoom_factor)

        # Update the geometry of each rubber band
        for rubber_band, top_left_scene, bottom_right_scene in rubber_band_positions:
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            new_rect = QRect(new_top_left_view, new_bottom_right_view)
            rubber_band.setGeometry(new_rect)


        


    def mousePressEvent(self, event: QMouseEvent):
        print("mousePressEvent: entered", self.isEmpty(), self.begin_crop, self.select)
        if not self.isEmpty() and self.begin_crop:
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = CustomRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
            return
                
        elif self.select:

            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            self.starting_x = int(image_pos.x())
            self.starting_y = int(image_pos.y())

            print("mousePressEvent: with select mouse click detected")
            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                rubberBand = CustomRubberBand(QRubberBand.Shape.Rectangle, self)
                self.rubberBands.append(rubberBand)
                self.rubberBandColors.append(rubberBand.color)  # Store the color of the rubber band
                rubberBand.setGeometry(QRect(self.origin, QSize()))
                rubberBand.show()
            return
        
        else: 
            # pass
            super().mousePressEvent(event)

        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            self.starting_x = int(image_pos.x())
            self.starting_y = int(image_pos.y())

        if not self.select:
            for r in self.rubberBands:
                r.mousePressEvent(event)

    def mouseMoveEvent(self, event:QMouseEvent):
        
        combined_layers=""
        
        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            x = int(image_pos.x())
            y = int(image_pos.y())
            # Get the image pixel RGB value
            img = self.pixmapItem.pixmap().toImage()

            # mouse tool tip 
            if 0 <= x < img.width() and 0 <= y < img.height():

                color = QColor(img.pixel(x, y))
                r, g, b = color.red(), color.green(), color.blue()

                global_pos = self.mapToGlobal(event.pos())
                QToolTip.showText(global_pos, f"", self)
                layers = self.enc.view_tab.get_layer_values_at(x, y)
                
                # controls mouse tool tip
                if layers:
                    layers = [f"{layer}: {value[0]}\n" for layer, value in layers]
                    combined_layers = ''.join(layers)[:-1]
                    QToolTip.showText(global_pos, combined_layers, self)
                else:                    
                    QToolTip.showText(global_pos, f"R: {r}, G: {g}, B: {b}", self)

                # controls upper xy label
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
            # pass
            super().mouseMoveEvent(event)

        if not self.select:
            for r in self.rubberBands:
                r.mouseMoveEvent(event)

        # else: 
        #     for r in self.rubberBands:
        #         r.mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):

        for r in self.rubberBands:
            # print("try to relesa")
            r.mouseReleaseEvent(event)
            # print("w is this not worin")

        if self.isEmpty(): # exit if there is no image
            return
        
        if not self.begin_crop and not self.select:
            return
        
        

        if event.button() == Qt.MouseButton.LeftButton:  
            rubberband = self.rubberBand if self.begin_crop else self.rubberBands[-1]
             
            if self.begin_crop:
                rubberband.hide() 

                selectedRect = rubberband.geometry() 
                if selectedRect.isEmpty():
                    return

                view_rect = self.viewport().rect()            
                x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
                y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()

                left = int(selectedRect.left() * x_ratio)
                top = int(selectedRect.top() * y_ratio)

                height = int(selectedRect.height() * y_ratio)
                width = int(selectedRect.width() * x_ratio)
                self.__qt_image_rect = QRect(left, top, width, height)
                self.showCroppedImage(image_rect)

            if self.select:
                self.select = False
                self.origin = None

                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
                image_rect = (self.starting_x, self.starting_y, int(image_pos.x()), int(image_pos.y()))
            
                data = self.enc.view_tab.get_layer_values_from_to(*image_rect)
                print(image_rect)

                # this is stupid but i have to do this weird manipulation
                # for the data to be displayed in the stats tab                
                names = [i[0] for i in data]
                expr = [i[1][i[1] != 0]  for i in data]

                # list(filter(lambda x: x != 0, arr))

                random_data = pd.DataFrame({
                'Protein': names,
                'Expression': expr
                })

                self.enc.analysis_tab.analyze_region(random_data, rubberband, image_rect)

                return
            
            if self.pixmapItem:
                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
                
                print(self.starting_x - int(image_pos.x()))
                print(self.starting_y - int(image_pos.y()))

    

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
            self.crop_worker.finished.connect(self.crop_worker.deleteLater)
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


        cropped_arrays_cont = {key: np.ascontiguousarray(a, dtype="uint16") for key, a in cropped_arrays.items() if not a.data.contiguous}

        return cropped_arrays_cont
    
    cropSignal = pyqtSignal(dict, bool)
    @pyqtSlot(dict)
    def onCropCompleted(self, cropped_images:dict):
        import cv2

        self.endCrop()

        self.cropSignal.emit(cropped_images, False)

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
