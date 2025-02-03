from PyQt6.QtWidgets import QToolTip, QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,  QGraphicsRectItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage, QPalette, QPainter, QBrush, QColor, QPen
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot, QPointF
import Dialogs, numpy as np, matplotlib as mpl, cv2
from PyQt6.QtWidgets import QToolTip, QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal, pyqtSlot, QPointF
import Dialogs
import numpy as np
import cv2
from qt_threading import Worker
import utils
from PyQt6.QtGui import QColor
import random
import pandas as pd


from PyQt6.QtWidgets import QApplication, QRubberBand, QMainWindow
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCore import QRect, QPoint, Qt, QSize

class CircularRubberBand(QRubberBand):
    def __init__(self, shape, parent=None):
        super().__init__(shape, parent)
        self.fill = self.getRandomColor()

        # self.starting_x = starting_x
        # self.starting_y = starting_y
        self.color = QColor(*self.fill[0:3])

        self.f = QColor(*self.fill)
        self.filled = False
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.hello = False

    def getRandomColor(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw a circle inside the current geometry
        rect = self.rect()
        painter.drawEllipse(rect)


    def mousePressEvent1(self, event):
    # if self.mousePressPos is not None:
        # print("from  AnalysisRubberBand mouse press event")
        self.mousePressPos = event.pos()                # global
        self.mouseMovePos = event.pos() - self.pos()    # local
        # super(AnalysisRubberBand, self).mousePressEvent(event)
        self.hello = True

    def mouseMoveEvent1(self, event):
        if self.mousePressPos is not None and self.hello:
            pos = event.pos()
            moved = pos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = pos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = pos - self.pos()
            super(CircularRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent1(self, event):
        # print("from  AnalysisRubberBand mouseReleaseEvent")
        self.hello = False
        if self.mousePressPos is not None:
            moved = event.pos() - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Do not call click event or so on
                event.ignore()
            self.mousePressPos = None
            self.mouseMovePos = None


        
class AnalysisRubberBand(QRubberBand):
    def __init__(self, shape, starting_x, starting_y, parent=None):
        super(AnalysisRubberBand, self).__init__(shape, parent)
        self.fill = self.getRandomColor()

        self.starting_x = starting_x
        self.starting_y = starting_y
        self.color = QColor(*self.fill[0:3])
        self.f = QColor(*self.fill)
        self.filled = False
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.hello = False

    def getRandomColor(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)

    def getRubberBandSizeRelativeToScene(self):
        rect = self.geometry()  # QRect of the rubberband
        top_left_scene = self.mapToScene(rect.topLeft())
        bottom_right_scene = self.mapToScene(rect.bottomRight())
        
        width = bottom_right_scene.x() - top_left_scene.x()
        height = bottom_right_scene.y() - top_left_scene.y()
        
        return width, height

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

    def mousePressEvent1(self, event):
        # if self.mousePressPos is not None:
            # print("from  AnalysisRubberBand mouse press event")
            self.mousePressPos = event.pos()                # global
            self.mouseMovePos = event.pos() - self.pos()    # local
            # super(AnalysisRubberBand, self).mousePressEvent(event)
            self.hello = True

    def mouseMoveEvent1(self, event):
        if self.mousePressPos is not None and self.hello:
            pos = event.pos()
            moved = pos - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = pos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = pos - self.pos()
            super(AnalysisRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent1(self, event):
        # print("from  AnalysisRubberBand mouseReleaseEvent")
        self.hello = False
        if self.mousePressPos is not None:
            moved = event.pos() - self.mousePressPos
            if moved.manhattanLength() > self.dragging_threshold:
                # Do not call click event or so on
                event.ignore()
            self.mousePressPos = None
            self.mouseMovePos = None



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
        self.select = False
        self.circle_select = False
        self.zoom = 1

        self.rubber_band_positions = []

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
        '''Update the pixmap of the existing image or add a new one'''
        print("addNewImage: entered")

        if not hasattr(self, 'pixmapItem') or self.pixmapItem is None:
            # If no pixmapItem exists, add it to the scene
            self.pixmapItem = pixmapItem
            
            self.pixmapItem.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.scene().addItem(self.pixmapItem)
            self.__centerImage(self.pixmapItem)
            print("addNewImage: created and added new pixmapItem")
        else:
            # Update the pixmap of the existing item
            self.pixmapItem.setPixmap(pixmapItem.pixmap())
            print("addNewImage: updated pixmap of existing pixmapItem")

        if not self.pixmapItem.pixmap().isNull():
            print("addNewImage: there is a pixmapItem")
        else:
            print("addNewImage; there is no pixmapItem")

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
        
        zooming_out = event.angleDelta().y() > 0

        # this is to prevent zooming in too much or zooming out too much
        if self.zoom > 1.1**90 and zooming_out: # if max zoomed out, and not zooming in, quit. not as nessecary
            return
        
        if self.zoom < 1/(1.1**2) and not zooming_out: # if max zoomed in, and not zooming out, quit.
            return

        zoom_factor = 1.1 if zooming_out else 0.9

        self.zoom *= zoom_factor


        # part a) storing values -- i forget why this has to be this way, but I dont think im going to change it
        if len(self.rubber_band_positions) == 0: # this is important though, to only store the values once per zoom operation
            self.rubber_band_positions = []
            for rubber_band in self.rubberBands:  # Assuming self.rubber_bands is your list of QRubberBand objects
                rubber_band_geometry = rubber_band.geometry()
                top_left_scene = self.mapToScene(rubber_band_geometry.topLeft())
                bottom_right_scene = self.mapToScene(rubber_band_geometry.bottomRight())
                self.rubber_band_positions.append((rubber_band, top_left_scene, bottom_right_scene))

        # Perform the zoom
        self.scale(zoom_factor, zoom_factor)

        # part b) updating values
        for rubber_band, top_left_scene, bottom_right_scene in self.rubber_band_positions:
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            new_rect = QRect(new_top_left_view, new_bottom_right_view)
            rubber_band.setGeometry(new_rect)


    def mousePressEvent(self, event: QMouseEvent):

        if self.isEmpty():
            return
            
        print("mousePressEvent: entered", self.isEmpty(), self.begin_crop, self.select)

        if self.begin_crop:

            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            self.starting_x = int(image_pos.x())
            self.starting_y = int(image_pos.y())

            if event.button() == Qt.MouseButton.LeftButton:
                self.origin = event.pos()
                if not self.rubberBand:
                    self.rubberBand = AnalysisRubberBand(QRubberBand.Shape.Rectangle, self.starting_x, self.starting_y, self)
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
                rubberBand = AnalysisRubberBand(QRubberBand.Shape.Rectangle, self.
                starting_x, self.starting_y, self)

                self.rubberBands.append(rubberBand)
                self.rubberBandColors.append(rubberBand.color)  # Store the color of the rubber band
                
                rubberBand.setGeometry(QRect(self.origin, QSize()))
                rubberBand.show()
            return

        elif self.circle_select:

            self.origin = event.pos()
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            self.starting_x = int(image_pos.x())
            self.starting_y = int(image_pos.y())

            self.center = QPoint(self.starting_x, self.starting_y)
            
            self.rubberband = CircularRubberBand(QRubberBand.Shape.Rectangle, self)
            self.rubberBands.append(self.rubberband)
            self.rubberBandColors.append(self.rubberband.color)

            self.rubberband.setGeometry(QRect(self.origin, QSize())) 
            self.rubberband.show()
        
        else: 
            # pass
            super().mousePressEvent(event)

        if self.pixmapItem:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmapItem.mapFromScene(scene_pos)
            
            self.starting_x = int(image_pos.x())
            self.starting_y = int(image_pos.y())

        if not self.select and not self.circle_select:
            for r in self.rubberBands:
                r.mousePressEvent1(event)

    def mouseMoveEvent(self, event:QMouseEvent):
        super().mouseMoveEvent(event)
        
        combined_layers=""
        self.rubber_band_positions = []
        
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

                    layers = [f"{layer}: { value[0]}\n" for layer, value in layers]
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

        if self.circle_select and self.rubberBands and self.origin != None:
            # current = event.position().toPoint()
            # radius = int((self.center - current).manhattanLength())  # Calculate the radius
            # top_left = self.center - QPoint(radius, radius)
            # size = 2 * radius

            center = self.origin
            corner = event.pos()
            size = max(abs(center.x() - corner.x()), abs(center.y() - corner.y())) * 2
            self.rubberBands[-1].setGeometry(QRect(center.x() - size // 2, center.y() - size // 2, size, size))

            # self.rubberBands[-1].setGeometry(QRect(self.origin, QSize(size, size)))

        else:
            # pass
            super().mouseMoveEvent(event)

        if not self.select and not self.circle_select:
            for r in self.rubberBands:
                r.mouseMoveEvent1(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        self.rubber_band_positions = []

        for r in self.rubberBands:
            r.mouseReleaseEvent1(event)

        if self.isEmpty(): # exit if there is no image
            return
        
        if not self.begin_crop and not self.select and not self.circle_select:
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

                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
            

                image_rect = (self.starting_x, self.starting_y, int(image_pos.x()), int(image_pos.y()))
                self.showCroppedImage(image_rect)

            if self.select:
                self.select = False
                self.origin = None
                    

                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
                image_rect = (self.starting_x, self.starting_y, int(image_pos.x()), int(image_pos.y()))
            

                self.enc.analysis_tab.analyze_region(rubberband, image_rect)

                return

            if self.circle_select:
                self.circle_select = False
                self.origin = None
                # self.rubberBands[-1].hide()
                # self.rubberBands[-1] = None  # Optionally keep it visible if needed
            
            if self.pixmapItem:
                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
                
                print(self.starting_x - int(image_pos.x()))
                print(self.starting_y - int(image_pos.y()))

    

    def showCroppedImage(self, image_rect):

        # TODO add if statement to diff. between single page tiffs/jpeg/png and multi-page tiffs
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
########################################

class ResizableRect(QGraphicsRectItem):
    selected_edge = None
    def __init__(self, x, y, width, height, onCenter=False):
        if onCenter:
            super().__init__(-width / 2, -height / 2, width, height)
        else:
            super().__init__(0, 0, width, height)
        self.setPos(x, y)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QBrush(Qt.GlobalColor.blue), 3,  Qt.PenStyle.DotLine))

        # a child item that shows the current position; note that this is only
        # provided for explanation purposes, a *proper* implementation should
        # use the ItemSendsGeometryChanges flag for *this* item and then
        # update the value within an itemChange() override that checks for
        # ItemPositionHasChanged changes.
        self.posItem = QGraphicsSimpleTextItem(
            '{}, {}'.format(self.x(), self.y()), parent=self)
        self.posItem.setPos(
            self.boundingRect().x(), 
            self.boundingRect().y() - self.posItem.boundingRect().height()
        )

    def getEdges(self, pos):
        # return a proper Qt.Edges flag that reflects the possible edge(s) at
        # the given position; note that this only works properly as long as the
        # shape() override is consistent and for *pure* rectangle items; if you
        # are using other shapes (like QGraphicsEllipseItem) or items that have
        # a different boundingRect or different implementation of shape(), the
        # result might be unexpected.
        # Finally, a simple edges = 0 could suffice, but considering the new
        # support for Enums in PyQt6, it's usually better to use the empty flag
        # as default value.

        edges = Qt.Edge(0)
        rect = self.rect()
        border = self.pen().width() / 2

        if pos.x() < rect.x() + border:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() > rect.right() - border:
            edges |= Qt.Edge.RightEdge
        if pos.y() < rect.y() + border:
            edges |= Qt.Edge.TopEdge
        elif pos.y() > rect.bottom() - border:
            edges |= Qt.Edge.BottomEdge

        return edges

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_edge = self.getEdges(event.pos())
            self.offset = QPointF()
        else:
            self.selected_edge = Qt.Edge(0)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selected_edge:
            mouse_delta = event.pos() - event.buttonDownPos(Qt.MouseButton.LeftButton)
            rect = self.rect()
            pos_delta = QPointF()
            border = self.pen().width()

            if self.selected_edge & Qt.Edge.LeftEdge:
                # ensure that the width is *always* positive, otherwise limit
                # both the delta position and width, based on the border size
                diff = min(mouse_delta.x() - self.offset.x(), rect.width() - border)
                if rect.x() < 0:
                    offset = diff / 2
                    self.offset.setX(self.offset.x() + offset)
                    pos_delta.setX(offset)
                    rect.adjust(offset, 0, -offset, 0)
                else:
                    pos_delta.setX(diff)
                    rect.setWidth(rect.width() - diff)
            elif self.selected_edge & Qt.Edge.RightEdge:
                if rect.x() < 0:
                    diff = max(mouse_delta.x() - self.offset.x(), border - rect.width())
                    offset = diff / 2
                    self.offset.setX(self.offset.x() + offset)
                    pos_delta.setX(offset)
                    rect.adjust(-offset, 0, offset, 0)
                else:
                    rect.setWidth(max(border, event.pos().x() - rect.x()))

            if self.selected_edge & Qt.Edge.TopEdge:
                # similarly to what done for LeftEdge, but for the height
                diff = min(mouse_delta.y() - self.offset.y(), rect.height() - border)
                if rect.y() < 0:
                    offset = diff / 2
                    self.offset.setY(self.offset.y() + offset)
                    pos_delta.setY(offset)
                    rect.adjust(0, offset, 0, -offset)
                else:
                    pos_delta.setY(diff)
                    rect.setHeight(rect.height() - diff)
            elif self.selected_edge & Qt.Edge.BottomEdge:
                if rect.y() < 0:
                    diff = max(mouse_delta.y() - self.offset.y(), border - rect.height())
                    offset = diff / 2
                    self.offset.setY(self.offset.y() + offset)
                    pos_delta.setY(offset)
                    rect.adjust(0, -offset, 0, offset)
                else:
                    rect.setHeight(max(border, event.pos().y() - rect.y()))

            if rect != self.rect():
                self.setRect(rect)
                if pos_delta:
                    self.setPos(self.pos() + pos_delta)
        else:
            # use the default implementation for ItemIsMovable
            super().mouseMoveEvent(event)

        self.posItem.setText('{},{} ({})'.format(
            self.x(), self.y(), self.rect().getRect()))
        self.posItem.setPos(
            self.boundingRect().x(), 
            self.boundingRect().y() - self.posItem.boundingRect().height()
        )

    def mouseReleaseEvent(self, event):
        self.selected_edge = Qt.Edge(0)
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        edges = self.getEdges(event.pos())
        if not edges:
            self.unsetCursor()
        elif edges in (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.Edge.BottomEdge | Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edges in (Qt.Edge.BottomEdge | Qt.Edge.LeftEdge, Qt.Edge.TopEdge | Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeVerCursor)