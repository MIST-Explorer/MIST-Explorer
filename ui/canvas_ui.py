from PyQt6.QtWidgets import QToolTip, QGraphicsView, QRubberBand, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,  QGraphicsRectItem, QGraphicsOpacityEffect, QGraphicsItemGroup, QGraphicsSimpleTextItem
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDragMoveEvent, QMouseEvent, QCursor, QImage, QPalette, QPainter, QBrush, QColor, QPen, QIcon
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal, pyqtSlot, QPointF, QPropertyAnimation, QEasingCurve, QRectF, QSizeF
import ui.Dialogs as Dialogs, numpy as np, matplotlib as mpl, cv2
import ui.Dialogs as Dialogs
import numpy as np
import cv2
from core.Worker import Worker
import utils
import random
import pandas as pd

import traceback

from PyQt6.QtWidgets import QApplication, QRubberBand, QMainWindow
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCore import QRect, QPoint, Qt, QSize


from ui.lassos.CircleLasso import CircleLasso
from ui.lassos.RectLasso import RectLasso
from ui.lassos.PolyLasso import PolyLasso

class ArrowItem(QGraphicsPixmapItem):
    """Arrow with hover effect"""
    def __init__(self, pixmap, position):
        super().__init__(pixmap)
        rect = QRectF(self.boundingRect())
        self.bg_rect = QGraphicsRectItem(rect, parent=self)
        self.bg_rect.setBrush(QBrush(QColor(255, 255, 255, 100)))  
        self.bg_rect.setZValue(1)  #make sure it goes behind the arrows
        self.base_opacity = 0.4
        self.hover_opacity = 1  # darker when hovered
        self.setPos(position)
        self.applyHoverEffect()

    
    def applyHoverEffect(self):
        self.effect = QGraphicsOpacityEffect()
        self.effect.setOpacity(self.base_opacity)
        self.setGraphicsEffect(self.effect)

        self.bg_rect.setGraphicsEffect(self.effect)

        self.fade_in = QPropertyAnimation(self.effect, b"opacity")
        self.fade_in.setDuration(300)
        self.fade_in.setEndValue(self.hover_opacity)
        self.fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)


        self.fade_out = QPropertyAnimation(self.effect, b"opacity")
        self.fade_out.setDuration(300)
        self.fade_out.setEndValue(self.base_opacity)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fade_out.stop() 
        self.fade_in.start()  

    def hoverLeaveEvent(self, event):
        self.fade_in.stop()   
        self.fade_out.start() 


class ReferenceGraphicsViewUI(QGraphicsView):
    
    imageDropped = pyqtSignal(str)  
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom = 1
        self.left_arrow = None
        self.right_arrow = None
        self.pixmapItem = None
        self.current_index = 1
        self.np_channels = {}
        self.pixmap = None
        self.setMinimumSize(QSize(300, 300))
        self.setScene(QGraphicsScene(self))
        # self.setSceneRect(0, 0, 300, 300)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")
        
        self.parent = parent
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse) 



    def isEmpty(self) -> bool:
        return self.pixmapItem == None
    def load_channels(self, np_channels):
        self.np_channels = np_channels

    def slideshow(self):

        if not self.right_arrow  == None:
            return
        
        self.right_arrow = ArrowItem(QPixmap("assets/icons/right-arrow.png").scaled(25,25, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation),
                                     QPointF(250, 275))
        
        self.left_arrow = ArrowItem(QPixmap("assets/icons/left-arrow.png").scaled(25,25, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation), 
                                    QPointF(10, 275))
        



        self.scene().addItem(self.right_arrow)
        self.scene().addItem(self.left_arrow)

        print("scene: ", self.scene().width()/self.right_arrow.pixmap().width())

        self.left_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.right_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


    def mouseDoubleClickEvent(self, event):
        if not self.isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


    def wheelEvent(self, event):
        if self.pixmapItem != None:
            zooming_out = event.angleDelta().y() > 0

            # this is to prevent zooming in too much or zooming out too much
            if self.zoom > 1.1**90 and zooming_out: # if max zoomed out, and not zooming in, quit. not as nessecary
                return
            
            if self.zoom < 1/(1.1**2) and not zooming_out: # if max zoomed in, and not zooming out, quit.
                return

            zoom_factor = 1.1 if zooming_out else 0.9

            self.zoom *= zoom_factor

            self.scale(zoom_factor, zoom_factor)
        else:
            super().wheelEvent(event)

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


    def mousePressEvent(self, event):
        if not self.isEmpty():
            scene_pos = self.mapToScene(event.pos())
            if self.left_arrow.contains(self.left_arrow.mapFromScene(scene_pos)):
                print("prev")
                self.prev_slide()
            elif self.right_arrow.contains(self.right_arrow.mapFromScene(scene_pos)):
                print("next")
                self.next_slide()
            super().mousePressEvent(event)

    def prev_slide(self):
        """Show previous slide"""
        if self.current_index > 1:
            self.current_index -= 1
            print("prev")
            self.update_slide()

    def next_slide(self):
        """Show next slide"""
        if self.current_index < len(self.np_channels.keys()):
            print("next")
            self.current_index += 1
            self.update_slide()

    def update_slide(self):
        """Update displayed image"""
        self.pixmap = QPixmap(utils.numpy_to_qimage(self.np_channels[f"Channel {self.current_index}"]))

        self.pixmapItem.setPixmap(self.pixmap)

        item_rect = self.pixmapItem.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


    def display(self, pixmapItem: QGraphicsPixmapItem):

        self.slideshow() # init arrows

        if self.pixmapItem == None:
            self.pixmapItem = pixmapItem
            self.pixmap = self.pixmapItem.pixmap()

        else:
            self.pixmapItem.setPixmap(pixmapItem.pixmap())

        self.scene().addItem(self.pixmapItem)
        
        rw = int(self.scene().width()/10.6)
        rh = int(self.scene().height()/10.6) # scale arrows up
        print(rw, rh)
        self.right_arrow.bg_rect.setRect(0, 0, rw, rh) 
        self.left_arrow.bg_rect.setRect(0, 0, rw, rh)

        self.right_arrow.setPixmap(QPixmap("assets/icons/right-arrow.png").scaled(rw,rh))
        self.left_arrow.setPixmap(QPixmap("assets/icons/left-arrow.png").scaled(rw,rh))

        scene_height = self.scene().height()
        scene_width = self.scene().width()
        self.right_arrow.setToolTip("Next")
        self.left_arrow.setToolTip("Previous")


        right_arrow_pos_x = int(scene_width)
        left_arrow_pos_x = 0
        arrow_pos_y = int(scene_height / 2)

        self.right_arrow.setPos(self.mapToScene(right_arrow_pos_x, arrow_pos_y))
        self.left_arrow.setPos(self.mapToScene(left_arrow_pos_x, arrow_pos_y))
        item_rect = self.pixmapItem.boundingRect()


        self.setSceneRect(item_rect)

        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # rect_item = QGraphicsRectItem(self.sceneRect())
        # rect_item.setPen(QPen(Qt.GlobalColor.blue, 2, Qt.PenStyle.DashLine))
        # self.scene().addItem(rect_item)
        

        # make sure arrows go on top of image
        self.pixmapItem.setZValue(0)
        self.right_arrow.setZValue(2)
        self.left_arrow.setZValue(2)

        # self.posItem = QGraphicsSimpleTextItem("hey", parent=self.pixmapItem)
        # self.posItem.setPos(0,0
            
        # )
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
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.select = False
        self.circle_select = False
        self.zoom = 1

        self.polygons = []
        self.current_polygon = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        

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

        # if not self.pixmapItem.pixmap().isNull():
        #     print("addNewImage: there is a pixmapItem")
        # else:
        #     print("addNewImage; there is no pixmapItem")

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


    # shouldnt be an instance method, rather a class method
    def create_rubber_band(self, rubber_band_class, shape, x, y, parent, origin):
        rubber_band = rubber_band_class(shape, x, y, self)
        rubber_band.setGeometry(QRect(origin, QSize()))
        rubber_band.show()
        return rubber_band

    def update_starting_position(self, event):
        scene_pos = self.mapToScene(event.pos())
        self.image_pos = self.pixmapItem.mapFromScene(scene_pos)
        self.starting_x = int(self.image_pos.x())
        self.starting_y = int(self.image_pos.y())

    def mousePressEvent(self, event: QMouseEvent):

        print("mousePressEvent: entered", self.isEmpty(), self.begin_crop, self.select)

        if event.button() == Qt.MouseButton.LeftButton:

            if self.begin_crop or self.select or self.circle_select:

                self.origin = event.pos()
                self.update_starting_position(event)

                if self.begin_crop:
                    self.update_starting_position(event)

                    if not self.rubberBand: 
                        self.rubberBand = RectLasso(self)
                    

                elif self.select == "rect": 
                    self.rubberBand = RectLasso(self)
                elif self.select == "circle": 
                    self.center = QPoint(self.starting_x, self.starting_y)
                    self.rubberBand = CircleLasso(self)
                elif self.select == "poly": 
                    # self.rubberband = PolyLasso()
                    if not self.current_polygon:
                        self.current_polygon = PolyLasso()

                    self.current_polygon.add_point(event.pos())
                    self.painter = QPainter(self)
                    self.current_polygon.draw(self.painter)
                    # self.current_polygon.draw(self.painter)
                    print("repaint?")
                    # self.viewport.repaint()

                    # self.paintEvent(self.p_event)
                    self.repaint()
                    # self.update()   

                    # event.acceptProposedAction()

                    super().mousePressEvent(event)
                    self.mouseMoveEvent(event)

                    return 


                self.rubberBands.append(self.rubberBand)
                self.rubberBandColors.append(self.rubberBand.color)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()

                return
                    
                

            super().mousePressEvent(event)


        
        for r in self.rubberBands:
            r.mousePressEvent(event)

    def keyPressEvent(self, event):
        print(f"Key press event: {event.key()}")
        if event.key() == Qt.Key.Key_Return and self.current_polygon:
            print("Enter key pressed - finalizing polygon")
            self.current_polygon.completed = True
            self.polygons.append(self.current_polygon)
            self.current_polygon = None
            self.repaint()

            # self.paint()



    def paint(self):

        self.painter = QPainter(self)
        
        # Explicitly draw on the viewport
        self.painter.begin(self.viewport())
        
        # Draw polygons
        for polygon in self.polygons:
            print(f"Drawing stored polygon with {len(polygon.points)} points")
            polygon.draw(self.painter)
        
        # Draw current polygon if exists
        if self.current_polygon:
            print(f"Drawing current polygon with {len(self.current_polygon.points)} points")
            self.current_polygon.draw(self.painter)

    def paintEvent(self, event):



        
        print(f"Paint event: Polygons count = {len(self.polygons)}")
        print(f"Current polygon = {self.current_polygon}")
        
        super().paintEvent(event)
        
        painter = QPainter(self)
        
        # Explicitly draw on the viewport
        painter.begin(self.viewport())
        
        # Draw polygons
        for polygon in self.polygons:
            print(f"Drawing stored polygon with {len(polygon.points)} points")
            polygon.draw(painter)
        
        # Draw current polygon if exists
        if self.current_polygon:
            print(f"Drawing current polygon with {len(self.current_polygon.points)} points")
            self.current_polygon.draw(painter)

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
            center = self.origin
            corner = event.pos()
            size = max(abs(center.x() - corner.x()), abs(center.y() - corner.y())) * 2
            self.rubberBands[-1].setGeometry(QRect(center.x() - size // 2, center.y() - size // 2, size, size))


        else:
            # pass
            super().mouseMoveEvent(event)

        if not self.select and not self.circle_select:
            for r in self.rubberBands:
                r.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)

        self.rubber_band_positions = []

        if len(self.rubberBands) == 0:
            return

        for r in self.rubberBands:
            r.mouseReleaseEvent(event)
        
        if event.button() == Qt.MouseButton.LeftButton:
            rubberband = self.rubberBand if self.begin_crop else self.rubberBands[-1]
             
            if self.begin_crop:
                rubberband.hide() 

                selectedRect = rubberband.geometry() 
                if selectedRect.isEmpty():
                    return

                # view_rect = self.viewport().rect()            
                # x_ratio = self.pixmapItem.pixmap().width() / view_rect.width()
                # y_ratio = self.pixmapItem.pixmap().height() / view_rect.height()

                # left = int(selectedRect.left() * x_ratio)
                # top = int(selectedRect.top() * y_ratio)

                # height = int(selectedRect.height() * y_ratio)
                # width = int(selectedRect.width() * x_ratio)
                # self.__qt_image_rect = QRect(left, top, width, height)

                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmapItem.mapFromScene(scene_pos)
            

                self.image_rect = QRect(int(self.starting_x), 
                                        int(self.starting_y), 
                                        int(image_pos.x()), 
                                        int(image_pos.y()))
                
                self.showCroppedImage(self.image_rect)

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
        cropped = pix.copy(self.image_rect).toImage()
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
            self.endCrop()
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