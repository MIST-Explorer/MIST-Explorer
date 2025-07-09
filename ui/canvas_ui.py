"""Main Class to handle display of images"""

import typing

import numpy as np
import pandas as pd
from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QMouseEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolTip,
    QWidget,
)

import utils
from ui.lassos.CircleLasso import CircleLasso
from ui.lassos.PolyLasso import PolyLasso
from ui.lassos.RectLasso import RectLasso
from utils import resource_path

if typing.TYPE_CHECKING:
    from ui.app import Ui_MainWindow


class ArrowItem(QGraphicsPixmapItem):
    """Arrow with hover effect"""

    def __init__(self, pixmap, position):
        super().__init__(pixmap)
        rect = QRectF(self.boundingRect())
        self.bg_rect = QGraphicsRectItem(rect, parent=self)
        self.bg_rect.setBrush(QBrush(QColor(255, 255, 255, 100)))
        self.bg_rect.setZValue(1)  # Make sure it goes behind the arrows
        self.base_opacity = 0.4
        self.hover_opacity = 1  # Darker when hovered

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
    """Reference view for displaying images with navigation arrows"""

    image_dropped = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.zoom = 1
        self.right_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/right-arrow.png")).scaled(
                25,
                25,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation,
            ),
            QPointF(250, 275),
        )
        self.left_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/left-arrow.png")).scaled(
                25,
                25,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.SmoothTransformation,
            ),
            QPointF(10, 275),
        )
        self.pixmap_item = QGraphicsPixmapItem()
        self.current_index = 1
        self.np_channels = {}
        self.pixmap = None

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(QSize(300, 300))
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setStyleSheet("QGraphicsView { border: 1px solid black; }")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def is_empty(self) -> bool:
        return self.pixmap_item is None

    def load_channels(self, np_channels):
        self.np_channels = np_channels

    def slideshow(self, zoom=0):
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        if zoom > 0:
            scene.removeItem(self.right_arrow)
            scene.removeItem(self.left_arrow)
        scene.addItem(self.right_arrow)
        scene.addItem(self.left_arrow)

        self.left_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.right_arrow.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self.left_arrow.setScale(zoom)
        self.right_arrow.setScale(zoom)

        self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        if not self.is_empty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event is None:
            return
        elif self.pixmap_item is not None:
            zooming_out = event.angleDelta().y() > 0

            # Prevent excessive zooming in either direction
            if self.zoom > 1.1**90 and zooming_out:  # Max zoom out
                return

            zoom_factor = 1.1 if zooming_out else 0.9
            self.zoom *= zoom_factor
            self.scale(zoom_factor, zoom_factor)
            # self.slideshow(self.zoom)
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
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        if not self.is_empty() and len(self.np_channels) > 1:
            scene_pos = self.mapToScene(event.pos())
            arrow_clicked = False

            # Check if left arrow was clicked
            if self.left_arrow and self.left_arrow.contains(
                self.left_arrow.mapFromScene(scene_pos)
            ):
                self.prev_slide()
                arrow_clicked = True

            # Check if right arrow was clicked
            elif self.right_arrow and self.right_arrow.contains(
                self.right_arrow.mapFromScene(scene_pos)
            ):
                self.next_slide()
                arrow_clicked = True

            # If an arrow was clicked, don't pass the event to parent
            if arrow_clicked:
                return

        # Let parent handle the event (enables panning)
        super().mousePressEvent(event)

    def prev_slide(self):
        """Show previous slide"""
        if self.current_index > 1:
            self.current_index -= 1
            self.update_slide()

    def next_slide(self):
        """Show next slide"""
        if self.current_index < len(self.np_channels.keys()):
            self.current_index += 1
            self.update_slide()

    def update_slide(self):
        """Update displayed image"""
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        scene.clear()  # Clear existing items
        self.pixmap = QPixmap(
            utils.numpy_to_qimage(
                self.np_channels[f"Channel {self.current_index}"].data
            )
        )
        self.pixmap_item.setPixmap(self.pixmap)
        assert isinstance(self.pixmap_item, QGraphicsPixmapItem)
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def __centerImage(self):
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.pixmap_item)

    # def mouseMoveEvent(self, event: QMouseEvent):
    #     """Handle mouse move events for panning"""
    #     if not self.is_empty():
    #         scene_pos = self.mapToScene(event.pos())
    #         image_pos = self.pixmapItem.mapFromScene(scene_pos)

    #         x = int(image_pos.x())
    #         y = int(image_pos.y())
    #         img = self.pixmapItem.pixmap().toImage()

    #         if 0 <= x < img.width() and 0 <= y < img.height():
    #             global_pos = self.mapToGlobal(event.pos())
    #     super().mouseMoveEvent(event)

    def display(self, pixmap: QPixmap, is_layer: bool):
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        scene.clear()
        # reset
        self.current_index = 1

        # if not hasattr(self, "right_arrow"):
        self.slideshow()  # Initialize arrows

        self.pixmap = pixmap
        if not hasattr(self, "pixmapItem") or self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
            scene.addItem(self.pixmap_item)

        else:
            print("setting pixmap")
            self.pixmap_item.setPixmap(self.pixmap)

        print("is layer: ", is_layer)

        # if is_layer:

        print("has np channels")
        # Scale arrows appropriately;  !TODO, this should be done dynamically and repositioned dynamically
        rw = int(scene.width() / 10.6)
        rh = int(scene.height() / 10.6)

        self.right_arrow.bg_rect.setRect(0, 0, rw, rh)
        self.left_arrow.bg_rect.setRect(0, 0, rw, rh)

        self.right_arrow.setPixmap(
            QPixmap(resource_path("assets/icons/right-arrow.png")).scaled(rw, rh)
        )
        self.left_arrow.setPixmap(
            QPixmap(resource_path("assets/icons/left-arrow.png")).scaled(rw, rh)
        )

        scene_height = self.get_scene().height()
        scene_width = self.get_scene().width()
        self.right_arrow.setToolTip("Next")
        self.left_arrow.setToolTip("Previous")

        # Position the arrows
        right_arrow_pos_x = int(scene_width)
        left_arrow_pos_x = 0
        arrow_pos_y = int(scene_height / 2)

        self.right_arrow.setPos(self.mapToScene(right_arrow_pos_x, arrow_pos_y))
        self.left_arrow.setPos(self.mapToScene(left_arrow_pos_x, arrow_pos_y))

        # Set Z-order
        self.pixmap_item.setZValue(0)
        self.right_arrow.setZValue(1)
        self.left_arrow.setZValue(1)

        # Setup scene
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # self.move(int(self.parent.width() - 2*self.parent.width()), 10)

    def highlight_pixel(self, x, y):
        """Highlight the pixel at the given coordinates"""
        # Remove existing pixel highlight if any
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            self.get_scene().removeItem(self.pixel_highlight)

        if self.pixmap_item is None:
            return

        # Create a small rectangle to highlight the pixel
        # Convert pixel coordinates to scene coordinates
        pixel_rect = QRectF(x, y, 1, 1)  # 1x1 pixel
        scene_rect = self.pixmap_item.mapRectToScene(pixel_rect)

        # Create highlight rectangle
        self.pixel_highlight = QGraphicsRectItem(scene_rect)

        # Style the highlight (you can customize this)
        pen = QPen(QColor(255, 255, 0, 180))  # Yellow with transparency
        pen.setWidth(0)  # Cosmetic pen (always 1 pixel wide regardless of zoom)
        pen.setCosmetic(True)
        self.pixel_highlight.setPen(pen)

        # Optional: Add a semi-transparent fill
        brush = QBrush(QColor(255, 255, 0, 50))  # Light yellow fill
        self.pixel_highlight.setBrush(brush)

        # Add to scene
        self.get_scene().addItem(self.pixel_highlight)

    def hide_pixel_highlight(self):
        """Hide the pixel highlight"""
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            self.get_scene().removeItem(self.pixel_highlight)
            self.pixel_highlight = None

    def get_scene(self):
        s = self.scene()
        assert s is not None, "Scene should be initialized"
        return s


class ImageGraphicsViewUI(QGraphicsView):
    """Main image view with support for selection, cropping and other operations"""

    image_dropped = pyqtSignal(str)
    show_crop = pyqtSignal(QRect)
    horizontal_flip = pyqtSignal()  # Signal to request horizontal flip
    vertical_flip = pyqtSignal()  # Signal to request vertical flip
    clear_canvas = pyqtSignal()

    def __init__(self, parent, enc: "Ui_MainWindow", show_buttons=True):
        super().__init__(parent)
        self.enc = enc
        self.show_buttons = show_buttons
        self.setupUI()

        self.pixmap_item: QGraphicsPixmapItem = QGraphicsPixmapItem()
        self.rubberBand = None
        self.rubberBands = []
        self.rubberBandColors = []
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.select = False
        self.zoom = 1
        self.polygons = []
        self.current_polygon = None
        self.rubber_band_positions = []

        # Improved crop system
        self.crop_mode = False
        self.crop_start_pos = None
        self.active_crop_rect = None
        self.is_resizing = False

        self.reference_view = None

        # Setup interaction
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.view_pixmap_item = QGraphicsPixmapItem()

        self.get_scene().addItem(self.view_pixmap_item)
        self.get_scene().addItem(self.pixmap_item)
        self.view_pixmap_item.hide()
        self.pixmap_item.hide()

    def _show_view_tab_image(self):
        self.pixmap_item.hide()
        self.view_pixmap_item.show()

    def _show_images_tab_image(self):
        self.pixmap_item.show()
        self.view_pixmap_item.hide()

    def get_scene(self):
        s = self.scene()
        assert s is not None, "Scene should be initialized"
        return s

    def flip_horizontal(self):
        """Flip the image horizontally"""
        if self.pixmap_item:
            img = self.pixmap_item.pixmap().toImage()
            flipped_img = img.mirrored(horizontal=True, vertical=False)
            flipped_pixmap = QPixmap.fromImage(flipped_img)
            self.pixmap_item.setPixmap(flipped_pixmap)
            self.get_scene().update()
            # Emit signal to update the underlying data model
            self.horizontal_flip.emit()

    def flip_vertical(self):
        """Flip the image vertically"""
        if self.pixmap_item:
            img = self.pixmap_item.pixmap().toImage()
            flipped_img = img.mirrored(horizontal=False, vertical=True)
            flipped_pixmap = QPixmap.fromImage(flipped_img)
            self.pixmap_item.setPixmap(flipped_pixmap)
            self.get_scene().update()
            # Emit signal to update the underlying data model
            self.vertical_flip.emit()

    def setupUI(self):
        self.setMinimumSize(QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True)
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSceneRect(0, 0, 800, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Create floating selection buttons
        if self.show_buttons:
            self.create_floating_buttons()

    def create_floating_buttons(self):
        """Create floating selection buttons that appear over the canvas"""
        # Create a container widget for the buttons
        self.floating_container = QWidget(self)

        # Create horizontal layout for the buttons
        button_layout = QHBoxLayout(self.floating_container)
        button_layout.setContentsMargins(10, 5, 10, 5)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Add label
        label = QLabel("Select region of interest:", self.floating_container)
        label.setStyleSheet(
            "QLabel { color: white; padding: 5px; border-radius: 3px; }"
        )
        button_layout.addWidget(label)

        # Create the selection buttons
        self.rect_button = QPushButton()
        self.circle_button = QPushButton()
        self.poly_button = QPushButton()

        # Set button sizes and styles
        for button in [self.rect_button, self.circle_button, self.poly_button]:
            button.setFixedSize(40, 40)
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.1);
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background-color: rgba(200, 200, 200, 0.2);
                }
            """
            )

        # Set icons for the buttons
        self.rect_button.setIcon(QIcon(resource_path("assets/icons/square.png")))
        self.circle_button.setIcon(QIcon(resource_path("assets/icons/circle.png")))
        self.poly_button.setIcon(QIcon(resource_path("assets/icons/poly.png")))

        # Connect button signals to selection modes
        self.rect_button.clicked.connect(lambda: self.set_selection_mode("rect"))
        self.circle_button.clicked.connect(lambda: self.set_selection_mode("circle"))
        self.poly_button.clicked.connect(lambda: self.enc.poly_select())

        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)

        # Position the container at the top-right of the view
        self.update_floating_buttons_position()

    def update_floating_buttons_position(self):
        """Update the position of the floating buttons"""
        if hasattr(self, "floating_container"):
            # Position at the top-right of the view with some padding
            self.floating_container.move(
                self.width() - self.floating_container.width() - 20, 10
            )

    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_floating_buttons_position()

    def set_selection_mode(self, mode):
        """Set the current selection mode"""
        # Reset all modes
        self.select = False
        self.current_polygon = None
        self.crop_mode = False
        self.begin_crop = False

        # Set the new mode
        if mode == "rect":
            self.select = "rect"
            self.enc.select()
        elif mode == "circle":
            self.select = "circle"
            self.enc.circle_select()
        elif mode == "poly":
            self.select = "poly"
            self.enc.poly_select()

    def start_crop_mode(self):
        """Start crop mode - called from external crop button"""
        self.crop_mode = True
        self.begin_crop = True
        self.select = False
        self.current_polygon = None

        # Clear any existing crop rectangle
        if self.active_crop_rect:
            self.get_scene().removeItem(self.active_crop_rect)
            self.active_crop_rect = None

        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_crop_mode(self):
        """Cancel crop mode and clean up"""
        self.crop_mode = False
        self.begin_crop = False
        self.crop_start_pos = None

        if self.active_crop_rect:
            self.get_scene().removeItem(self.active_crop_rect)
            self.active_crop_rect = None

        self.unsetCursor()

    def isEmpty(self) -> bool:
        return self.pixmap_item is None

    def mouseDoubleClickEvent(self, event):
        if not self.isEmpty():
            self.__centerImage()

    def update_canvas(self, pixmap: QPixmap, is_view=False, crop=False):
        """Updates canvas when current image is operated on"""

        if self.pixmap_item:
            if is_view:
                self._show_view_tab_image()
                self.view_pixmap_item.setPixmap(pixmap)
            else:
                self._show_images_tab_image()
                self.pixmap_item.setPixmap(pixmap)
            self.__centerImage()

    def add_new_image(self, pixmapItem: QGraphicsPixmapItem):
        """Update the pixmap of the existing image or add a new one"""
        self.pixmap_item.setPixmap(pixmapItem.pixmap())
        self.__centerImage()

    def __centerImage(self):
        pixmap_item = (
            self.pixmap_item if self.pixmap_item.isVisible() else self.view_pixmap_item
        )
        item_rect = pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmap_item)
        # if self.reference_view:
        #     self.reference_view.__centerImage()

    def dragEnterEvent(self, event: QDragEnterEvent):  # type: ignore
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):  # type: ignore
        event.acceptProposedAction()

    def dropEvent(self, event):
        if event is None:
            return
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.acceptProposedAction()

    def wheelEvent(self, event):
        if event is None:
            return
        elif self.pixmap_item is None:
            return
        zooming_out = event.angleDelta().y() > 0

        # Prevent excessive zooming in either direction
        if self.zoom > 1.1**90 and zooming_out:  # Max zoom out
            return

        if self.zoom < 1 / (1.1**2) and not zooming_out:  # Max zoom in
            return
        # if self.reference_view:
        #     self.reference_view.wheelEvent(event)
        zoom_factor = 1.1 if zooming_out else 0.9
        self.zoom *= zoom_factor

        # Store rubber band positions before zooming
        if not self.rubber_band_positions:
            self.rubber_band_positions = []
            for rubber_band in self.rubberBands:
                rubber_band_geometry = rubber_band.geometry()
                top_left_scene = self.mapToScene(rubber_band_geometry.topLeft())
                bottom_right_scene = self.mapToScene(rubber_band_geometry.bottomRight())
                self.rubber_band_positions.append(
                    (rubber_band, top_left_scene, bottom_right_scene)
                )

        # Perform the zoom
        self.scale(zoom_factor, zoom_factor)

        # Update rubber band positions after zooming
        for (
            rubber_band,
            top_left_scene,
            bottom_right_scene,
        ) in self.rubber_band_positions:
            new_top_left_view = self.mapFromScene(top_left_scene)
            new_bottom_right_view = self.mapFromScene(bottom_right_scene)
            new_rect = QRect(new_top_left_view, new_bottom_right_view)
            rubber_band.setGeometry(new_rect)

        # Force redraw to update polygon positions
        vp = self.viewport()
        assert vp is not None, "Viewport should be initialized"
        vp.update()

    def create_rubber_band(self, rubber_band_class, shape, x, y, parent, origin):
        """Create a rubber band of the specified class"""
        rubber_band = rubber_band_class(shape, x, y, self)
        rubber_band.setGeometry(QRect(origin, QSize()))
        rubber_band.show()
        return rubber_band

    def update_starting_position(self, event):
        """Update the starting position for rubber band operations"""
        scene_pos = self.mapToScene(event.pos())
        self.image_pos = self.pixmap_item.mapFromScene(scene_pos)
        self.starting_x = int(self.image_pos.x())
        self.starting_y = int(self.image_pos.y())

    def mousePressEvent(self, event: QMouseEvent | None):
        if self.isEmpty() or event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # Handle crop mode
            if self.crop_mode and not self.active_crop_rect:
                scene_pos = self.mapToScene(event.pos())
                self.crop_start_pos = scene_pos

                # Create a new crop rectangle starting from this position
                if self.active_crop_rect:
                    self.get_scene().removeItem(self.active_crop_rect)

                # Create a resizable crop rectangle

                self.active_crop_rect = QGraphicsRectItem()
                self.active_crop_rect.setRect(scene_pos.x(), scene_pos.y(), 0, 0)

                # Style the crop rectangle
                pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine)
                self.active_crop_rect.setPen(pen)
                self.active_crop_rect.setBrush(QBrush(QColor(255, 255, 255, 30)))

                self.get_scene().addItem(self.active_crop_rect)
                self.is_resizing = True

            # Handle regular selection modes
            if self.begin_crop or self.select:
                self.origin = event.pos()
                self.update_starting_position(event)

                if self.begin_crop:
                    if not self.rubberBand:
                        self.rubberBand = RectLasso(self)
                elif self.select == "rect":
                    self.rubberBand = RectLasso(self)
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                elif self.select == "circle":
                    self.center = QPoint(self.starting_x, self.starting_y)
                    self.rubberBand = CircleLasso(self)
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                elif self.select == "poly":
                    if not self.current_polygon:
                        self.current_polygon = PolyLasso(
                            self.pixmap_item
                        )  # Set pixmapItem as parent
                        self.get_scene().addItem(self.current_polygon)
                        # Enable mouse tracking for live preview
                        self.setMouseTracking(True)

                    # Add point in scene coordinates, but relative to the image
                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmap_item.mapFromScene(scene_pos)
                    # Convert image_pos to scene coordinates relative to the image
                    polygon_pos = self.pixmap_item.mapToScene(image_pos)
                    self.current_polygon.add_point(polygon_pos, image_pos)

                if self.begin_crop:
                    self.rubberBands.append(self.rubberBand)
                    self.rubberBandColors.append(self.rubberBand.color)
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()

        if not self.is_resizing and not self.select:
            super().mousePressEvent(event)

        # Propagate event to rubber bands
        for r in self.rubberBands:
            r.mousePressEvent(event)

    def keyPressEvent(self, event):
        # Handle crop confirmation with Enter key
        if (
            event.key() == Qt.Key.Key_Return
            and self.crop_mode
            and self.active_crop_rect
        ):
            self.confirm_crop()
            return

        # Handle crop cancellation with Escape key
        if event.key() == Qt.Key.Key_Escape and self.crop_mode:
            self.cancel_crop_mode()
            return

        super().keyPressEvent(event)

    def confirm_crop(self):
        """Confirm the current crop selection"""
        if not self.active_crop_rect or not self.pixmap_item:
            return

        # Get the crop rectangle in scene coordinates
        crop_rect = self.active_crop_rect.rect()

        # Convert to image coordinates
        image_top_left = self.pixmap_item.mapFromScene(crop_rect.topLeft())
        image_bottom_right = self.pixmap_item.mapFromScene(crop_rect.bottomRight())

        # Create image rectangle and clamp to image bounds
        image_rect = QRect(
            max(0, int(image_top_left.x())),
            max(0, int(image_top_left.y())),
            int(image_bottom_right.x() - image_top_left.x()),
            int(image_bottom_right.y() - image_top_left.y()),
        ).normalized()

        # Make sure rect is within image bounds
        image = self.pixmap_item.pixmap().toImage()
        image_width, image_height = image.width(), image.height()
        image_rect = image_rect.intersected(QRect(0, 0, image_width, image_height))

        if image_rect.isEmpty():
            return

        # Emit the crop signal
        self.show_crop.emit(image_rect)

        # Clean up crop mode
        self.cancel_crop_mode()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

        # Handle crop rectangle resizing
        if (
            self.crop_mode
            and self.crop_start_pos is not None
            and self.active_crop_rect is not None
            and self.is_resizing
        ):
            current_pos = self.mapToScene(event.pos())
            # Update the crop rectangle
            x1, y1 = self.crop_start_pos.x(), self.crop_start_pos.y()
            x2, y2 = current_pos.x(), current_pos.y()

            # Ensure proper rectangle (top-left to bottom-right)
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            self.active_crop_rect.setRect(left, top, width, height)

        # Store current mouse position for polygon preview
        if self.current_polygon and len(self.current_polygon.points) > 0:
            # Update temp point in scene coordinates relative to the image
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmap_item.mapFromScene(scene_pos)
            polygon_pos = self.pixmap_item.mapToScene(image_pos)
            self.current_polygon.set_temp_point(polygon_pos)

        # # Handle pixel info display
        if self.pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmap_item.mapFromScene(scene_pos)

            x = int(image_pos.x())
            y = int(image_pos.y())
            img = self.pixmap_item.pixmap().toImage()

            # Show pixel info in tooltip
            if 0 <= x < img.width() and 0 <= y < img.height():
                color = QColor(img.pixel(x, y))
                r, g, b = color.red(), color.green(), color.blue()

                global_pos = self.mapToGlobal(event.pos())
                QToolTip.showText(global_pos, f"", self)

                # Get layer values if available
                if self.enc and self.enc.toolBarUI.tabButtonGroup.checkedId() == 2:
                    layers = self.enc.view_tab.get_layer_values_at(x, y)
                else:
                    layers = None

                combined_layers = None  # added this so we don't get reference error

                if layers:
                    layers = [f"{layer}: {value[0]}\n" for layer, value in layers]
                    combined_layers = "".join(layers)[:-1]
                    QToolTip.showText(global_pos, combined_layers, self)
                else:
                    QToolTip.showText(global_pos, f"R: {r}, G: {g}, B: {b}", self)

                # Update position display in main window
                if combined_layers:
                    combined_layers = combined_layers.replace("\n", ", ")
                    combined_layers += ";"
                    self.enc.updateMousePositionLabel(
                        f"{combined_layers} X: {x}, Y: {y}"
                    )
                else:
                    self.enc.updateMousePositionLabel(
                        f"R: {r}, G: {g}, B: {b} X: {x}, Y: {y}"
                    )

                # Highlight the pixel under cursor
                self.highlight_pixel(x, y)
                # if self.reference_view:
                #     self.reference_view.highlight_pixel(x, y)
            else:
                self.enc.updateMousePositionLabel(f"")
                # Hide pixel highlight when outside image bounds
                self.hide_pixel_highlight()
                # if self.reference_view:
                # self.reference_view.hide_pixel_highlight()
        # Handle rubber band updates for old crop system
        if (
            not self.isEmpty()
            and self.begin_crop
            and self.rubberBand
            and not self.crop_mode
        ):
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

        if (
            (self.select == "rect" or self.select == "circle")
            and self.rubberBands
            and self.origin is not None
        ):
            if self.select == "circle":
                center = self.origin
                corner = event.pos()
                size = (
                    max(abs(center.x() - corner.x()), abs(center.y() - corner.y())) * 2
                )
                self.rubberBands[-1].setGeometry(
                    QRect(center.x() - size // 2, center.y() - size // 2, size, size)
                )
            else:
                self.rubberBands[-1].setGeometry(
                    QRect(self.origin, event.pos()).normalized()
                )

        # Propagate event to rubber bands when not in selection mode
        if not self.select:
            for r in self.rubberBands:
                r.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.isEmpty():
            return
        super().mouseReleaseEvent(event)

        self.rubber_band_positions = []
        # Handle crop mode mouse release
        if self.crop_mode and self.active_crop_rect and self.is_resizing:
            # Don't auto-confirm, let user press Enter or Escape
            self.initial_crop = True
            self.initial_crop_rect = self.active_crop_rect.rect()
            self.scene().removeItem(self.active_crop_rect)

            self.active_crop_rect = ResizableRect(
                self.initial_crop_rect.x(),
                self.initial_crop_rect.y(),
                self.initial_crop_rect.width(),
                self.initial_crop_rect.height(),
            )
            self.active_crop_rect.setZValue(10)
            self.is_resizing = False

            self.scene().addItem(self.active_crop_rect)
            return
        if not self.rubberBands:
            return

        # Propagate event to rubber bands
        for r in self.rubberBands:
            r.mouseReleaseEvent(event)

        if event.button() == Qt.MouseButton.LeftButton:
            rubberband = self.rubberBand if self.begin_crop else self.rubberBands[-1]

            if self.select:
                self.origin = None

                if self.select == "rect" or self.select == "circle":
                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmap_item.mapFromScene(scene_pos)
                    image_rect = (
                        self.select,
                        (
                            self.starting_x,
                            self.starting_y,
                            int(image_pos.x()),
                            int(image_pos.y()),
                        ),
                    )
                    if not self.enc.analysis_tab.analyze_region(rubberband, image_rect):
                        self.rubberBands.remove(self.rubberBand)
                        self.rubberBandColors.pop()
                        if rubberband:
                            rubberband.deleteLater()
                    self.select = False
                    return

    def loadChannels(self, np_channels):
        """Load channel data"""
        self.np_channels = np_channels
        if self.pixmap_item is not None:
            self.__centerImage()

    def highlight_pixel(self, x, y):
        """Highlight the pixel at the given coordinates"""
        # Remove existing pixel highlight if any
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            self.get_scene().removeItem(self.pixel_highlight)
        if self.pixmap_item is None:
            return

        # Create a small rectangle to highlight the pixel
        # Convert pixel coordinates to scene coordinates
        pixel_rect = QRectF(x, y, 1, 1)  # 1x1 pixel
        scene_rect = self.pixmap_item.mapRectToScene(pixel_rect)

        # Create highlight rectangle
        self.pixel_highlight = QGraphicsRectItem(scene_rect)

        # Style the highlight (you can customize this)
        pen = QPen(QColor(255, 255, 0, 180))  # Yellow with transparency
        pen.setWidth(0)  # Cosmetic pen (always 1 pixel wide regardless of zoom)
        pen.setCosmetic(True)
        self.pixel_highlight.setPen(pen)

        # Optional: Add a semi-transparent fill
        brush = QBrush(QColor(255, 255, 0, 50))  # Light yellow fill
        self.pixel_highlight.setBrush(brush)

        # Add to scene
        self.get_scene().addItem(self.pixel_highlight)

    def hide_pixel_highlight(self):
        """Hide the pixel highlight"""
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            self.get_scene().removeItem(self.pixel_highlight)
            self.pixel_highlight = None


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, cursor_shape: Qt.CursorShape, parent: QGraphicsItem):
        """Resize handle for ResizableRect"""
        w, h = parent.boundingRect().width(), parent.boundingRect().height()
        w, h = w // 16, h // 16
        w, h = max(16, w), max(16, h)  # Ensure minimum size

        super().__init__(-w // 2, -h // 2, w, h, parent)  # Center the handle
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setZValue(11)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.cursor_shape = cursor_shape
        self.edge = Qt.Edge(0)
        self.parent = parent

    def setEdge(self, edge):
        """Set the edge this handle is associated with"""
        self.edge = edge

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(QCursor(self.cursor_shape))

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def mousePressEvent(self, event):
        """Handle mouse press events on the resize handle"""
        self.parent.selected_edge = self.edge
        # super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events on the resize handle"""
        self.parent.selected_edge = Qt.Edge(0)
        # super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events on the resize handle"""
        super().mouseMoveEvent(event)
        # Update the parent rectangle's position if needed
        self.parent.mouseMoveEvent(event)


class ResizableRect(QGraphicsRectItem):
    def __init__(self, x=0.0, y=0.0, width=0.0, height=0.0, onCenter=False):
        if onCenter:
            super().__init__(-width / 2, -height / 2, width, height)
        else:
            super().__init__(x, y, width, height)

        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.selected_edge = None
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QBrush(Qt.GlobalColor.blue), 3, Qt.PenStyle.DotLine))
        self.setBrush(QBrush(QColor(0, 255, 0, 30)))
        # Create 8 resize handles with correct cursor
        self.handles = []
        cursor_shapes = [
            Qt.CursorShape.SizeFDiagCursor,  # top-left
            Qt.CursorShape.SizeVerCursor,  # top-center
            Qt.CursorShape.SizeBDiagCursor,  # top-right
            Qt.CursorShape.SizeHorCursor,  # mid-right
            Qt.CursorShape.SizeFDiagCursor,  # bottom-right
            Qt.CursorShape.SizeVerCursor,  # bottom-center
            Qt.CursorShape.SizeBDiagCursor,  # bottom-left
            Qt.CursorShape.SizeHorCursor,  # mid-left
        ]
        edges = [
            Qt.Edge.TopEdge | Qt.Edge.LeftEdge,  # top-left
            Qt.Edge.TopEdge,  # top-center
            Qt.Edge.TopEdge | Qt.Edge.RightEdge,  # top-right
            Qt.Edge.RightEdge,  # mid-right
            Qt.Edge.BottomEdge | Qt.Edge.RightEdge,  # bottom-right
            Qt.Edge.BottomEdge,  # bottom-center
            Qt.Edge.BottomEdge | Qt.Edge.LeftEdge,  # bottom-left
            Qt.Edge.LeftEdge,  # mid-left
        ]

        for cursor, edge in zip(cursor_shapes, edges):
            handle = ResizeHandle(cursor, self)
            handle.setEdge(edge)
            self.handles.append(handle)

        self.updateHandles()

    def updateHandles(self):
        rect = self.rect()
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        positions = [
            QPointF(x, y),  # top-left
            QPointF(x + w / 2, y),  # top-center
            QPointF(x + w, y),  # top-right
            QPointF(x + w, y + h / 2),  # mid-right
            QPointF(x + w, y + h),  # bottom-right
            QPointF(x + w / 2, y + h),  # bottom-center
            QPointF(x, y + h),  # bottom-left
            QPointF(x, y + h / 2),  # mid-left
        ]

        for handle, pos in zip(self.handles, positions):
            handle.setPos(pos)
            handle.setRect(
                -handle.rect().width() / 2,
                -handle.rect().height() / 2,
                handle.rect().width(),
                handle.rect().height(),
            )

    def setRect(self, rect):
        super().setRect(rect)
        self.updateHandles()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selected_edge:
            rect = self.rect()
            pos = self.mapFromScene(event.scenePos())
            new_rect = QRectF(rect)

            if self.selected_edge & Qt.Edge.LeftEdge:
                diff = pos.x()
                new_width = rect.right() - diff
                if new_width > 10:
                    new_rect.setLeft(diff)

            if self.selected_edge & Qt.Edge.RightEdge:
                diff = pos.x()
                new_width = diff - rect.left()
                if new_width > 10:
                    new_rect.setRight(diff)

            if self.selected_edge & Qt.Edge.TopEdge:
                diff = pos.y()
                new_height = rect.bottom() - diff
                if new_height > 10:
                    new_rect.setTop(diff)

            if self.selected_edge & Qt.Edge.BottomEdge:
                diff = pos.y()
                new_height = diff - rect.top()
                if new_height > 10:
                    new_rect.setBottom(diff)

            self.setRect(new_rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.selected_edge = Qt.Edge(0)
        super().mouseReleaseEvent(event)
        self.updateHandles()

    def getEdges(self, pos):
        """Fallback edge hit detection for resize dragging (not hover)"""
        edges = Qt.Edge(0)
        rect = self.rect()
        buffer = 30

        if pos.x() < rect.x() + buffer:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() > rect.right() - buffer:
            edges |= Qt.Edge.RightEdge
        if pos.y() < rect.y() + buffer:
            edges |= Qt.Edge.TopEdge
        elif pos.y() > rect.bottom() - buffer:
            edges |= Qt.Edge.BottomEdge

        return edges
