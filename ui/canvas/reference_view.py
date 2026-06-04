"""
Reference graphics view module.
"""

import logging

import pyqtgraph as pg

# pylint: disable=no-name-in-module, missing-final-newline, fixme
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsView,
    QSizeGrip,
)

import utils
from ui.canvas.items import ArrowItem
from utils import resource_path

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class ReferenceGraphicsViewUI(QGraphicsView):
    """Reference view for displaying images with navigation arrows"""

    image_dropped = pyqtSignal(str)
    channel_changed = pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.zoom = 1
        self.right_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/right-arrow.png")),
            QPointF(285, 150),
            self.next_slide,  # Action to perform on click
        )
        self.left_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/left-arrow.png")),
            QPointF(15, 150),
            self.prev_slide,  # Action to perform on click
        )
        self.pixmap_item: QGraphicsPixmapItem = QGraphicsPixmapItem()
        self.current_index = 1
        self.np_channels = {}
        self.pixmap = None
        self.right_arrow.setZValue(999)  # Ensure arrows are above the image
        self.left_arrow.setZValue(999)  # Ensure arrows are above the image
        self.pixel_highlight = None

        self._resizing = False
        self._resize_start_pos = None
        self._initial_size = None
        self._hide_threshold = 100

        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        self.setMinimumSize(QSize(0, 0))
        # self.setMaximumSize(QSize(300, 300))
        self.resize(300, 300)
        self.setScene(pg.GraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        from ui.theme import ThemeManager
        tc = ThemeManager.instance().get_current()
        self.setStyleSheet(f"QGraphicsView {{ border: 1px solid {tc['border']}; }}")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Add size grip for visual handle only - we'll override its behavior
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet(f"""
            QSizeGrip {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0.5 transparent,
            stop:0.5 {tc["size_grip_color"]});
            width: 16px;
            height: 16px;
            }}
        """)
        self.size_grip.setCursor(Qt.CursorShape.SizeFDiagCursor)
        # Install event filter to intercept size grip events
        self.size_grip.installEventFilter(self)
        self.setAcceptDrops(True)

        # self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def eventFilter(self, obj, event):
        """Filter events from the size grip to handle resize manually"""
        if hasattr(self, "size_grip") and obj == self.size_grip:
            if event.type() == event.Type.MouseButtonPress:
                self._resizing = True
                self._resize_start_pos = event.globalPosition().toPoint()
                self._initial_size = self.size()
                return True  # Block the event from size grip
            elif event.type() == event.Type.MouseButtonRelease:
                self._resizing = False
                return True
            elif event.type() == event.Type.MouseMove and self._resizing:
                delta = event.globalPosition().toPoint() - self._resize_start_pos
                new_width = max(50, self._initial_size.width() + delta.x())
                new_height = max(50, self._initial_size.height() + delta.y())

                # Manually resize without affecting parent layout
                self.setFixedSize(new_width, new_height)
                return True
        return super().eventFilter(obj, event)

    def is_empty(self) -> bool:
        """Check if the view is empty"""
        return (
            self.pixmap_item is None
            or self.pixmap_item.pixmap().isNull()
            or self.pixmap_item.pixmap().width() == 0
        )

    def load_channels(self, np_channels):
        """Load channels"""
        self.np_channels = np_channels

    # pylint: disable=invalid-name, unused-argument
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click event"""
        if not self.is_empty():
            self._center_image()
            self.position_arrows()

    # pylint: disable=invalid-name
    def wheelEvent(self, event):
        """Handle wheel event for zooming"""
        if event is None:
            return
        if self.pixmap_item is not None:
            logger.debug("Reference view wheel event: %d", event.angleDelta().y())
            # y > 0 is scroll UP (Zoom IN), y < 0 is scroll DOWN (Zoom OUT)
            zooming_out = event.angleDelta().y() < 0

            # Prevent excessive zooming out (image too small)
            if self.zoom <= 0.1 and zooming_out:
                return

            # Prevent excessive zooming in (image too large)
            if self.zoom >= 5000 and not zooming_out:
                return

            zoom_factor = 0.9 if zooming_out else 1.1
            self.zoom *= zoom_factor
            self.scale(zoom_factor, zoom_factor)
            self.position_arrows()
            self.viewport().update()
        else:
            super().wheelEvent(event)

    # pylint: disable=invalid-name
    def dragEnterEvent(self, event: QDragEnterEvent | None):
        """Handle drag enter event"""
        if event is not None:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()

    # pylint: disable=invalid-name
    def dragMoveEvent(self, event: QDragMoveEvent | None):
        """Handle drag move event"""
        if event is not None:
            mime = event.mimeData()
            if mime and mime.hasUrls():
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()

    # pylint: disable=invalid-name
    def dropEvent(self, event: QDropEvent | None):
        """Handle drop event"""
        if event is not None and event.mimeData() is not None:
            mime = event.mimeData()
            if mime is None:
                return
            event_urls = mime.urls()
            for url in event_urls:
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

    def position_arrows(self):
        """Position the navigation arrows"""
        vp = self.viewport()
        assert vp is not None, "Viewport should be initialized"
        view_width = vp.width()
        view_height = vp.height()

        y_center = self.mapToScene(QPoint(0, view_height // 2)).y()

        left_x = self.mapToScene(QPoint(10, 0)).x()
        right_x = self.mapToScene(QPoint(view_width - 40, 0)).x()

        self.left_arrow.setPos(
            QPointF(left_x, y_center - 15)
        )  # -15 to vertically center 30px arrow
        self.right_arrow.setPos(QPointF(right_x, y_center - 15))

    def prev_slide(self):
        """Show previous slide"""
        if self.current_index > 1:
            self.current_index -= 1
            self.update_slide()
            self.channel_changed.emit(self.current_index)

    def next_slide(self):
        """Show next slide"""
        if self.current_index < len(self.np_channels.keys()):
            self.current_index += 1
            self.update_slide()
            self.channel_changed.emit(self.current_index)

    def arrow_visibility(self):
        """Update visibility of arrows based on current index"""
        if self.current_index == 1:
            self.left_arrow.hide()
        else:
            self.left_arrow.show()

        if self.current_index == len(self.np_channels.keys()):
            self.right_arrow.hide()
        else:
            self.right_arrow.show()

    def update_slide(self):
        """Update displayed image"""
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        # scene.removeItem(self.pixmap_item)  # Clear previous image
        self.pixmap = QPixmap(
            utils.numpy_to_qimage(
                utils.to_uint8(self.np_channels[f"Channel {self.current_index}"].data)
            )
        )
        self.pixmap_item.setPixmap(self.pixmap)
        assert isinstance(self.pixmap_item, QGraphicsPixmapItem)
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(item_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom = 1.0  # Reset zoom level when fitting to view
        self.arrow_visibility()

    def _center_image(self):
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.pixmap_item)
        self.zoom = 1  # Reset zoom level when centering

    def display(self, pixmap: QPixmap, channel_index: int = 1):
        """Display the given pixmap"""
        self.show()
        # self.raise_()
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        scene.clear()  # Clear previous image
        # reset
        self.current_index = channel_index

        # if not hasattr(self, "right_arrow"):
        # self.slideshow()  # Initialize arrows

        self.pixmap = pixmap
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        scene.addItem(self.pixmap_item)

        logger.debug("has np channels")
        # TODO: Scale and reposition arrows dynamically

        # Setup scene
        item_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(item_rect)
        self.fitInView(item_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom = 1.0  # Reset zoom level when fitting to view
        if self.pixmap.width() > 0:
            self.add_arrows()
            self.position_arrows()

    def add_arrows(self):
        """Add navigation arrows to the scene"""
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        self.right_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/right-arrow.png")),
            QPointF(285, 150),
            self.next_slide,
        )
        self.left_arrow = ArrowItem(
            QPixmap(resource_path("assets/icons/left-arrow.png")),
            QPointF(15, 150),
            self.prev_slide,
        )
        scene.addItem(self.right_arrow)
        scene.addItem(self.left_arrow)
        self.arrow_visibility()

    # pylint: disable=invalid-name
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        self.setFocus()
        super().mousePressEvent(event)

    # pylint: disable=invalid-name
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)

        # Position the size grip in the bottom-right corner
        grip_size = self.size_grip.sizeHint()
        self.size_grip.move(
            self.width() - grip_size.width(), self.height() - grip_size.height()
        )

        # Re-fit image and reposition arrows when resized
        if not self.is_empty():
            self.fitInView(
                self.pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            # Apply the preserved zoom factor on top of the fit-in-view scale
            if self.zoom != 1.0:
                self.scale(self.zoom, self.zoom)
        self.position_arrows()

    def get_scene(self):
        """Get the scene"""
        s = self.scene()
        assert s is not None, "Scene should be initialized"
        return s

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
