"Image graphics view module."

import logging
import typing

import numpy as np
import pyqtgraph as pg
import tifffile
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QIcon,
    QImage,
    QMouseEvent,
    QPalette,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from core.project_naming import is_segmentation_channel
from ui.canvas.items import CropRectItem, ResizableRect
from ui.lassos.CircleLasso import CircleLasso
from ui.lassos.PolyLasso import PolyLasso
from ui.lassos.RectLasso import RectLasso
from utils import resource_path

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ui.app import MainWindow

TOOLTIP_CURSOR_OFFSET = QPoint(16, 24)
TOOLTIP_BOX_PADDING_PX = 6

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("useOpenGL", True)

# Only enable CuPy if the library is available and can be imported
try:
    has_cupy = True
except Exception:
    has_cupy = False
pg.setConfigOption("useCupy", has_cupy)

pg.setConfigOption("useNumba", False)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class ImageGraphicsViewUI(QGraphicsView):
    """Main image view with support for selection, cropping and other operations"""

    image_dropped = pyqtSignal(str)
    show_crop = pyqtSignal(QRect)
    horizontal_flip = pyqtSignal()  # Signal to request horizontal flip
    vertical_flip = pyqtSignal()  # Signal to request vertical flip
    clear_canvas = pyqtSignal()
    sigRangeChanged = pyqtSignal(object)  # placeholder for pyqtgraph compatibility
    sigTransformChanged = pyqtSignal()  # placeholder for pyqtgraph compatibility

    @property
    def select(self):
        """Get the current selection mode"""
        return getattr(self, "_select", False)

    @select.setter
    def select(self, value):
        """Set the current selection mode and update cursor/drag mode"""
        self._select = value
        if value:
            # Disable drag-panning and show crosshair cursor when in selection mode
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            # Re-enable drag-panning and revert cursor when not in selection mode
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()

    # pylint: disable=too-many-instance-attributes
    def __init__(self, parent, enc: "MainWindow", show_buttons=True):
        super().__init__(parent)
        self.enc = enc
        self.show_buttons = show_buttons

        self.pixmap_item = QGraphicsPixmapItem(QPixmap())
        self.rubber_band = None
        self.rubber_bands = []
        self.rubber_band_colors = []
        self.polygon_colors = []
        self.begin_crop = False
        self.origin = None
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self._select = False
        self.zoom = 1
        self.polygons = []
        self.current_polygon = None
        self.rubber_band_positions = []
        self.select_start_pos = None

        # Improved crop system
        self.crop_mode = False
        self.crop_start_pos = None
        self.active_crop_rect = None
        self.is_resizing = False

        # Move ROI state
        self.moving_lasso = None
        self.last_mouse_pos = None
        self._roi_origin_scene = None  # Scene-space origin for rect/circle drag

        self.reference_view = None

        # Setup interaction
        # self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.view_pixmaps = []
        # self.view_pixmap_item = QGraphicsPixmapItem()

        # self.get_scene().addItem(self.view_pixmap_item)
        self.view_mode = False

        # Attributes initialized
        self.floating_container = None
        self.rect_button = None
        self.circle_button = None
        self.poly_button = None
        self.image_pos = None
        self.starting_x = 0
        self.starting_y = 0
        self.center = None
        self.initial_crop = None
        self.initial_crop_rect = None
        self.np_channels = None
        self.pixel_highlight = None
        self._last_tooltip_text = None
        self._last_tooltip_pos = None
        self._tooltip_label = None
        self.setup_ui()
        self.get_scene().addItem(self.pixmap_item)
        self.pixmap_item.show()

    def show_view_tab_image(self):
        """Show the view tab image."""
        self.pixmap_item.hide()
        for pixmap in self.view_pixmaps:
            pixmap.show()

        # Update the view transform to properly fit the view_pixmaps
        if self.view_pixmaps:
            self._center_image()

        # Show rubberbands when in view/analyze tabs
        for rubber_band in self.rubber_bands:
            rubber_band.show()
            rubber_band.setZValue(1)

        # Show polygons when in view/analyze tabs
        for polygon in self.polygons:
            polygon.show()
            polygon.setZValue(1)
        if self.current_polygon is not None:
            self.current_polygon.show()
            self.current_polygon.setZValue(1)

    def show_images_tab_image(self):
        """Show the images tab image."""
        self.pixmap_item.show()
        for pixmap in self.view_pixmaps:
            pixmap.hide()

        # Also hide rubberbands when switching to Extract tab
        for rubber_band in self.rubber_bands:
            rubber_band.hide()

        # Hide polygons when switching to Extract tab
        for polygon in self.polygons:
            polygon.hide()
        if self.current_polygon is not None:
            self.current_polygon.hide()

    def get_scene(self):
        """Get the scene."""
        scene_obj = self.scene()
        assert scene_obj is not None, "Scene should be initialized"
        return scene_obj

    def reset_view_tab(self):
        """Reset the view tab by clearing all additional layers"""
        for pixmap in self.view_pixmaps:
            self.get_scene().removeItem(pixmap)
        self.view_pixmaps = []

        # Remove and clear rubber bands from scene and tracking lists
        for rb in list(self.rubber_bands):
            scene = rb.scene()
            if scene:
                scene.removeItem(rb)
        self.rubber_bands = []
        self.rubber_band_colors = []

        # Remove and clear polygons from scene and tracking lists
        for poly in list(self.polygons):
            scene = poly.scene()
            if scene:
                scene.removeItem(poly)
        self.polygons = []
        self.polygon_colors = []

        if self.current_polygon is not None:
            scene = self.current_polygon.scene()
            if scene:
                scene.removeItem(self.current_polygon)
            self.current_polygon = None

        self.select = False
        self.origin = None
        self._roi_origin_scene = None

        if self.enc.stacked_widget and self.enc.stacked_widget.currentIndex() == 0:
            self.show_images_tab_image()
        else:
            self.show_view_tab_image()
        self._center_image()

    def flip_horizontal(self):
        """Flip the image horizontally"""
        if self.pixmap_item:
            # img = self.pixmap_item.pixmap().toImage()
            # flipped_img = img.mirrored(horizontal=True, vertical=False)
            # flipped_pixmap = QPixmap.fromImage(flipped_img)
            # self.pixmap_item.setPixmap(flipped_pixmap)
            # self.get_scene().update()
            # Emit signal to update the underlying data model
            self.horizontal_flip.emit()

    def flip_vertical(self):
        """Flip the image vertically"""
        if self.pixmap_item:
            # img = self.pixmap_item.pixmap().toImage()
            # flipped_img = img.mirrored(horizontal=False, vertical=True)
            # flipped_pixmap = QPixmap.fromImage(flipped_img)
            # self.pixmap_item.setPixmap(flipped_pixmap)
            # self.get_scene().update()
            # Emit signal to update the underlying data model
            self.vertical_flip.emit()

    def setup_ui(self):
        """Setup the UI."""
        self.setMinimumSize(QSize(600, 600))
        self.setObjectName("canvas")
        self.setAcceptDrops(True)
        self.setScene(pg.GraphicsScene())
        self.get_scene().setBackgroundBrush(QBrush(QColor("black")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSceneRect(0, 0, 800, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # self.setContentsMargins(1000, 1000, 1000, 1000)

        # Create floating selection buttons
        if self.show_buttons:
            self.create_floating_buttons()
            self.set_buttons_visible(False)

    def create_floating_buttons(self):
        """Create floating selection buttons that appear over the canvas"""
        from ui.theme import ThemeManager

        tc = ThemeManager.instance().get_current()

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
            f"QLabel {{ color: {tc['text_primary']}; padding: 5px; border-radius: 3px; }}"
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
                f"""
                QPushButton {{
                    background-color: {tc["bg_hover"]};
                    border: 1px solid {tc["border"]};
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {tc["bg_pressed"]};
                }}
                QPushButton:pressed {{
                    background-color: {tc["accent_dim"]};
                }}
            """
            )

        # Set icons for the buttons
        self.rect_button.setIcon(QIcon(resource_path("assets/icons/square.png")))
        self.circle_button.setIcon(QIcon(resource_path("assets/icons/circle.png")))
        self.poly_button.setIcon(QIcon(resource_path("assets/icons/poly.png")))

        self.rect_button.clicked.connect(lambda: self.set_selection_mode("rect"))
        self.circle_button.clicked.connect(lambda: self.set_selection_mode("circle"))
        self.poly_button.clicked.connect(self.enc.poly_select)

        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)

        # Position the container at the top-right of the view
        self.update_buttons_position()

    def set_buttons_visible(self, visible: bool):
        """Set the visibility of the floating buttons"""
        if self.floating_container:
            self.floating_container.setVisible(visible)
            self.update_buttons_position()

    def update_buttons_position(self):
        """Update the position of the floating buttons"""
        if self.floating_container and self.floating_container.isVisible():
            # Position at the top-right of the view with some padding
            self.floating_container.move(
                self.width() - self.floating_container.width() - 20,
                10,
            )

    # pylint: disable=invalid-name, useless-parent-delegation
    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_buttons_position()

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
        self.select = False
        self.current_polygon = None
        self.setFocus()

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
        self._roi_origin_scene = None

        if self.active_crop_rect:
            self.get_scene().removeItem(self.active_crop_rect)
            self.active_crop_rect = None

        self.unsetCursor()

    def cancel_roi_operation(self):
        """Cancel any in-progress ROI drawing without submitting it."""
        if self.select == "poly" and self.current_polygon:
            self.get_scene().removeItem(self.current_polygon)
            self.current_polygon = None
            if self.polygon_colors:
                self.polygon_colors.pop()
            self.select = False
        elif self.select in ("rect", "circle") and self.origin is not None:
            # Mid-draw: remove the in-progress rubber band
            if self.rubber_bands:
                rb = self.rubber_bands.pop()
                if self.rubber_band_colors:
                    self.rubber_band_colors.pop()
                rb.remove_from_scene()
            self.rubber_band = None
            self.origin = None
            self._roi_origin_scene = None
            self.select = False
        elif self.select:
            # In selection mode but no draw started yet: just exit the mode
            self.select = False

    def is_empty(self) -> bool:
        """Check if canvas is empty."""
        has_pixmap = (
            self.pixmap_item is not None and not self.pixmap_item.pixmap().isNull()
        )
        has_view_layers = len(self.view_pixmaps) > 0
        return not (has_pixmap or has_view_layers)

    # pylint: disable=invalid-name, unused-argument
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click."""
        if not self.is_empty() and self.select != "poly":
            self._center_image()

    def update_canvas(self, pixmap: QPixmap):
        """Updates canvas when current image is operated on"""
        if self.pixmap_item:
            prev_pixmap_shape = self.pixmap_item.boundingRect()
            self.pixmap_item.setPixmap(pixmap)
            if self.zoom == 1 or self.pixmap_item.boundingRect() != prev_pixmap_shape:
                self._center_image()

    def update_layer_levels(self, layer_idx, value):
        """Update layer levels."""
        self.view_pixmaps[layer_idx].setLevels(value, True)

    def update_layer_cmap(self, layer_idx, cmap):
        """Update layer colormap."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setLookupTable(cmap)

    def update_layer_opacity(self, layer_idx, opacity):
        """Update layer opacity."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setOpacity(opacity)

    def update_layer_visibility(self, layer_idx, visible):
        """Update layer visibility."""
        layer = self.view_pixmaps[layer_idx]
        assert isinstance(layer, pg.ImageItem)
        layer.setVisible(visible)

    def update_view_tab_canvas(self, pixmap: np.ndarray, layer_idx):
        """Update view tab canvas."""
        # self._show_view_tab_image()
        if layer_idx > (len(self.view_pixmaps) - 1):
            new_layer = pg.ImageItem(pixmap, levels=None)
            new_layer.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            new_layer.setZValue(2)
            self.view_pixmaps.append(new_layer)
            self.get_scene().addItem(new_layer)
            self._center_view_tab_image(layer_idx)
            # Button visibility is now controlled by tab changes
        else:
            logger.debug(f"layer {layer_idx}")
            logger.debug(f"pixmap shape {pixmap.shape}")
            # Button visibility is now controlled by tab changes
            if pixmap.shape == (0,):
                removed_item = self.view_pixmaps.pop(layer_idx)
                self.get_scene().removeItem(removed_item)
            else:
                self.view_pixmaps[layer_idx].setImage(pixmap)

    def _center_view_tab_image(self, layer_idx):
        pixmap_item = self.view_pixmaps[layer_idx]
        item_rect = pixmap_item.boundingRect()
        item_rect = QRectF(
            item_rect.x() - item_rect.width() // 2,
            item_rect.y() - item_rect.height() // 2,
            item_rect.width() * 2,
            item_rect.height() * 2,
        )
        self.setSceneRect(item_rect)
        self.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmap_item)

    def _center_image(self):
        if self.pixmap_item.isVisible():
            pixmap_item = self.pixmap_item
        elif self.view_pixmaps:
            pixmap_item = self.view_pixmaps[0]
        elif self.pixmap_item is not None and not self.pixmap_item.pixmap().isNull():
            pixmap_item = self.pixmap_item
        else:
            return

        item_rect = pixmap_item.boundingRect()
        item_rect = QRectF(
            item_rect.x() - item_rect.width() // 2,
            item_rect.y() - item_rect.height() // 2,
            item_rect.width() * 2,
            item_rect.height() * 2,
        )
        self.setSceneRect(item_rect)
        self.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(pixmap_item)
        self.zoom = 1  # Reset zoom level when centering
        # if self.reference_view:
        #     self.reference_view.__centerImage()

    # pylint: disable=invalid-name
    def dragEnterEvent(self, event: QDragEnterEvent):  # type: ignore
        """Handle drag enter."""
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

    # pylint: disable=invalid-name
    def dragMoveEvent(self, event: QDragMoveEvent):  # type: ignore
        """Handle drag move."""
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    # pylint: disable=invalid-name
    def dropEvent(self, event):
        """Handle drop."""
        if event is None:
            return
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

    # pylint: disable=invalid-name
    def wheelEvent(self, event):
        """Handle wheel event for zooming."""
        if event is None:
            return
        if self.pixmap_item is None:
            return
        zooming_out = event.angleDelta().y() < 0

        # Prevent excessive zooming in either direction
        # if self.zoom > 1.1**2 and zooming_out:  # Max zoom out
        #     return

        # if self.zoom < 1 / (1.1**20) and not zooming_out:
        #     return
        # if self.reference_view:
        #     self.reference_view.wheelEvent(event)
        if self.zoom <= 0.1 and zooming_out:
            return
        zoom_factor = 0.9 if zooming_out else 1.1
        self.zoom *= zoom_factor

        # Perform the zoom (ROIs are scene items and auto-follow the transform)
        self.scale(zoom_factor, zoom_factor)

        # Force redraw to update polygon positions
        vp = self.viewport()
        assert vp is not None, "Viewport should be initialized"
        vp.update()

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def create_rubber_band(self, rubber_band_class, shape, x, y, origin):
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

    def _get_existing_roi_colors(self):
        """Return all current ROI colors as (r, g, b) tuples for distinctness checks."""
        return [c.getRgb()[:3] for c in self.rubber_band_colors] + self.polygon_colors

    def _find_top_rubber_band(self, view_pos):
        """Return the top-most visible ROI under the given view position."""
        scene_pos = self.mapToScene(view_pos)
        # Check completed PolyLasso items first (they're scene items, not in rubber_bands)
        for item in self.get_scene().items(scene_pos):
            if isinstance(item, PolyLasso) and item.completed:
                return item
        for rubber_band in reversed(self.rubber_bands):
            if rubber_band and rubber_band.isVisible():
                local_pos = rubber_band.mapFromScene(scene_pos)
                if rubber_band.shape().contains(local_pos):
                    return rubber_band
        return None

    def _get_poly_snap_target(self, view_pos, threshold=15):
        """Return scene pos of the nearest existing polygon point within threshold screen pixels, else None."""
        if not self.current_polygon or not self.current_polygon.points:
            return None
        vx, vy = view_pos.x(), view_pos.y()
        for point in self.current_polygon.points:
            vp = self.mapFromScene(point)
            dx, dy = vx - vp.x(), vy - vp.y()
            if (dx * dx + dy * dy) ** 0.5 <= threshold:
                return point
        return None

    def _finalize_poly(self):
        """Complete and submit the current polygon as an ROI."""
        if not self.current_polygon:
            return
        if self.current_polygon.complete():
            self.current_polygon.set_snap_point(None)
            self.polygons.append(self.current_polygon)
            self.enc.analysis_tab.analyze_poly_region(
                self.current_polygon, ("poly", self.current_polygon.im_points)
            )
            self.current_polygon = None
            self.select = False

    def _cancel_rubber_band_drag(self):
        """Reset any in-progress ROI drag state."""
        self.moving_lasso = None
        self.last_mouse_pos = None
        for rubber_band in self.rubber_bands:
            if hasattr(rubber_band, "is_dragging"):
                rubber_band.is_dragging = False
            if hasattr(rubber_band, "mouse_press_pos"):
                rubber_band.mouse_press_pos = None
            if hasattr(rubber_band, "mouse_move_pos"):
                rubber_band.mouse_move_pos = None

    # pylint: disable=invalid-name, too-many-branches, too-many-statements
    def mousePressEvent(self, event: QMouseEvent | None):
        """Handle mouse press."""
        if self.is_empty() or event is None:
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

                self.active_crop_rect = CropRectItem()
                self.active_crop_rect.setRect(scene_pos.x(), scene_pos.y(), 0, 0)

                # Style the crop rectangle
                pen = QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine)
                self.active_crop_rect.setPen(pen)
                self.get_scene().addItem(self.active_crop_rect)
                self.is_resizing = True

            # Handle Shift+Click to move ROI
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Find the top-most lasso under the mouse
                top_lasso = self._find_top_rubber_band(event.pos())
                if top_lasso is not None:
                    self.moving_lasso = top_lasso
                    self.last_mouse_pos = event.pos()
                    event.accept()
                    return

            # Handle regular selection modes
            if self.begin_crop or self.select:
                logger.debug("beginning selection")
                self.origin = event.pos()
                self.update_starting_position(event)
                scene_pos = self.mapToScene(event.pos())
                image_pos = self.pixmap_item.mapFromScene(scene_pos)
                self.select_start_pos = image_pos
                self._roi_origin_scene = scene_pos
                if self.begin_crop:
                    if not self.rubber_band:
                        self.rubber_band = RectLasso(
                            pos=(scene_pos.x(), scene_pos.y()),
                            size=(0, 0),
                        )
                        self.get_scene().addItem(self.rubber_band)
                elif self.select == "rect":
                    self.rubber_band = RectLasso(
                        pos=(scene_pos.x(), scene_pos.y()),
                        size=(0, 0),
                        existing_colors=self._get_existing_roi_colors(),
                    )
                    self.get_scene().addItem(self.rubber_band)
                    self.rubber_bands.append(self.rubber_band)
                    self.rubber_band_colors.append(self.rubber_band.color)
                    self.rubber_band.sigRegionChangeFinished.connect(
                        lambda: self.update_roi_from_lasso(self.rubber_band)
                    )
                elif self.select == "circle":
                    self.rubber_band = CircleLasso(
                        pos=(scene_pos.x(), scene_pos.y()),
                        size=(0, 0),
                        existing_colors=self._get_existing_roi_colors(),
                    )
                    self.get_scene().addItem(self.rubber_band)
                    self.rubber_bands.append(self.rubber_band)
                    self.rubber_band_colors.append(self.rubber_band.color)
                    self.rubber_band.sigRegionChangeFinished.connect(
                        lambda: self.update_roi_from_lasso(self.rubber_band)
                    )
                elif self.select == "poly":
                    if not self.current_polygon:
                        self.current_polygon = PolyLasso(
                            None,
                            existing_colors=self._get_existing_roi_colors(),
                        )
                        self.polygon_colors.append(
                            self.current_polygon.color.getRgb()[:3]
                        )
                        self.get_scene().addItem(self.current_polygon)
                        self.setMouseTracking(True)

                    snap_target = self._get_poly_snap_target(event.pos())
                    if snap_target is not None:
                        # Snapping to first point with enough vertices → auto-close
                        if (
                            snap_target is self.current_polygon.points[0]
                            and len(self.current_polygon.points) >= 3
                        ):
                            self._finalize_poly()
                        # Any other snap: skip adding duplicate point
                        return

                    polygon_pos = self.pixmap_item.mapToScene(image_pos)
                    self.current_polygon.add_point(polygon_pos, image_pos)

                if self.begin_crop:
                    self.rubber_bands.append(self.rubber_band)
                    assert self.rubber_band is not None, (
                        "Rubber band should be initialized"
                    )
                    self.rubber_band_colors.append(self.rubber_band.color)

        if not self.is_resizing and not self.select:
            super().mousePressEvent(event)

    # pylint: disable=invalid-name
    def keyPressEvent(self, event):
        """Handle key press."""
        if event is None:
            return
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            if self.crop_mode and self.active_crop_rect:
                logger.debug("confirming")
                self.confirm_crop()
                return
        if event.key() == Qt.Key.Key_Escape:
            if self.crop_mode:
                self.cancel_crop_mode()
                return
            if self.select:
                self.cancel_roi_operation()
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

    def save_as_png(self, num_layers):
        """Save as PNG."""
        if num_layers == 0:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save PNG File", "protein_layers.png", "*.png;;All Files (*)"
        )
        if not file_name:
            return

        scene = self.get_scene()
        scene_rect = scene.sceneRect()

        # Create a QImage to render the scene onto
        # Use ARGB32_Premultiplied for transparency support
        img = QImage(
            scene_rect.size().toSize(), QImage.Format.Format_ARGB32_Premultiplied
        )
        img.fill(Qt.GlobalColor.black)  # Fill with transparent background

        # Create a QPainter to draw on the QImage
        painter = QPainter(img)

        # Render the scene onto the QImage
        scene.render(painter, QRectF(img.rect()), scene_rect)

        # End the painter
        painter.end()
        img.save(file_name)

    def save_as_tif(self, num_layers):
        """Save as TIF."""
        if num_layers == 0:
            return  # the second QRectF is the source rect from the scene.
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save PNG File", "protein_layers.png", "*.png;;All Files (*)"
        )
        if not file_name:
            return

        assert num_layers == len(self.view_pixmaps)
        img_stack = []
        for i in self.view_pixmaps:
            assert isinstance(i, pg.ImageItem)
            qimage = (
                i.getPixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
            )
            ptr = qimage.bits()
            ptr.setsize(qimage.width() * qimage.height() * 4)
            # pylint: disable=too-many-function-args
            arr = np.array(ptr, dtype=np.uint8).reshape(
                qimage.height(), qimage.width(), 3
            )
            img_stack.append(arr)

        img_stack = np.array(img_stack)
        # pylint: disable=too-many-function-args
        tifffile.imwrite(file_name, img_stack, imagej=True)

    # pylint: disable=invalid-name, too-many-locals, too-many-statements, too-many-branches
    def mouseMoveEvent(self, event: QMouseEvent | None):
        """Handle mouse move."""
        super().mouseMoveEvent(event)

        # Handle crop rectangle re  sizing
        if event is None or self.is_empty():
            return
        event.accept()
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
            r = QRectF(left, top, width, height)
            r = QRectF(left, top, width, height)
            self.active_crop_rect.setRect(r)

        # Handle moving lasso
        if self.moving_lasso and self.last_mouse_pos:
            old_scene = self.mapToScene(self.last_mouse_pos)
            new_scene = self.mapToScene(event.pos())
            cur = self.moving_lasso.pos()
            new_x = cur.x() + new_scene.x() - old_scene.x()
            new_y = cur.y() + new_scene.y() - old_scene.y()
            if isinstance(self.moving_lasso, PolyLasso):
                # QGraphicsItem.setPos() has no finish parameter
                self.moving_lasso.setPos(new_x, new_y)
            else:
                self.moving_lasso.setPos(new_x, new_y, finish=False)
            self.last_mouse_pos = event.pos()
            event.accept()

        # Store current mouse position for polygon preview
        if self.current_polygon and len(self.current_polygon.points) > 0:
            scene_pos = self.mapToScene(event.pos())
            image_pos = self.pixmap_item.mapFromScene(scene_pos)
            snap_target = self._get_poly_snap_target(event.pos())
            if snap_target is not None:
                self.current_polygon.set_temp_point(snap_target)
                self.current_polygon.set_snap_point(snap_target)
                self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                polygon_pos = self.pixmap_item.mapToScene(image_pos)
                self.current_polygon.set_temp_point(polygon_pos)
                self.current_polygon.set_snap_point(None)
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)

        # # Handle pixel info display
        # Determine reference item for coordinates and bounds
        reference_item = self.pixmap_item
        has_main_pixmap = (
            self.pixmap_item is not None and not self.pixmap_item.pixmap().isNull()
        )

        # If main pixmap is invalid but we have view layers, use the first view layer as reference
        if not has_main_pixmap and self.view_pixmaps:
            reference_item = self.view_pixmaps[0]

        if reference_item:
            scene_pos = self.mapToScene(event.pos())

            # Map scene position to image coordinates relative to the reference item
            # QGraphicsItem.mapFromScene works for both QGraphicsPixmapItem and pg.ImageItem
            image_pos = reference_item.mapFromScene(scene_pos)

            x = int(image_pos.x())
            y = int(image_pos.y())

            # Determine dimensions of the reference item
            width = 0
            height = 0

            if (
                isinstance(reference_item, QGraphicsPixmapItem)
                and not reference_item.pixmap().isNull()
            ):
                pixmap = reference_item.pixmap()
                width = pixmap.width()
                height = pixmap.height()
            elif isinstance(reference_item, pg.ImageItem):
                # pg.ImageItem stores image data in .image
                if reference_item.image is not None:
                    # Shape is usually (h, w) or (w, h) depending on axis order,
                    # but ImageItem exposes width() and height() methods usually
                    width = reference_item.width()
                    height = reference_item.height()

            # Check bounds
            if width > 0 and height > 0 and 0 <= x < width and 0 <= y < height:
                global_pos = self.mapToGlobal(event.pos())

                # Get layer values if available
                if self.enc and self.enc.tool_bar.tabButtonGroup.checkedId() != 0:
                    layers = self.enc.view_tab.get_layer_values_at(x, y)
                else:
                    layers = None

                combined_layers = None

                if layers:
                    layers = [f"{layer}: {value}\n" for layer, value in layers]
                    combined_layers = "".join(layers)[:-1]
                    self._show_tooltip(global_pos, combined_layers)
                else:
                    # Fast path: read intensity directly from model data
                    raw_intensity_str = None
                    try:
                        from controller import Controller

                        ctrl = Controller.get()
                        if (
                            self == self.enc.canvas
                            and ctrl.model_canvas.image_wrapper is not None
                            and ctrl.model_canvas.image_wrapper.data.size > 0
                        ):
                            val = ctrl.model_canvas.image_wrapper.data[y, x]
                            if is_segmentation_channel(ctrl.model_canvas.image_wrapper):
                                raw_intensity_str = f"Cell ID: {val}"
                            else:
                                seg_labels = ctrl.model_canvas.segmentation_labels
                                if (
                                    ctrl.model_canvas.stardist_overlay_enabled
                                    and seg_labels is not None
                                    and 0 <= y < seg_labels.shape[0]
                                    and 0 <= x < seg_labels.shape[1]
                                ):
                                    cell_id = seg_labels[y, x]
                                    raw_intensity_str = (
                                        f"Cell ID: {cell_id}\nIntensity: {val}"
                                    )
                                else:
                                    raw_intensity_str = f"Intensity: {val}"
                        elif (
                            hasattr(self.enc, "small_view")
                            and self == self.enc.small_view
                            and ctrl.model_reference_canvas.image_wrapper is not None
                            and ctrl.model_reference_canvas.image_wrapper.data.size > 0
                        ):
                            val = ctrl.model_reference_canvas.image_wrapper.data[y, x]
                            raw_intensity_str = f"Intensity: {val}"
                    except Exception:
                        pass

                    # Slow path: render scene pixel (only when model data is unavailable)
                    if raw_intensity_str is None:
                        img = QImage(1, 1, QImage.Format.Format_ARGB32_Premultiplied)
                        img.fill(Qt.GlobalColor.black)
                        painter = QPainter(img)
                        scene = self.get_scene()
                        scene.render(
                            painter,
                            QRectF(0, 0, 1, 1),
                            QRectF(scene_pos.x(), scene_pos.y(), 1, 1),
                        )
                        painter.end()
                        color = QColor(img.pixel(0, 0))
                        r, g, b = color.red(), color.green(), color.blue()
                        raw_intensity_str = f"R: {r}, G: {g}, B: {b}"

                    self._show_tooltip(global_pos, raw_intensity_str)

                # Update position display in main window
                if combined_layers:
                    combined_layers = combined_layers.replace("\n", ", ")
                    combined_layers += ";"
                    self.enc.update_mouse_position_label(
                        f"{combined_layers} X: {x}, Y: {y}"
                    )
                else:
                    self.enc.update_mouse_position_label(
                        f"{raw_intensity_str} X: {x}, Y: {y}"
                    )

                # Highlight the pixel under cursor
                self.highlight_pixel(x, y)
                # if self.reference_view:
                #     self.reference_view.highlight_pixel(x, y)
            else:
                self.enc.update_mouse_position_label("")
                self._hide_tooltip()
                # Hide pixel highlight when outside image bounds
                self.hide_pixel_highlight()
                # if self.reference_view:
                # self.reference_view.hide_pixel_highlight()
        # Handle rubber band updates for old crop system
        if (
            not self.is_empty()
            and self.begin_crop
            and self.rubber_band
            and not self.crop_mode
        ):
            if self._roi_origin_scene is None:
                self.origin = event.pos()
                self.update_starting_position(event)
                sp = self.mapToScene(event.pos())
                self._roi_origin_scene = sp
            cur = self.mapToScene(event.pos())
            origin = self._roi_origin_scene
            w = abs(cur.x() - origin.x())
            h = abs(cur.y() - origin.y())
            self.rubber_band.setPos(
                min(cur.x(), origin.x()), min(cur.y(), origin.y()), finish=False
            )
            self.rubber_band.setSize((w, h), update=True, finish=False)

        if (
            self.select in ("rect", "circle")
            and self.rubber_bands
            and self._roi_origin_scene is not None
        ):
            cur = self.mapToScene(event.pos())
            origin = self._roi_origin_scene
            if self.select == "circle":
                half = max(abs(cur.x() - origin.x()), abs(cur.y() - origin.y()))
                self.rubber_bands[-1].setPos(
                    origin.x() - half, origin.y() - half, finish=False
                )
                self.rubber_bands[-1].setSize(
                    (half * 2, half * 2), update=True, finish=False
                )
            else:
                w = abs(cur.x() - origin.x())
                h = abs(cur.y() - origin.y())
                self.rubber_bands[-1].setPos(
                    min(cur.x(), origin.x()), min(cur.y(), origin.y()), finish=False
                )
                self.rubber_bands[-1].setSize((w, h), update=True, finish=False)

    # pylint: disable=invalid-name
    def mouseReleaseEvent(self, event: QMouseEvent | None):
        """Handle mouse release."""
        if event is None or self.is_empty():
            return

        super().mouseReleaseEvent(event)

        # Handle moving lasso release
        if self.moving_lasso:
            self.update_roi_from_lasso(self.moving_lasso)
            self.moving_lasso = None
            self.last_mouse_pos = None
            return

        self.rubber_band_positions = []
        # Handle crop mode mouse release
        if self.crop_mode and self.active_crop_rect and self.is_resizing:
            # Don't auto-confirm, let user press Enter or Escape
            self.initial_crop = True
            self.initial_crop_rect = self.active_crop_rect.rect()
            self.get_scene().removeItem(self.active_crop_rect)

            # pylint: disable=too-many-function-args
            self.active_crop_rect = ResizableRect(
                self.initial_crop_rect.x(),
                self.initial_crop_rect.y(),
                self.initial_crop_rect.width(),
                self.initial_crop_rect.height(),
            )
            self.active_crop_rect.setZValue(10)

            self.is_resizing = False

            self.get_scene().addItem(self.active_crop_rect)
            return
        if not self.rubber_bands:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            rubberband = self.rubber_band if self.begin_crop else self.rubber_bands[-1]

            if self.select:
                self.origin = None
                self._roi_origin_scene = None
                assert self.select_start_pos is not None
                if self.select in ("rect", "circle"):
                    assert rubberband is not None
                    scene_pos = self.mapToScene(event.pos())
                    image_pos = self.pixmap_item.mapFromScene(scene_pos)
                    # if select is circle, then image_pos dist to initial left click pos is radius
                    # if select is rectangle, then initial is top let and image_pos is bottom right
                    # initial pos is self.select_start_pos
                    logger.debug(image_pos)
                    image_rect = (
                        self.select,
                        (
                            self.select_start_pos.x(),
                            self.select_start_pos.y(),
                            image_pos.x(),
                            image_pos.y(),
                        ),
                    )
                    # if failed to analyze region, then remove rubber band
                    if not self.enc.analysis_tab.analyze_region(rubberband, image_rect):
                        self.rubber_bands.remove(self.rubber_band)
                        self.rubber_band_colors.pop()
                        if rubberband:
                            rubberband.remove_from_scene()
                    self.select = False
                    self._roi_origin_scene = None
                    return

    def _ensure_tooltip_label(self):
        if self._tooltip_label is not None:
            return self._tooltip_label

        viewport = self.viewport()
        tooltip = QLabel(viewport)
        tooltip.setObjectName("canvasTooltip")
        tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        tooltip.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        tooltip.setMargin(TOOLTIP_BOX_PADDING_PX)
        tooltip.setIndent(0)
        tooltip.setWordWrap(False)
        tooltip.hide()

        palette = viewport.palette()
        bg = palette.color(QPalette.ColorRole.ToolTipBase).name()
        fg = palette.color(QPalette.ColorRole.ToolTipText).name()
        border = palette.color(QPalette.ColorRole.Mid).name()
        tooltip.setStyleSheet(
            "QLabel#canvasTooltip {"
            f"background-color: {bg};"
            f"color: {fg};"
            f"border: 1px solid {border};"
            f"padding: {TOOLTIP_BOX_PADDING_PX}px;"
            "}"
        )
        self._tooltip_label = tooltip
        return tooltip

    def _show_tooltip(self, pos, text):
        tooltip = self._ensure_tooltip_label()
        local_pos = self.viewport().mapFromGlobal(pos) + TOOLTIP_CURSOR_OFFSET

        if text == self._last_tooltip_text and local_pos == self._last_tooltip_pos:
            return

        self._last_tooltip_text = text
        self._last_tooltip_pos = local_pos

        tooltip.setText(text)
        tooltip.adjustSize()

        max_x = max(0, self.viewport().width() - tooltip.width())
        max_y = max(0, self.viewport().height() - tooltip.height())
        tooltip.move(
            max(0, min(local_pos.x(), max_x)),
            max(0, min(local_pos.y(), max_y)),
        )
        tooltip.raise_()
        tooltip.show()

    def _hide_tooltip(self):
        self._last_tooltip_text = None
        self._last_tooltip_pos = None
        if self._tooltip_label is not None:
            self._tooltip_label.hide()

    def leaveEvent(self, event):
        """Cancel ROI dragging when pointer leaves the view."""
        self._cancel_rubber_band_drag()
        self._hide_tooltip()
        super().leaveEvent(event)

    # pylint: disable=invalid-name
    def scrollContentsBy(self, dx, dy):
        """Scroll the view; ROIs are scene items and auto-follow the transform."""
        super().scrollContentsBy(dx, dy)

    def load_channels(self, np_channels):
        """Load channel data"""
        self.np_channels = np_channels
        if self.pixmap_item is not None:
            self._center_image()

    def highlight_pixel(self, x, y):
        """Highlight the pixel at the given coordinates"""
        if self.pixmap_item is None:
            return

        # Create a small rectangle to highlight the pixel
        # Convert pixel coordinates to scene coordinates
        pixel_rect = QRectF(x, y, 1, 1)  # 1x1 pixel
        scene_rect = self.pixmap_item.mapRectToScene(pixel_rect)

        # Reuse existing item if possible
        if hasattr(self, "pixel_highlight") and self.pixel_highlight:
            # Ensure it's in the scene
            if self.pixel_highlight.scene() != self.get_scene():
                self.get_scene().addItem(self.pixel_highlight)

            self.pixel_highlight.setRect(scene_rect)
            self.pixel_highlight.show()
        else:
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
            self.pixel_highlight.hide()

    def _roi_scene_to_image_rect(self, roi):
        """Convert a pg.ROI's scene-space pos/size to an (x1,y1,x2,y2) image-coord tuple."""
        pos = roi.pos()
        size = roi.size()
        tl = self.pixmap_item.mapFromScene(QPointF(pos.x(), pos.y()))
        br = self.pixmap_item.mapFromScene(
            QPointF(pos.x() + size.x(), pos.y() + size.y())
        )
        return (tl.x(), tl.y(), br.x(), br.y())

    def update_roi_from_lasso(self, rubberband):
        """Update ROI region based on lasso position."""
        if not rubberband or not self.pixmap_item:
            return

        if isinstance(rubberband, PolyLasso):
            new_im_points = []
            for pt in rubberband.points:
                scene_pt = rubberband.mapToScene(pt)
                img_pt = self.pixmap_item.mapFromScene(scene_pt)
                new_im_points.append(img_pt)
            rubberband.im_points = new_im_points
            self.enc.analysis_tab.update_roi_region(rubberband, ("poly", new_im_points))
        elif isinstance(rubberband, CircleLasso):
            pos = rubberband.pos()
            size = rubberband.size()
            cx = pos.x() + size.x() / 2
            cy = pos.y() + size.y() / 2
            ex = pos.x() + size.x()
            ey = pos.y() + size.y() / 2
            center = self.pixmap_item.mapFromScene(QPointF(cx, cy))
            edge = self.pixmap_item.mapFromScene(QPointF(ex, ey))
            region = ("circle", (center.x(), center.y(), edge.x(), edge.y()))
            self.enc.analysis_tab.update_roi_region(rubberband, region)
        else:
            region = ("rect", self._roi_scene_to_image_rect(rubberband))
            self.enc.analysis_tab.update_roi_region(rubberband, region)
