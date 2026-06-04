import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QCoreApplication
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QIntValidator,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from utils import adjust_contrast, to_uint8


# ---------------------------------------------------------------------------
# Shared helpers (used by both AlignmentViewDialog and AlignmentPreviewDialog)
# ---------------------------------------------------------------------------

class NullableIntValidator(QIntValidator):
    def validate(self, input_str, pos):
        if input_str == "":
            return (self.State.Acceptable, input_str, pos)
        return super().validate(input_str, pos)


def readable_matrix_string(matrix: np.ndarray) -> str:
    if matrix.shape != (2, 3):
        return str(matrix)
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    angle_rad = math.atan2(c, a)
    angle_deg = math.degrees(angle_rad)
    scale_x = math.sqrt(a**2 + c**2)
    scale_y = math.sqrt(b**2 + d**2)
    return f"Translation: ({tx:.2f}, {ty:.2f}), Rotation: {angle_deg:.2f}°, Scale: (x: {scale_x:.2f}, y: {scale_y:.2f})"


def colorize_grayscale(gray_img: np.ndarray, color: str) -> QPixmap:
    """Colorize grayscale image and make black pixels fully transparent."""
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if color == "red":
        rgba[:, :, 0] = gray_img
    elif color == "green":
        rgba[:, :, 1] = gray_img
    elif color == "blue":
        rgba[:, :, 2] = gray_img
    mask = gray_img > 0
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    qimage = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


def transform_to_matrix(t: QTransform) -> np.ndarray:
    return np.array(
        [
            [t.m11(), t.m21(), t.dx()],
            [t.m12(), t.m22(), t.dy()],
        ],
        dtype=np.float32,
    )


def _norm_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize any dtype to uint8 for display."""
    if img.dtype == np.uint8:
        return img
    mn, mx = img.min(), img.max()
    if mx > mn:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return np.zeros_like(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# ZoomableImageView
# ---------------------------------------------------------------------------

class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.target_item = QGraphicsPixmapItem()
        self.moving_item = QGraphicsPixmapItem()
        self.moving_item.setZValue(0.5)
        self.moving_item.setOpacity(0.5)
        self._scene.addItem(self.target_item)
        self._scene.addItem(self.moving_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self._is_dragging_layer = False
        self._drag_start_pos = None
        self._parent_dialog = None

        self._max_zoom = 150.0
        self._min_zoom = 0.1
        self._current_zoom = 1.0

    def set_images(self, target_pixmap: QPixmap, moving_pixmap: QPixmap):
        self.target_item.setPixmap(target_pixmap)
        self.moving_item.setPixmap(moving_pixmap)
        QTimer.singleShot(0, self.reset_zoom)

    def reset_zoom(self):
        self.get_scene().setSceneRect(self.get_scene().itemsBoundingRect())
        self.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.target_item)
        self._current_zoom = 1.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.target_item and not self.target_item.pixmap().isNull():
            self.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)
            if self._current_zoom != 1.0:
                self.scale(self._current_zoom, self._current_zoom)

    def get_scene(self):
        s = self.scene()
        assert s is not None
        return s

    def update_moving_image(self, new_pixmap: QPixmap):
        self.moving_item.setPixmap(new_pixmap)

    def wheelEvent(self, event):
        if event is None:
            return
        angle = event.angleDelta().y()
        zoom_factor = 1.15 if angle > 0 else 1 / 1.15
        new_zoom = self._current_zoom * zoom_factor
        if new_zoom > self._max_zoom:
            zoom_factor = self._max_zoom / self._current_zoom
            new_zoom = self._max_zoom
        if new_zoom < 0.1:
            zoom_factor = 0.1 / self._current_zoom
            new_zoom = 0.1
        self._current_zoom = new_zoom
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        ):
            if self._parent_dialog and self._parent_dialog.can_edit:
                self._is_dragging_layer = True
                self._drag_start_pos = self.mapToScene(event.pos())
                event.accept()
                return
        if (
            self._parent_dialog
            and getattr(self._parent_dialog, "_lm_mode", False)
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            scene_pt = self.mapToScene(event.pos())
            self._parent_dialog._on_landmark_click(scene_pt.x(), scene_pt.y())
            event.accept()
            return
        if (
            self._parent_dialog
            and getattr(self._parent_dialog, "_lm_mode", False)
            and event.button() == Qt.MouseButton.RightButton
        ):
            scene_pt = self.mapToScene(event.pos())
            if hasattr(self._parent_dialog, "_on_landmark_right_click"):
                self._parent_dialog._on_landmark_right_click(scene_pt.x(), scene_pt.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging_layer and self._drag_start_pos is not None:
            current_pos = self.mapToScene(event.pos())
            delta = current_pos - self._drag_start_pos
            transform = self.moving_item.transform()
            new_transform = QTransform(
                transform.m11(),
                transform.m12(),
                transform.m21(),
                transform.m22(),
                transform.dx() + delta.x(),
                transform.dy() + delta.y(),
            )
            self.moving_item.setTransform(new_transform)
            if self._parent_dialog:
                self._parent_dialog.update_offset_label()
            self._drag_start_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging_layer:
            self._is_dragging_layer = False
            self._drag_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# AlignmentViewDialog — read-only base, used for arrays-preview case
# ---------------------------------------------------------------------------

class AlignmentViewDialog(QDialog):
    """
    Read-only alignment viewer. Base class for AlignmentPreviewDialog.

    Subclass extension points (template methods):
      _init_state()                  — init extra instance state before UI build
      _add_header_widgets(layout)    — insert widgets after preview_label
      _add_editing_controls(layout)  — insert widgets after image_view
      _setup_buttons()               — populate self.button_layout
      reset_zoom(event)              — override to also reset moving_item.transform
      _on_confirm()                  — override to compute/emit transforms
      _update_preview_label()        — override to change label text/style
    """

    can_edit: bool = False  # read by ZoomableImageView shift+drag guard
    moving_image_changed = pyqtSignal(np.ndarray)

    def __init__(self, snapshot_data: dict, progress_callback=None):
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.target_image = snapshot_data["target_image"]
        self.aligned_image = snapshot_data["aligned_image"].copy()
        self._snapshot_data = snapshot_data
        self.metadata = snapshot_data.get("metadata", {})
        self.adjust_contrast = True
        self._progress_callback = progress_callback
        if self._progress_callback:
            self._progress_callback(10, "Initializing state")
        self._init_state()
        self._setup_ui()
        self._refresh_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom
        self._snapshot_data = None
        if self._progress_callback:
            self._progress_callback(100, "Done")
        self._progress_callback = None
        self.finished.connect(self._cleanup)

    # ------------------------------------------------------------------
    # Template hooks
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        pass

    def _add_header_widgets(self, layout: QVBoxLayout) -> None:
        pass

    def _add_editing_controls(self, layout: QVBoxLayout) -> None:
        pass

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        stage = self.metadata.get("stage", "Preview").replace("_", " ").title()
        self.setWindowTitle(f"Alignment Preview: {stage}")
        self.resize(1000, 800)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        main_layout = QVBoxLayout(self)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_preview_label()
        main_layout.addWidget(self.preview_label)

        self._add_header_widgets(main_layout)

        controls_row = QHBoxLayout()
        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(True)
        self.enhance_contrast_checkbox.stateChanged.connect(self._on_contrast_changed)
        controls_row.addWidget(self.enhance_contrast_checkbox)
        self.invert_fixed_checkbox = QCheckBox("Invert Fixed")
        self.invert_fixed_checkbox.setToolTip(
            "Invert target (red) image — useful for white-background RGB images"
        )
        self.invert_fixed_checkbox.stateChanged.connect(self._on_invert_changed)
        controls_row.addWidget(self.invert_fixed_checkbox)
        self.invert_moving_checkbox = QCheckBox("Invert Moving")
        self.invert_moving_checkbox.setToolTip(
            "Invert aligned (green) image — useful for white-background RGB images"
        )
        self.invert_moving_checkbox.stateChanged.connect(self._on_invert_changed)
        controls_row.addWidget(self.invert_moving_checkbox)
        controls_row.addStretch()
        main_layout.addLayout(controls_row)

        # Opacity slider — setOpacity on QGraphicsPixmapItem, no pixmap rebuild
        opacity_row = QHBoxLayout()
        opacity_row.addStretch()
        opacity_row.addWidget(QLabel("Overlay Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setFixedWidth(160)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        main_layout.addLayout(opacity_row)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)
        self.image_view._parent_dialog = self
        main_layout.addWidget(self.image_view)

        self._add_editing_controls(main_layout)

        self.button_layout = QHBoxLayout()
        self._setup_buttons()
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)

    def _setup_buttons(self) -> None:
        self.confirm_button = QPushButton("Confirm Alignment")
        self.cancel_button = QPushButton("Cancel")
        self.confirm_button.clicked.connect(self._on_confirm)
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addStretch()

    # ------------------------------------------------------------------
    # Slots / overridable actions
    # ------------------------------------------------------------------

    def _on_opacity_changed(self, value: int) -> None:
        self.image_view.moving_item.setOpacity(value / 100.0)

    def _on_contrast_changed(self, _state) -> None:
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self._refresh_overlay()

    def _on_invert_changed(self, _state) -> None:
        self._refresh_overlay()

    def _update_preview_label(self) -> None:
        self.preview_label.setText(
            "Red = Target, Green = Aligned | "
            "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        self.preview_label.setStyleSheet("")

    def _refresh_overlay(self) -> None:
        target_img = self.target_image
        if self._progress_callback:
            self._progress_callback(30, "Converting images to grayscale")
        if target_img.ndim == 3:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        aligned_img = self.aligned_image
        if aligned_img.ndim == 3:
            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)

        if self._progress_callback:
            self._progress_callback(45, "Normalizing image intensities")
        target_gray = _norm_to_uint8(target_img)
        aligned_gray = _norm_to_uint8(aligned_img)

        if self.invert_fixed_checkbox.isChecked():
            target_gray = 255 - target_gray
        if self.invert_moving_checkbox.isChecked():
            aligned_gray = 255 - aligned_gray

        if self.adjust_contrast:
            if self._progress_callback:
                self._progress_callback(60, "Enhancing image contrast")
            target_gray = to_uint8(adjust_contrast(target_gray.astype(np.float32), 30, 99))
            aligned_gray = to_uint8(adjust_contrast(aligned_gray.astype(np.float32), 30, 99))

        if self._progress_callback:
            self._progress_callback(75, "Colorizing target and aligned images")

        red_pixmap = colorize_grayscale(target_gray, "red")
        green_pixmap = colorize_grayscale(aligned_gray, "green")

        if self._progress_callback:
            self._progress_callback(90, "Updating overlay view")
        self.image_view.set_images(red_pixmap, green_pixmap)

        # Restore opacity after pixmap swap (set_images does not touch opacity)
        self.image_view.moving_item.setOpacity(self._opacity_slider.value() / 100.0)

    def reset_zoom(self, event=None) -> None:
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _cleanup(self) -> None:
        """Break reference cycles and release large arrays when the dialog finishes."""
        self.image_view._parent_dialog = None
        try:
            del self.image_view.mouseDoubleClickEvent
        except AttributeError:
            pass
        self.target_image = None
        self.aligned_image = None

    def _on_confirm(self) -> None:
        self.moving_image_changed.emit(self.aligned_image)
        self.accept()
