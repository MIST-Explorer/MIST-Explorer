import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QDoubleValidator,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPainter,
    QPixmap,
    QTransform
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from utils import adjust_contrast, to_uint8

class NullableIntValidator(QIntValidator):
    def validate(self, input_str, pos):
        if input_str == "":
            return (self.State.Acceptable, input_str, pos)
        return super().validate(input_str, pos)

class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.target_item = QGraphicsPixmapItem()
        self.moving_item = QGraphicsPixmapItem()
        self.moving_item.setZValue(1)
        self.moving_item.setOpacity(0.5)
        self._scene.addItem(self.target_item)
        self._scene.addItem(self.moving_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def set_images(self, target_pixmap: QPixmap, moving_pixmap: QPixmap):
        self.target_item.setPixmap(target_pixmap)
        self.moving_item.setPixmap(moving_pixmap)
        QTimer.singleShot(0, self.reset_zoom) # center after render updates


    def reset_zoom(self):
        self.get_scene().setSceneRect(self.get_scene().itemsBoundingRect())
        self.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.target_item)
    def get_scene(self):
        s = self.scene()
        assert s is not None
        return s

    def update_moving_image(self, new_pixmap: QPixmap):
        self.moving_item.setPixmap(new_pixmap)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event is None:
            return
        angle = event.angleDelta().y()
        if angle > 0:
            zoom_factor = 1.15  # Zoom in
        else:
            zoom_factor = 1 / 1.15  # Zoom out

        self.scale(zoom_factor, zoom_factor)

class AlignmentPreviewDialog(QDialog):
    moving_image_changed = pyqtSignal(np.ndarray)

    def __init__(self, snapshot_data: dict, can_edit: bool = False, can_emit=False):
        super().__init__(None)

        self.target_image = snapshot_data["target_image"].copy()
        self.aligned_image = snapshot_data["aligned_image"].copy()
        print(self.target_image.shape, self.aligned_image.shape)
        self.metadata = snapshot_data.get("metadata", {})
        self.can_edit = can_edit
        self.original_aligned_image = self.aligned_image.copy()
        self.result_accepted = False
        self.transformations = [[0.0, []]]
        self.offset_x, self.offset_y, self.move_step = 0, 0, 1
        self.adjust_contrast = True
        self.can_emit = can_emit
        self.downscaled = False
        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

    def _setup_ui(self):
        stage_name = self.metadata.get("stage", "Preview").replace("_", " ").title()
        self.setWindowTitle(f"Alignment Preview: {stage_name}")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)
        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )
        instruction_text = (
            "Arrow keys/Inputs: move, Mouse wheel: zoom, Drag: pan, Double-click: reset view"
            if self.can_edit
            else "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        self.preview_label = QLabel(
            f"Red = Target, Green = Aligned | {instruction_text}"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label.setVisible(self.can_edit)

        self.metadata_groupbox = QGroupBox("Stage Information")
        metadata_layout = QFormLayout()
        for key, value in self.metadata.items():
            key_str = key.replace("_", " ").title()
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and np.array(value).ndim == 2
            ):
                val_str = readable_matrix_string(np.array(value))
            elif isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
            metadata_layout.addRow(QLabel(f"{key_str}:"), QLabel(val_str))
        self.metadata_groupbox.setLayout(metadata_layout)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        if self.can_edit:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._setup_editable_controls()
        else:
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if self.can_emit:
                self._setup_confirm_cancel_buttons()
            else:
                self._setup_view_only_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.offset_label)
        if self.metadata:
            main_layout.addWidget(self.metadata_groupbox)
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)
        main_layout.addLayout(self.control_layout)
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)
        self.update_offset_label()

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""

        trans_group = QGroupBox("Translate (Display Pixels)")
        trans_layout = QHBoxLayout()
        int_validator = NullableIntValidator(-99999, 99999)

        self.dx_input = QLineEdit("0")
        self.dx_input.setValidator(int_validator)
        self.dx_input.setFixedWidth(50)

        self.dy_input = QLineEdit("0")
        self.dy_input.setValidator(int_validator)
        self.dy_input.setFixedWidth(50)

        self.apply_trans_button = QPushButton("Apply")
        trans_layout.addWidget(QLabel("dx:"))
        trans_layout.addWidget(self.dx_input)
        trans_layout.addWidget(QLabel("dy:"))
        trans_layout.addWidget(self.dy_input)
        trans_layout.addWidget(self.apply_trans_button)
        trans_group.setLayout(trans_layout)

        rot_group = QGroupBox("Rotate (°)")
        rot_layout = QHBoxLayout()
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Angle")
        self.rotation_input.setValidator(QDoubleValidator(-360.0, 360.0, 2))
        self.rotate_button = QPushButton("Apply")
        rot_layout.addWidget(self.rotation_input)
        rot_layout.addWidget(self.rotate_button)
        rot_group.setLayout(rot_layout)
        
        scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("1.0")
        self.scale_input.setValidator(QDoubleValidator(0.0001, 10000, 2))
        self.scale_button = QPushButton("Apply")
        scale_layout.addWidget(self.scale_input)
        scale_layout.addWidget(self.scale_button)
        scale_group.setLayout(scale_layout)

        self.apply_trans_button.clicked.connect(self.apply_manual_translation)
        # self.dx_input.returnPressed.connect(self.apply_manual_translation)
        # self.dy_input.returnPressed.connect(self.apply_manual_translation)
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.scale_button.clicked.connect(self.apply_scale)
        # self.rotation_input.returnPressed.connect(self.apply_rotation)

        self.reset_button = QPushButton("Reset Transformations")
        self.reset_button.clicked.connect(self.reset_zoom)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(rot_group)
        self.control_layout.addWidget(scale_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)
        self._setup_confirm_cancel_buttons()

    def _setup_confirm_cancel_buttons(self):
        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.clicked.connect(self.accept_alignment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addStretch()

    def apply_manual_translation(self):
        """Applies translation based on the dx/dy input fields."""
        try:
            xtext = self.dx_input.text()
            ytext = self.dy_input.text() 
            if xtext == "":
                xtext = "0"
            if ytext == "":
                ytext = "0"
            dx = int(xtext)
            dy = int(ytext)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid integer values for dx and dy.",
            )
            return

        if dx == 0 and dy == 0:
            return  # Nothing to do

        self.move_aligned_image(dx, dy)

    def apply_rotation(self):
        if not self.rotation_input.text():
            return
        try:
            angle = float(self.rotation_input.text())
            self.transformations.append([angle, []])

            transform = self.image_view.moving_item.transform()
            center = self.image_view.moving_item.boundingRect().center()
            t = QTransform()
            t.translate(center.x(), center.y())
            t.rotate(angle)
            t.translate(-center.x(), -center.y())

            self.image_view.moving_item.setTransform(transform * t)
            self.update_offset_label()
            # self.rotation_input.clear()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid rotation angle.")
    
    def apply_scale(self):
        if not self.scale_input.text():
            return
        try:
            scale = float(self.scale_input.text())
            if scale < 1.0:
                self.downscaled = True
            self.transformations[-1].append("x" + str(scale))

            transform = self.image_view.moving_item.transform()
            center = self.image_view.moving_item.boundingRect().center()
            t = QTransform()
            t.translate(center.x(), center.y())
            t.scale(scale, scale)
            t.translate(-center.x(), -center.y())

            self.image_view.moving_item.setTransform(transform * t)
            self.update_offset_label()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid scale factor.")

    def move_aligned_image(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.transformations[-1][1].append((dx, dy))

        transform = self.image_view.moving_item.transform()
        transform.translate(dx, dy)
        self.image_view.moving_item.setTransform(transform)

        self.update_offset_label()

    def reset_zoom(self, event=None):
        self.image_view.moving_item.resetTransform()
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _setup_view_only_controls(self):
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.close_button)
        self.button_layout.addStretch()

    def update_offset_label(self):
        transform_matrix = self.image_view.moving_item.transform()
        transform_text = readable_matrix_string(transform_to_matrix(transform_matrix))
        self.offset_label.setText(transform_text)

    def accept_alignment(self):
        self.result_accepted = True
        final_transformation = self.image_view.moving_item.transform()
        transf_matrix = transform_to_matrix(final_transformation)
        final_image = cv2.warpAffine(self.original_aligned_image,transf_matrix, self.target_image.shape[::-1])
        self.moving_image_changed.emit(final_image)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if not self.can_edit:
            super().keyPressEvent(event)
            return
        # Prevent arrow keys from being processed if a text input has focus
        if self.focusWidget() in [self.dx_input, self.dy_input, self.rotation_input]:
            super().keyPressEvent(event)
            return
        key_map = {
            Qt.Key.Key_Left: (-self.move_step, 0),
            Qt.Key.Key_Right: (self.move_step, 0),
            Qt.Key.Key_Up: (0, -self.move_step),
            Qt.Key.Key_Down: (0, self.move_step),
        }
        if event.key() in key_map:
            self.move_aligned_image(*key_map[event.key()])
        else:
            super().keyPressEvent(event)

    def create_direct_overlay(self):
        target_gray = self.to_uint8(self.target_image)
        aligned_gray = self.to_uint8(self.aligned_image)
        h, w = target_gray.shape
        ah, aw = aligned_gray.shape

        start_y = (ah - h) // 2
        start_x = (aw - w) // 2
        aligned_gray = aligned_gray[start_y:start_y + h, start_x:start_x + w]

        if self.adjust_contrast:
            target_gray = to_uint8(adjust_contrast(target_gray.astype(np.float64), 30, 99))
            aligned_gray = to_uint8(adjust_contrast(aligned_gray.astype(np.float64), 30, 99))

        # Create separate QPixmaps for both layers
        aligned_pixmap = colorize_grayscale(aligned_gray, "green")  
        target_pixmap = colorize_grayscale(target_gray, "red")  

        self.image_view.set_images(target_pixmap, aligned_pixmap)

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    
    def to_uint8(self, image):
        if image.dtype == np.uint8:
            return image
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.zeros_like(image, dtype=np.uint8)

    def get_current_aligned_image(self):
        return self.aligned_image


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
        rgba[:, :, 0] = gray_img  # R
    elif color == "green":
        rgba[:, :, 1] = gray_img  # G
    elif color == "blue":
        rgba[:, :, 2] = gray_img  # B

    # Make black (value 0) transparent
    mask = gray_img > 0
    rgba[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha

    qimage = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)

def transform_to_matrix(t: QTransform):
    matrix = np.array([
        [t.m11(), t.m21(), t.dx()],
        [t.m12(), t.m22(), t.dy()],
    ], dtype=np.float32)
    return matrix

