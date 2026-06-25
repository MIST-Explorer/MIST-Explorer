import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QFont,
    QKeyEvent,
    QPen,
    QTransform,
)
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.alignment.alignment_view_dialog import (
    AlignmentViewDialog,
    NullableIntValidator,
    ZoomableImageView,  # re-exported for backward compat
    colorize_grayscale,  # re-exported for backward compat
    readable_matrix_string,  # re-exported for backward compat
    transform_to_matrix,  # re-exported for backward compat
)
from utils import apply_ransac_affine_warp, to_uint8

__all__ = [
    "AlignmentPreviewDialog",
    "ZoomableImageView",
    "colorize_grayscale",
    "readable_matrix_string",
    "transform_to_matrix",
    "NullableIntValidator",
]


class AlignmentPreviewDialog(AlignmentViewDialog):
    """
    Full-featured alignment dialog for manual align and register-images use cases.

    Inherits overlay, contrast, opacity slider, and template infrastructure from
    AlignmentViewDialog. Adds transform controls, landmark RANSAC, and two confirm
    buttons that emit a single unified transformation_ready signal.
    """

    can_edit = True

    # Single signal replaces the old transformation_matrix, combined_transform_ready,
    # and edit-mode usage of moving_image_changed.
    # Payload: {"matrix": np.ndarray (2x3 combined), "action": "add_layer" | "replace_channel"}
    transformation_ready = pyqtSignal(object)

    def __init__(self, snapshot_data: dict, can_edit: bool = True, progress_callback=None):
        # can_edit param kept for backward compatibility (always True in subclass)
        super().__init__(snapshot_data, progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Template overrides
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        self.result_accepted = False
        self.transformations = [[0.0, []]]
        self.offset_x, self.offset_y, self.move_step = 0, 0, 1
        self.original_aligned_image = self._snapshot_data[
            "aligned_image"
        ]  # kept for test access; _snapshot_data cleared after __init__
        self.aligned_image = to_uint8(self.aligned_image)
        self.downscaled = False

        self._ransac_M: np.ndarray | None = None
        self._last_lm_src: np.ndarray | None = None
        self._last_lm_dst: np.ndarray | None = None

        self._lm_mode = False
        self._lm_waiting_for = "reference"  # "reference" | "moving"
        self._lm_src_pts: list[tuple[float, float]] = []
        self._lm_dst_pts: list[tuple[float, float]] = []
        self._lm_pending_ref: tuple[float, float] | None = None
        self._lm_ref_markers: list[tuple] = []
        self._lm_mov_markers: list[tuple] = []

    def _add_header_widgets(self, layout: QVBoxLayout) -> None:
        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.offset_label)

        if self.metadata:
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
            layout.addWidget(self.metadata_groupbox)

    def _add_editing_controls(self, layout: QVBoxLayout) -> None:
        idx = layout.indexOf(self.image_view)
        if idx != -1:
            layout.removeWidget(self.image_view)
            self.view_layout = QHBoxLayout()
            self.view_layout.addWidget(self.image_view, stretch=1)

            self.landmarks_sidebar = QWidget()
            self.landmarks_sidebar.setFixedWidth(150)
            sidebar_layout = QVBoxLayout(self.landmarks_sidebar)
            sidebar_layout.setContentsMargins(0, 0, 0, 0)

            header_layout = QHBoxLayout()
            header_layout.addWidget(QLabel("<b>Landmarks</b>"))
            self.hide_sidebar_btn = QPushButton("✕")
            self.hide_sidebar_btn.setFixedSize(20, 20)
            self.hide_sidebar_btn.setToolTip("Hide sidebar")
            self.hide_sidebar_btn.clicked.connect(self._hide_landmarks_sidebar)
            header_layout.addWidget(self.hide_sidebar_btn)
            sidebar_layout.addLayout(header_layout)

            self.landmarks_list_widget = QListWidget()
            self.landmarks_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.landmarks_list_widget.customContextMenuRequested.connect(self._on_sidebar_right_click)
            self.landmarks_list_widget.itemClicked.connect(
                self._on_landmark_list_item_clicked
            )
            sidebar_layout.addWidget(self.landmarks_list_widget)

            self.view_layout.addWidget(self.landmarks_sidebar)
            self.landmarks_sidebar.setVisible(False)
            layout.insertLayout(idx, self.view_layout)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.control_layout = QHBoxLayout()
        self._setup_editable_controls()
        layout.addLayout(self.control_layout)
        self.landmark_layout = QHBoxLayout()
        self._setup_landmark_controls()
        layout.addLayout(self.landmark_layout)
        self.update_offset_label()

    def _setup_buttons(self) -> None:
        self.confirm_button = QPushButton("Save")
        self.replace_button = QPushButton("Confirm and Replace")
        self.cancel_button = QPushButton("Cancel")
        self.confirm_button.clicked.connect(lambda: self._on_confirm("add_layer"))
        self.replace_button.clicked.connect(lambda: self._on_confirm("replace_channel"))
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.replace_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addStretch()

    def _update_preview_label(self) -> None:
        if getattr(self, "_lm_mode", False):
            self.preview_label.setText(
                "LANDMARK MODE — click reference (red), then matching point on moving layer (green). "
                "Esc to cancel."
            )
            self.preview_label.setStyleSheet(
                "color: #FFA500; font-weight: bold; padding: 4px;"
            )
        else:
            default = (
                "Arrow keys/Inputs: move, Shift+Drag: move layer, Mouse wheel: zoom, "
                "Drag: pan, Double-click: reset view"
            )
            self.preview_label.setText(f"Red = Target, Green = Aligned | {default}")
            self.preview_label.setStyleSheet("")

    def reset_zoom(self, event=None) -> None:
        if event and getattr(self, "_lm_mode", False):
            scene_pt = self.image_view.mapToScene(event.pos())
            items = self.image_view.get_scene().items(scene_pt)
            if any(isinstance(item, QGraphicsEllipseItem) for item in items):
                event.accept()
                return
        super().reset_zoom(event)

    def _reset_all(self) -> None:
        self.image_view.moving_item.resetTransform()
        self.offset_x, self.offset_y = 0, 0
        self.transformations = [[0.0, []]]
        self._ransac_M = None
        self.aligned_image = to_uint8(self.original_aligned_image.copy())
        self._refresh_overlay()
        self.reset_zoom()
        self.update_offset_label()

    # ------------------------------------------------------------------
    # Unified confirm — compose all transforms → single 2x3 → emit
    # ------------------------------------------------------------------

    def _build_combined_matrix(self) -> np.ndarray:
        affine = transform_to_matrix(self.image_view.moving_item.transform())
        if self._ransac_M is None:
            return affine
        ransac_3x3 = np.vstack([self._ransac_M, [0, 0, 1]])
        affine_3x3 = np.vstack([affine, [0, 0, 1]])
        return (affine_3x3 @ ransac_3x3)[:2, :]

    def _on_confirm(self, action: str = "add_layer") -> None:
        self.result_accepted = True
        matrix = self._build_combined_matrix()
        self.transformation_ready.emit({"matrix": matrix, "action": action})
        self.accept()

    # ------------------------------------------------------------------
    # Editable controls (translate / rotate / scale / flip)
    # ------------------------------------------------------------------

    def _setup_editable_controls(self) -> None:
        self.trans_group = QGroupBox("Translate (Display Pixels)")
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
        self.trans_group.setLayout(trans_layout)

        self.rot_group = QGroupBox("Rotate (°)")
        rot_layout = QHBoxLayout()
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Angle")
        self.rotation_input.setValidator(QDoubleValidator(-360.0, 360.0, 6))
        self.rotate_button = QPushButton("Apply")
        rot_layout.addWidget(self.rotation_input)
        rot_layout.addWidget(self.rotate_button)
        self.rot_group.setLayout(rot_layout)

        self.scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("1.0")
        self.scale_input.setValidator(QDoubleValidator(0.000001, 10000, 6))
        self.scale_button = QPushButton("Apply")
        scale_layout.addWidget(self.scale_input)
        scale_layout.addWidget(self.scale_button)
        self.scale_group.setLayout(scale_layout)

        self.flip_group = QGroupBox("Flip")
        flip_layout = QHBoxLayout()
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        flip_layout.addWidget(self.flip_horizontal_btn)
        flip_layout.addWidget(self.flip_vertical_btn)
        self.flip_group.setLayout(flip_layout)

        self.apply_trans_button.clicked.connect(self.apply_manual_translation)
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.scale_button.clicked.connect(self.apply_scale)
        self.flip_horizontal_btn.clicked.connect(self.apply_flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.apply_flip_vertical)

        self.reset_button = QPushButton("Reset Transformations")
        self.reset_button.clicked.connect(self._reset_all)

        self.control_layout.addWidget(self.trans_group)
        self.control_layout.addWidget(self.rot_group)
        self.control_layout.addWidget(self.scale_group)
        self.control_layout.addWidget(self.flip_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)

    def apply_manual_translation(self) -> None:
        try:
            xtext = self.dx_input.text() or "0"
            ytext = self.dy_input.text() or "0"
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
            return
        self.move_aligned_image(dx, dy)

    def apply_rotation(self) -> None:
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
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid rotation angle."
            )

    def apply_scale(self) -> None:
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
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid scale factor."
            )

    def apply_flip_horizontal(self) -> None:
        self.transformations[-1].append("flip_h")
        transform = self.image_view.moving_item.transform()
        center = self.image_view.moving_item.boundingRect().center()
        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(-1, 1)
        t.translate(-center.x(), -center.y())
        self.image_view.moving_item.setTransform(transform * t)
        self.update_offset_label()

    def apply_flip_vertical(self) -> None:
        self.transformations[-1].append("flip_v")
        transform = self.image_view.moving_item.transform()
        center = self.image_view.moving_item.boundingRect().center()
        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(1, -1)
        t.translate(-center.x(), -center.y())
        self.image_view.moving_item.setTransform(transform * t)
        self.update_offset_label()

    def move_aligned_image(self, dx: int, dy: int) -> None:
        self.offset_x += dx
        self.offset_y += dy
        self.transformations[-1][1].append((dx, dy))
        transform = self.image_view.moving_item.transform()
        new_transform = QTransform(
            transform.m11(),
            transform.m12(),
            transform.m21(),
            transform.m22(),
            transform.dx() + dx,
            transform.dy() + dy,
        )
        self.image_view.moving_item.setTransform(new_transform)
        self.update_offset_label()

    def _hide_landmarks_sidebar(self) -> None:
        self.landmarks_sidebar.setVisible(False)
        if hasattr(self, "toggle_list_btn"):
            self.toggle_list_btn.setChecked(False)

    def _toggle_landmarks_list(self) -> None:
        self.landmarks_sidebar.setVisible(self.toggle_list_btn.isChecked())

    def _update_landmarks_list(self) -> None:
        if not hasattr(self, "landmarks_list_widget"):
            return
        self.landmarks_list_widget.clear()
        for i in range(len(self._lm_src_pts)):
            self.landmarks_list_widget.addItem(f"Pair {i + 1}")
        if self._lm_pending_ref is not None:
            n = len(self._lm_src_pts) + 1
            self.landmarks_list_widget.addItem(f"Pair {n} (pending)")

    def _on_landmark_list_item_clicked(self, item) -> None:
        idx = self.landmarks_list_widget.row(item)
        if 0 <= idx < len(self._lm_src_pts):
            src_x, src_y = self._lm_src_pts[idx]
            self.image_view.centerOn(src_x, src_y)
            # Zoom in if current zoom is too low
            target_zoom = 3.0
            if self.image_view._current_zoom < target_zoom:
                factor = target_zoom / self.image_view._current_zoom
                self.image_view.scale(factor, factor)
                self.image_view._current_zoom = target_zoom
            self.image_view._is_fitted = False

    # ------------------------------------------------------------------
    # Landmark controls
    # ------------------------------------------------------------------

    def _setup_landmark_controls(self) -> None:
        self.landmark_button = QPushButton("Start Landmark")
        self.landmark_reuse_button = QPushButton("Reuse Previous")
        self.landmark_reuse_button.setEnabled(False)
        self.landmark_reuse_button.setVisible(False)
        self.landmark_reuse_button.setToolTip(
            "Restore landmarks from the previous RANSAC estimation"
        )

        self.landmark_undo_button = QPushButton("Undo Last")
        self.landmark_undo_button.setEnabled(False)
        self.landmark_cancel_button = QPushButton("Cancel Landmark")
        self.landmark_cancel_button.setEnabled(False)
        self.landmark_cancel_button.setToolTip("Cancel landmark mode (Esc)")
        self.landmark_status_label = QLabel("")

        self.toggle_list_btn = QPushButton("Toggle List")
        self.toggle_list_btn.setCheckable(True)
        self.toggle_list_btn.setChecked(True)
        self.toggle_list_btn.setVisible(False)
        self.toggle_list_btn.clicked.connect(self._toggle_landmarks_list)

        self.import_landmarks_btn = QPushButton("Import Landmarks JSON")
        self.import_landmarks_btn.clicked.connect(self._import_landmarks_from_json)

        self.export_landmarks_btn = QPushButton("Export Landmarks JSON")
        self.export_landmarks_btn.setEnabled(False)
        self.export_landmarks_btn.clicked.connect(self._export_landmarks_to_json)

        self.landmark_button.clicked.connect(self._on_landmark_button_clicked)
        self.landmark_reuse_button.clicked.connect(self._reuse_last_landmarks)
        self.landmark_undo_button.clicked.connect(self._undo_landmark)
        self.landmark_cancel_button.clicked.connect(self._cancel_landmark_mode)

        ransac_threshold_label = QLabel("RANSAC Threshold (px):")
        self.landmark_ransac_threshold_spinbox = QDoubleSpinBox()
        self.landmark_ransac_threshold_spinbox.setRange(1.0, 50.0)
        self.landmark_ransac_threshold_spinbox.setSingleStep(0.5)
        self.landmark_ransac_threshold_spinbox.setValue(10.0)
        self.landmark_ransac_threshold_spinbox.setDecimals(1)
        self.landmark_ransac_threshold_spinbox.setToolTip(
            "RANSAC inlier threshold in pixels (lower = stricter outlier rejection)"
        )

        self.landmark_layout.addWidget(self.landmark_button)
        self.landmark_layout.addWidget(self.landmark_reuse_button)
        self.landmark_layout.addWidget(self.landmark_undo_button)
        self.landmark_layout.addWidget(self.landmark_cancel_button)
        self.landmark_layout.addWidget(self.toggle_list_btn)
        self.landmark_layout.addWidget(self.landmark_status_label)
        self.landmark_layout.addStretch()
        self.landmark_layout.addWidget(ransac_threshold_label)
        self.landmark_layout.addWidget(self.landmark_ransac_threshold_spinbox)
        self.landmark_layout.addWidget(self.import_landmarks_btn)
        self.landmark_layout.addWidget(self.export_landmarks_btn)

    def _set_transform_controls_enabled(self, enabled: bool) -> None:
        for attr in (
            "trans_group",
            "rot_group",
            "scale_group",
            "flip_group",
            "reset_button",
            "confirm_button",
            "replace_button",
            "enhance_contrast_checkbox",
            "invert_fixed_checkbox",
            "invert_moving_checkbox",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(
                    enabled
                )  # hide controls when in landmark mode to avoid confusion

    def _on_landmark_button_clicked(self) -> None:
        if not self._lm_mode:
            self._start_landmark_mode()
        elif len(self._lm_src_pts) >= 3:
            self._confirm_ransac_affine()

    def _start_landmark_mode(self) -> None:
        self._lm_mode = True
        self._lm_waiting_for = "reference"
        self.image_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.image_view.setCursor(Qt.CursorShape.CrossCursor)
        self.image_view.setStyleSheet("QGraphicsView { border: 2px solid #FFA500; }")
        self._set_transform_controls_enabled(False)
        self._update_preview_label()
        self.toggle_list_btn.setVisible(True)
        if self.toggle_list_btn.isChecked():
            self.landmarks_sidebar.setVisible(True)
        self._update_landmark_ui()

    def _exit_landmark_mode(self) -> None:
        self._lm_mode = False
        self.image_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.image_view.unsetCursor()
        self.image_view.setStyleSheet("")
        self._set_transform_controls_enabled(True)
        self._update_preview_label()
        self.toggle_list_btn.setVisible(False)
        self.landmarks_sidebar.setVisible(False)

    def _cancel_landmark_mode(self) -> None:
        if not self._lm_mode:
            return
        for e, t in self._lm_ref_markers + self._lm_mov_markers:
            self.image_view.get_scene().removeItem(e)
            self.image_view.get_scene().removeItem(t)
        self._lm_src_pts = []
        self._lm_dst_pts = []
        self._lm_pending_ref = None
        self._lm_ref_markers = []
        self._lm_mov_markers = []
        self._exit_landmark_mode()
        self._update_landmark_ui()

    def _on_landmark_click(self, scene_x: float, scene_y: float) -> None:
        radius = self.landmark_ransac_threshold_spinbox.value()
        if self._lm_waiting_for == "reference":
            self._lm_pending_ref = (scene_x, scene_y)
            n = len(self._lm_src_pts) + 1
            marker = self._add_scene_marker(
                scene_x, scene_y, n, QColor(255, 80, 80), radius=radius
            )
            self._lm_ref_markers.append(marker)
            self._lm_waiting_for = "moving"
        elif self._lm_waiting_for == "moving" and self._lm_pending_ref is not None:
            local_pt = self.image_view.moving_item.mapFromScene(
                QPointF(scene_x, scene_y)
            )
            self._lm_src_pts.append(self._lm_pending_ref)
            self._lm_dst_pts.append((local_pt.x(), local_pt.y()))
            self._lm_pending_ref = None
            n = len(self._lm_src_pts)
            marker = self._add_scene_marker(
                scene_x, scene_y, n, QColor(80, 210, 80), radius=radius
            )
            self._lm_mov_markers.append(marker)
            self._lm_waiting_for = "reference"
        self._update_landmark_ui()

    def _on_landmark_right_click(self, scene_x: float, scene_y: float) -> None:
        """Identify if a marker was clicked and delete it."""
        # Check reference markers
        for i, (ellipse, text) in enumerate(self._lm_ref_markers):
            if ellipse.contains(ellipse.mapFromScene(QPointF(scene_x, scene_y))):
                self._delete_landmark_at_index(i)
                return
        # Check moving markers
        for i, (ellipse, text) in enumerate(self._lm_mov_markers):
            if ellipse.contains(ellipse.mapFromScene(QPointF(scene_x, scene_y))):
                self._delete_landmark_at_index(i)
                return

    def _on_sidebar_right_click(self, pos) -> None:
        item = self.landmarks_list_widget.itemAt(pos)
        if item:
            idx = self.landmarks_list_widget.row(item)
            self._delete_landmark_at_index(idx)

    def _delete_landmark_at_index(self, index: int) -> None:
        num_completed = len(self._lm_src_pts)
        is_pending = (index == num_completed) and (self._lm_pending_ref is not None)

        if index < num_completed:
            # Delete completed pair
            self._lm_src_pts.pop(index)
            self._lm_dst_pts.pop(index)
            # Remove markers from scene
            for marker_list in (self._lm_ref_markers, self._lm_mov_markers):
                e, t = marker_list.pop(index)
                self.image_view.get_scene().removeItem(e)
                self.image_view.get_scene().removeItem(t)
        elif is_pending:
            # Delete pending reference
            self._lm_pending_ref = None
            self._lm_waiting_for = "reference"
            e, t = self._lm_ref_markers.pop(index)
            self.image_view.get_scene().removeItem(e)
            self.image_view.get_scene().removeItem(t)
        else:
            return

        # Re-number remaining markers
        for i in range(index, len(self._lm_ref_markers)):
            _, t = self._lm_ref_markers[i]
            t.setText(str(i + 1))
        for i in range(index, len(self._lm_mov_markers)):
            _, t = self._lm_mov_markers[i]
            t.setText(str(i + 1))

        self._update_landmark_ui()

    def _add_scene_marker(
        self, x: float, y: float, number: int, color: QColor, radius: float = 8.0
    ) -> tuple:
        r = max(4.0, radius)  # Ensure it's never too small to see
        ellipse = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
        fill = QColor(color.red(), color.green(), color.blue(), 130)
        ellipse.setBrush(QBrush(fill))
        ellipse.setPen(QPen(color, 2))
        ellipse.setZValue(20)
        self.image_view.get_scene().addItem(ellipse)

        text = QGraphicsSimpleTextItem(str(number))
        font = QFont("Arial", max(8, int(r * 0.8)), QFont.Weight.Bold)
        text.setFont(font)
        text.setBrush(QBrush(Qt.GlobalColor.white))
        text.setPos(x + r + 1, y - r - 1)
        text.setZValue(21)
        self.image_view.get_scene().addItem(text)

        return (ellipse, text)

    def _undo_landmark(self) -> None:
        if self._lm_waiting_for == "moving" and self._lm_pending_ref is not None:
            self._lm_pending_ref = None
            if self._lm_ref_markers:
                e, t = self._lm_ref_markers.pop()
                self.image_view.get_scene().removeItem(e)
                self.image_view.get_scene().removeItem(t)
            self._lm_waiting_for = "reference"
        elif self._lm_src_pts:
            self._lm_src_pts.pop()
            self._lm_dst_pts.pop()
            for marker_list in (self._lm_ref_markers, self._lm_mov_markers):
                if marker_list:
                    e, t = marker_list.pop()
                    self.image_view.get_scene().removeItem(e)
                    self.image_view.get_scene().removeItem(t)
        self._update_landmark_ui()

    def _update_landmark_ui(self) -> None:
        n = len(self._lm_src_pts)
        has_pending = self._lm_pending_ref is not None
        self.landmark_undo_button.setEnabled(self._lm_mode and (bool(n) or has_pending))
        self.landmark_cancel_button.setEnabled(self._lm_mode)

        self.landmark_reuse_button.setVisible(self._lm_mode)
        self.landmark_reuse_button.setEnabled(
            self._lm_mode and self._last_lm_src is not None
        )

        has_data = bool(n) or (self._ransac_M is not None)
        self.export_landmarks_btn.setEnabled(has_data)

        self._update_landmarks_list()

        if not self._lm_mode:
            self.landmark_button.setText("Start Landmark")
            self.landmark_button.setEnabled(True)
            self.landmark_status_label.setText("")
            self.landmark_status_label.setStyleSheet("")
            return

        if self._lm_waiting_for == "reference":
            self.landmark_status_label.setText(
                f"Pair {n + 1} — click REFERENCE point (red)"
            )
            self.landmark_status_label.setStyleSheet("color: #d33; font-weight: bold;")
        else:
            self.landmark_status_label.setText(
                f"Pair {n + 1} — click MATCHING point on moving layer (green)"
            )
            self.landmark_status_label.setStyleSheet(
                "color: #2a9d2a; font-weight: bold;"
            )

        if n >= 3:
            self.landmark_button.setText(f"Compute RANSAC Affine ({n} pairs)")
            self.landmark_button.setEnabled(True)
        else:
            self.landmark_button.setText(f"Need {3 - n} more pair(s)")
            self.landmark_button.setEnabled(False)

    def _reuse_last_landmarks(self) -> None:
        if self._last_lm_src is None or self._last_lm_dst is None:
            return

        # Clear any pending reference click to avoid mismatching
        if self._lm_waiting_for == "moving" and self._lm_pending_ref is not None:
            self._undo_landmark()

        radius = self.landmark_ransac_threshold_spinbox.value()
        for s, d in zip(self._last_lm_src, self._last_lm_dst):
            n = len(self._lm_src_pts) + 1
            # src (s) is in scene space
            self._lm_src_pts.append(tuple(s))
            ref_marker = self._add_scene_marker(
                s[0], s[1], n, QColor(255, 80, 80), radius=radius
            )
            self._lm_ref_markers.append(ref_marker)

            # dst (d) is in local space of CURRENT baked pixmap
            scene_pt = self.image_view.moving_item.mapToScene(QPointF(d[0], d[1]))
            self._lm_dst_pts.append(tuple(d))
            mov_marker = self._add_scene_marker(
                scene_pt.x(), scene_pt.y(), n, QColor(80, 210, 80), radius=radius
            )
            self._lm_mov_markers.append(mov_marker)

        self._update_landmark_ui()

    def _bake_ui_transform(self, qt_matrix: np.ndarray, w: int, h: int) -> None:
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        if np.allclose(qt_matrix, identity):
            return
        # qt_matrix is a forward map (SRC→DST). By default, cv2.warpAffine
        # expects a forward map and inverts it internally.
        baked = cv2.warpAffine(
            self.aligned_image, qt_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
        )
        self.aligned_image = baked
        qt_3x3 = np.vstack([qt_matrix, [0, 0, 1]])
        if self._ransac_M is None:
            self._ransac_M = qt_matrix
        else:
            prev_3x3 = np.vstack([self._ransac_M, [0, 0, 1]])
            self._ransac_M = (qt_3x3 @ prev_3x3)[:2, :]

    def _confirm_ransac_affine(self) -> None:
        if len(self._lm_src_pts) < 3:
            return
        src = np.array(self._lm_src_pts, dtype=np.float64)
        dst = np.array(self._lm_dst_pts, dtype=np.float64)
        threshold = self.landmark_ransac_threshold_spinbox.value()

        h, w = self.target_image.shape[:2]
        qt_matrix = transform_to_matrix(self.image_view.moving_item.transform())
        self._bake_ui_transform(qt_matrix, w, h)

        # _lm_dst_pts were collected via mapFromScene → local pixmap coords (pre-transform).
        # After baking, aligned_image has those pixels at qt_matrix @ local_pt positions.
        # RANSAC must receive coordinates in the baked image's space, not local item space.
        dst_h = np.hstack([dst, np.ones((len(dst), 1), dtype=np.float64)])
        dst_baked = dst_h @ qt_matrix.astype(np.float64).T  # (N,3) @ (3,2) → (N,2)

        try:
            warped, meta = apply_ransac_affine_warp(
                self.aligned_image,
                src,
                dst_baked,
                ransac_threshold=threshold,
                out_shape=(h, w),
            )
        except Exception as exc:
            QMessageBox.warning(self, "RANSAC Affine Error", str(exc))
            return

        new_M = meta["M"]
        if self._ransac_M is None:
            self._ransac_M = new_M
        else:
            prev_3x3 = np.vstack([self._ransac_M, [0, 0, 1]])
            new_3x3 = np.vstack([new_M, [0, 0, 1]])
            self._ransac_M = (new_3x3 @ prev_3x3)[:2, :]

        # Save for possible reuse
        self._last_lm_src = src.copy()
        # Project dst_baked through new_M to find their new locations in the warped image
        dst_baked_h = np.hstack([dst_baked, np.ones((len(dst_baked), 1), dtype=np.float64)])
        self._last_lm_dst = (dst_baked_h @ new_M.T).copy()

        self.aligned_image = warped
        self._refresh_overlay()
        self.image_view.moving_item.resetTransform()
        self.update_offset_label()
        for e, t in self._lm_ref_markers + self._lm_mov_markers:
            self.image_view.get_scene().removeItem(e)
            self.image_view.get_scene().removeItem(t)
        self._lm_src_pts = []
        self._lm_dst_pts = []
        self._lm_pending_ref = None
        self._lm_ref_markers = []
        self._lm_mov_markers = []
        self._exit_landmark_mode()
        self._update_landmark_ui()
        self.landmark_status_label.setText(
            f"Inliers: {meta['inliers']}/{len(src)}  reproj: {meta['reprojection_px']:.1f}px"
        )

    def _import_landmarks_from_json(self) -> None:
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Landmarks", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            batch = data["current_batch"]
            landmarks = batch["landmarks"]
            threshold = float(batch.get("ransac_threshold_px", 10.0))
        except Exception as exc:
            QMessageBox.warning(
                self, "Import Error", f"Could not parse landmarks file:\n{exc}"
            )
            return

        if not landmarks:
            QMessageBox.warning(self, "Import Error", "No landmarks found in file.")
            return

        if not self._lm_mode:
            self._start_landmark_mode()

        self.landmark_ransac_threshold_spinbox.setValue(threshold)

        for lm in landmarks:
            src = lm["src"]
            dst = lm["dst"]
            n = len(self._lm_src_pts) + 1

            ref_marker = self._add_scene_marker(
                src[0], src[1], n, QColor(255, 80, 80), radius=threshold
            )
            self._lm_ref_markers.append(ref_marker)
            self._lm_src_pts.append((src[0], src[1]))

            mov_scene = self.image_view.moving_item.mapToScene(QPointF(dst[0], dst[1]))
            mov_marker = self._add_scene_marker(
                mov_scene.x(), mov_scene.y(), n, QColor(80, 210, 80), radius=threshold
            )
            self._lm_mov_markers.append(mov_marker)
            self._lm_dst_pts.append((dst[0], dst[1]))

        self._lm_pending_ref = None
        self._lm_waiting_for = "reference"
        self._update_landmark_ui()

    def _export_landmarks_to_json(self) -> None:
        import datetime
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Landmarks", "landmarks.json", "JSON Files (*.json)"
        )
        if not path:
            return

        data = {
            "metadata": {
                "exported_at": datetime.datetime.now().isoformat(),
                "coordinate_format": "col_row",
                "note": "src = reference image, dst = moving image",
            },
            "current_batch": {
                "ransac_threshold_px": self.landmark_ransac_threshold_spinbox.value(),
                "landmarks": [
                    {
                        "index": i + 1,
                        "src": list(map(float, s)),
                        "dst": list(map(float, d)),
                    }
                    for i, (s, d) in enumerate(zip(self._lm_src_pts, self._lm_dst_pts))
                ],
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def update_offset_label(self) -> None:
        transform_matrix = self.image_view.moving_item.transform()
        transform_text = readable_matrix_string(transform_to_matrix(transform_matrix))
        self.offset_label.setText(transform_text)

    def get_current_aligned_image(self) -> np.ndarray:
        return self.aligned_image

    def accept_alignment(self) -> None:
        """Backward-compatible alias for _on_confirm('add_layer')."""
        self._on_confirm("add_layer")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._lm_mode:
            self._cancel_landmark_mode()
            event.accept()
            return
        if self._lm_mode:
            super().keyPressEvent(event)
            return
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
