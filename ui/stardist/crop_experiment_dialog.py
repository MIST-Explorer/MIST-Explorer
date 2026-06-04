import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.color import label2rgb as sk_label2rgb

from core.cellprofiler_segmentation import segment_primary_objects
from core.image_utils import create_lut, numpy_to_qimage, scale_adjust
from ui.stardist.cellprofiler_ui import CellProfilerAdvancedSettings

# ---------------------------------------------------------------------------
# Crop-capable QGraphicsView
# ---------------------------------------------------------------------------


class CropImageView(QGraphicsView):
    """QGraphicsView that lets the user draw a crop rectangle."""

    crop_selected = pyqtSignal(QRectF)  # scene-coordinate rect

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self._drawing = False
        self._origin: QPointF | None = None
        self._rect_item: QGraphicsRectItem | None = None
        self._current_zoom = 1.0

    def set_image(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_zoom = 1.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item and not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            if self._current_zoom != 1.0:
                self.scale(self._current_zoom, self._current_zoom)

    # -- rubber-band crop via mouse ----------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = self.mapToScene(event.pos())
            if self._rect_item is not None:
                self._scene.removeItem(self._rect_item)
            pen = QPen(Qt.GlobalColor.cyan, 0)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._rect_item = self._scene.addRect(
                QRectF(self._origin, self._origin), pen
            )
            self._drawing = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._origin is not None and self._rect_item is not None:
            current = self.mapToScene(event.pos())
            rect = QRectF(self._origin, current).normalized()
            self._rect_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if self._rect_item is not None:
                rect = self._rect_item.rect()
                if rect.width() > 2 and rect.height() > 2:
                    self.crop_selected.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event is None:
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1 / 1.15
        new_zoom = self._current_zoom * factor
        if new_zoom < 0.1 or new_zoom > 150.0:
            return
        self._current_zoom = new_zoom
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Simple preview QGraphicsView (non-interactive aside from zoom/pan)
# ---------------------------------------------------------------------------


class PreviewImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._current_zoom = 1.0

    def set_image(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_zoom = 1.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item and not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            if self._current_zoom != 1.0:
                self.scale(self._current_zoom, self._current_zoom)

    def update_image(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)

    def wheelEvent(self, event):
        if event is None:
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1 / 1.15
        new_zoom = self._current_zoom * factor
        if new_zoom < 0.1 or new_zoom > 150.0:
            return
        self._current_zoom = new_zoom
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class CropExperimentDialog(QDialog):
    """Two-phase dialog: crop a region, then experiment with segmentation settings."""

    settings_accepted = pyqtSignal(dict)

    def __init__(
        self,
        cell_image: np.ndarray,
        contrast_min: int,
        contrast_max: int,
        current_settings: dict,
        min_size: int,
        max_size: int,
        enable_dilation: bool,
        dilation_radius: int,
        use_contrasted_image: bool,
    ):
        super().__init__(None)
        self.setWindowTitle("Sample & Test")
        self.resize(1200, 800)

        self._cell_image = cell_image
        self._contrast_min = contrast_min
        self._contrast_max = contrast_max
        self._current_settings = dict(current_settings)
        self._init_min_size = int(min_size)
        self._init_max_size = int(max_size)
        self._init_enable_dilation = bool(enable_dilation)
        self._init_dilation_radius = int(dilation_radius)
        self._init_use_contrasted = bool(use_contrasted_image)

        self._cropped_raw: np.ndarray | None = None
        self._cropped_display: np.ndarray | None = None

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._run_segmentation)

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        main_layout.addWidget(self._stack)

        # --- Page 0: Crop ---
        crop_page = QWidget()
        crop_layout = QVBoxLayout(crop_page)

        crop_layout.addWidget(
            QLabel("Draw a rectangle on the image to select a crop region.")
        )

        self._crop_view = CropImageView()
        self._crop_view.setMinimumSize(600, 400)
        crop_layout.addWidget(self._crop_view)

        crop_btn_layout = QHBoxLayout()
        self._confirm_crop_btn = QPushButton("Confirm Crop")
        self._confirm_crop_btn.setEnabled(False)
        self._cancel_crop_btn = QPushButton("Cancel")
        crop_btn_layout.addStretch()
        crop_btn_layout.addWidget(self._confirm_crop_btn)
        crop_btn_layout.addWidget(self._cancel_crop_btn)
        crop_btn_layout.addStretch()
        crop_layout.addLayout(crop_btn_layout)

        self._stack.addWidget(crop_page)

        # --- Page 1: Experiment ---
        experiment_page = QWidget()
        experiment_layout = QVBoxLayout(experiment_page)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel — settings in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)

        # Basic controls
        settings_layout.addWidget(QLabel("<b>Basic Settings</b>"))

        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Size (diameter px)"))
        self._exp_min_size = QSpinBox()
        self._exp_min_size.setRange(1, 10000)
        self._exp_min_size.setValue(self._init_min_size)
        min_layout.addWidget(self._exp_min_size)
        settings_layout.addLayout(min_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Size (diameter px)"))
        self._exp_max_size = QSpinBox()
        self._exp_max_size.setRange(1, 10000)
        self._exp_max_size.setValue(self._init_max_size)
        max_layout.addWidget(self._exp_max_size)
        settings_layout.addLayout(max_layout)

        self._exp_enable_dilation = QCheckBox("Enable Dilation")
        self._exp_enable_dilation.setChecked(self._init_enable_dilation)
        settings_layout.addWidget(self._exp_enable_dilation)

        rad_layout = QHBoxLayout()
        rad_layout.addWidget(QLabel("Dilation Radius"))
        self._exp_dilation_radius = QSpinBox()
        self._exp_dilation_radius.setRange(0, 100)
        self._exp_dilation_radius.setValue(self._init_dilation_radius)
        self._exp_dilation_radius.setEnabled(self._init_enable_dilation)
        rad_layout.addWidget(self._exp_dilation_radius)
        settings_layout.addLayout(rad_layout)

        self._exp_use_contrasted = QCheckBox("Use contrasted image (toolbar sliders)")
        self._exp_use_contrasted.setChecked(self._init_use_contrasted)
        settings_layout.addWidget(self._exp_use_contrasted)

        # Advanced controls — reuse CellProfilerAdvancedSettings widget
        settings_layout.addWidget(QLabel("<b>Advanced Settings</b>"))
        self._exp_cp_advanced = CellProfilerAdvancedSettings()
        self._exp_cp_advanced.crop_experiment_button.hide()
        settings_layout.addWidget(self._exp_cp_advanced)

        settings_layout.addStretch()
        scroll.setWidget(settings_container)
        splitter.addWidget(scroll)

        # Right panel — preview
        self._preview_view = PreviewImageView()
        self._preview_view.setMinimumSize(400, 400)
        splitter.addWidget(self._preview_view)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])
        experiment_layout.addWidget(splitter)

        exp_btn_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save Config")
        self._cancel_exp_btn = QPushButton("Cancel")
        exp_btn_layout.addStretch()
        exp_btn_layout.addWidget(self._save_btn)
        exp_btn_layout.addWidget(self._cancel_exp_btn)
        exp_btn_layout.addStretch()
        experiment_layout.addLayout(exp_btn_layout)

        self._stack.addWidget(experiment_page)

        # Start on crop page
        self._stack.setCurrentIndex(0)

        # Render cell image for crop view
        display = self._render_contrasted(self._cell_image)
        qimg = numpy_to_qimage(np.ascontiguousarray(display))
        self._crop_view.set_image(QPixmap.fromImage(qimg))

        # Connections
        self._crop_view.crop_selected.connect(self._on_crop_selected)
        self._confirm_crop_btn.clicked.connect(self._on_confirm_crop)
        self._cancel_crop_btn.clicked.connect(self.reject)
        self._save_btn.clicked.connect(self._on_save)
        self._cancel_exp_btn.clicked.connect(self.reject)
        self._exp_enable_dilation.toggled.connect(self._exp_dilation_radius.setEnabled)

        # Initialize advanced widget values from current_settings
        self._load_settings_to_widgets(self._current_settings)

    # ------------------------------------------------------------------
    # Contrast rendering
    # ------------------------------------------------------------------

    def _render_contrasted(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast LUT and return uint8 image."""
        uint8 = scale_adjust(image)
        lut = create_lut(
            int(np.clip(self._contrast_min, 0, 255)),
            int(np.clip(self._contrast_max, 0, 255)),
        )
        return np.clip(cv2.LUT(uint8, lut), 0, 255).astype(np.uint8)

    def _window_image_by_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast windowing for segmentation input (returns uint16)."""
        uint8 = scale_adjust(image)
        cmin = int(np.clip(self._contrast_min, 0, 255))
        cmax = int(np.clip(self._contrast_max, 0, 255))
        if cmax < cmin:
            cmin, cmax = cmax, cmin
        if cmin == cmax:
            if cmax < 255:
                cmax += 1
            elif cmin > 0:
                cmin -= 1
            else:
                return uint8.astype(np.uint16) * 257
        lut = create_lut(cmin, cmax)
        contrasted = np.clip(cv2.LUT(uint8, lut), 0, 254).astype(np.uint8)
        return contrasted.astype(np.uint16) * 257

    # ------------------------------------------------------------------
    # Crop phase
    # ------------------------------------------------------------------

    _crop_rect: QRectF | None = None

    def _on_crop_selected(self, rect: QRectF):
        self._crop_rect = rect
        self._confirm_crop_btn.setEnabled(True)

    def _on_confirm_crop(self):
        if self._crop_rect is None:
            return
        r = self._crop_rect
        h, w = self._cell_image.shape[:2]
        y0 = max(0, int(r.top()))
        y1 = min(h, int(r.bottom()))
        x0 = max(0, int(r.left()))
        x1 = min(w, int(r.right()))
        if y1 - y0 < 2 or x1 - x0 < 2:
            return

        self._cropped_raw = self._cell_image[y0:y1, x0:x1].copy()
        self._cropped_display = self._render_contrasted(self._cropped_raw)

        self._stack.setCurrentIndex(1)
        self._connect_experiment_signals()
        self._run_segmentation()

    # ------------------------------------------------------------------
    # Experiment phase — live update
    # ------------------------------------------------------------------

    def _connect_experiment_signals(self):
        """Wire all experiment widgets to schedule a segmentation re-run."""
        cp = self._exp_cp_advanced

        for spin in (
            cp.threshold_correction_factor,
            cp.threshold_smoothing_scale,
            cp.threshold_lower_bound,
            cp.threshold_upper_bound,
            cp.manual_threshold,
            cp.object_fraction,
            cp.lower_outlier_fraction,
            cp.upper_outlier_fraction,
            cp.number_of_deviations,
            cp.smoothing_filter_size,
            cp.maxima_suppression_size,
            cp.adaptive_window_size,
            self._exp_min_size,
            self._exp_max_size,
            self._exp_dilation_radius,
        ):
            spin.valueChanged.connect(self._schedule_update)

        for combo in (
            cp.threshold_method,
            cp.threshold_scope,
            cp.otsu_class,
            cp.otsu_middle,
            cp.averaging_method,
            cp.variance_method,
        ):
            combo.currentTextChanged.connect(self._schedule_update)

        for cb in (
            cp.fill_holes_thresholding,
            cp.fill_holes_declumping,
            cp.automatic_smoothing,
            cp.automatic_maxima_suppression,
            cp.low_res_maxima,
            cp.exclude_border_objects,
            cp.invert_image,
            self._exp_enable_dilation,
            self._exp_use_contrasted,
        ):
            cb.toggled.connect(self._schedule_update)

    def _schedule_update(self, *_args):
        self._debounce_timer.start()

    def _collect_settings(self) -> dict:
        """Read all widget values into a settings dict."""
        cp = self._exp_cp_advanced
        return {
            "threshold_method": cp.threshold_method.currentText(),
            "threshold_scope": cp.threshold_scope.currentText(),
            "threshold_smoothing_scale": cp.threshold_smoothing_scale.value(),
            "threshold_correction_factor": cp.threshold_correction_factor.value(),
            "threshold_range": (
                cp.threshold_lower_bound.value(),
                cp.threshold_upper_bound.value(),
            ),
            "threshold_lower_bound": cp.threshold_lower_bound.value(),
            "threshold_upper_bound": cp.threshold_upper_bound.value(),
            "manual_threshold": cp.manual_threshold.value(),
            "two_class_otsu": cp.otsu_class.currentText() == "Two classes",
            "assign_middle_to_foreground": cp.otsu_middle.currentText() == "Foreground",
            "object_fraction": cp.object_fraction.value(),
            "lower_outlier_fraction": cp.lower_outlier_fraction.value(),
            "upper_outlier_fraction": cp.upper_outlier_fraction.value(),
            "averaging_method": cp.averaging_method.currentText(),
            "variance_method": cp.variance_method.currentText(),
            "number_of_deviations": cp.number_of_deviations.value(),
            "adaptive_window_size": cp.adaptive_window_size.value(),
            "fill_holes_after_thresholding": cp.fill_holes_thresholding.isChecked(),
            "fill_holes_after_declumping": cp.fill_holes_declumping.isChecked(),
            "automatic_smoothing": cp.automatic_smoothing.isChecked(),
            "smoothing_filter_size": cp.smoothing_filter_size.value(),
            "automatic_maxima_suppression": cp.automatic_maxima_suppression.isChecked(),
            "maxima_suppression_size": cp.maxima_suppression_size.value(),
            "low_res_maxima": cp.low_res_maxima.isChecked(),
            "exclude_border_objects": cp.exclude_border_objects.isChecked(),
            # basic settings
            "min_size": self._exp_min_size.value(),
            "max_size": self._exp_max_size.value(),
            "enable_dilation": self._exp_enable_dilation.isChecked(),
            "dilation_radius": self._exp_dilation_radius.value(),
            "use_contrasted_image": self._exp_use_contrasted.isChecked(),
            "invert_image": self._exp_cp_advanced.invert_image.isChecked(),
        }

    def _run_segmentation(self):
        if self._cropped_raw is None:
            return

        settings = self._collect_settings()

        # Choose segmentation input based on use_contrasted toggle
        if settings["use_contrasted_image"]:
            seg_input = self._window_image_by_contrast(self._cropped_raw)
        else:
            seg_input = self._cropped_raw

        try:
            labels = segment_primary_objects(seg_input, settings)
        except Exception:
            labels = np.zeros(self._cropped_raw.shape[:2], dtype=np.int32)

        if settings.get("invert_image", False) and self._cropped_display is not None:
            display_base = (255 - self._cropped_display).astype(self._cropped_display.dtype)
        else:
            display_base = self._cropped_display

        overlay = self._render_preview(labels, display_base)
        qimg = numpy_to_qimage(np.ascontiguousarray(overlay))
        self._preview_view.update_image(QPixmap.fromImage(qimg))

    def _render_preview(self, labels: np.ndarray, display_base: np.ndarray | None = None) -> np.ndarray:
        """Blend segmentation labels over the contrasted crop image."""
        base = display_base if display_base is not None else self._cropped_display
        if base is None:
            return np.zeros((*labels.shape, 3), dtype=np.uint8)

        if base.ndim == 2:
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = base[:, :, :3].copy()

        if labels.max() == 0:
            return base_rgb

        label_rgb = (sk_label2rgb(labels, bg_label=0, bg_color=(0, 0, 0)) * 255).astype(
            np.uint8
        )
        mask = labels > 0
        alpha = 0.4
        blended = base_rgb.astype(np.float32)
        blended[mask] = (1.0 - alpha) * blended[mask] + alpha * label_rgb[mask].astype(
            np.float32
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Settings init / save
    # ------------------------------------------------------------------

    def _load_settings_to_widgets(self, s: dict):
        """Initialize experiment widgets from a settings dict."""
        cp = self._exp_cp_advanced

        cp.threshold_method.setCurrentText(s.get("threshold_method", "MCT"))
        cp.threshold_scope.setCurrentText(s.get("threshold_scope", "Global"))
        cp.threshold_correction_factor.setValue(
            float(s.get("threshold_correction_factor", 1.0))
        )
        cp.threshold_smoothing_scale.setValue(
            float(s.get("threshold_smoothing_scale", 1.3488))
        )

        # Handle both threshold_range tuple and separate lower/upper keys
        if "threshold_range" in s:
            lo, hi = s["threshold_range"]
        else:
            lo = float(s.get("threshold_lower_bound", 0.0))
            hi = float(s.get("threshold_upper_bound", 1.0))
        cp.threshold_lower_bound.setValue(lo)
        cp.threshold_upper_bound.setValue(hi)

        cp.manual_threshold.setValue(float(s.get("manual_threshold", 0.5)))

        two_class = s.get("two_class_otsu", True)
        cp.otsu_class.setCurrentText("Two classes" if two_class else "Three classes")
        fg = s.get("assign_middle_to_foreground", True)
        cp.otsu_middle.setCurrentText("Foreground" if fg else "Background")

        cp.object_fraction.setValue(float(s.get("object_fraction", 0.2)))
        cp.lower_outlier_fraction.setValue(float(s.get("lower_outlier_fraction", 0.05)))
        cp.upper_outlier_fraction.setValue(float(s.get("upper_outlier_fraction", 0.05)))
        cp.averaging_method.setCurrentText(s.get("averaging_method", "Mean"))
        cp.variance_method.setCurrentText(
            s.get("variance_method", "Standard deviation")
        )
        cp.number_of_deviations.setValue(float(s.get("number_of_deviations", 2.0)))
        cp.adaptive_window_size.setValue(int(s.get("adaptive_window_size", 50)))
        cp.fill_holes_thresholding.setChecked(
            bool(s.get("fill_holes_after_thresholding", True))
        )
        cp.fill_holes_declumping.setChecked(
            bool(s.get("fill_holes_after_declumping", True))
        )
        cp.automatic_smoothing.setChecked(bool(s.get("automatic_smoothing", True)))
        cp.smoothing_filter_size.setValue(float(s.get("smoothing_filter_size", 10.0)))
        cp.automatic_maxima_suppression.setChecked(
            bool(s.get("automatic_maxima_suppression", True))
        )
        cp.maxima_suppression_size.setValue(
            float(s.get("maxima_suppression_size", 7.0))
        )
        cp.low_res_maxima.setChecked(bool(s.get("low_res_maxima", True)))
        cp.exclude_border_objects.setChecked(
            bool(s.get("exclude_border_objects", True))
        )
        cp.invert_image.setChecked(bool(s.get("invert_image", False)))

    def _on_save(self):
        settings = self._collect_settings()
        self.settings_accepted.emit(settings)
        self.accept()
