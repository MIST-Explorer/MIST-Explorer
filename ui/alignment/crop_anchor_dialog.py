"""Interactive dialog to auto-find and approve a crop anchor.

Runs :class:`CropAnchorFinder` on the reference (decoding cycle 1) top-left patch
and the moving (protein) image, overlays the ranked candidate crop boxes on a
downscaled protein overview, and lets the user pick one and approve. On approval
it emits the reference->protein transform plus the crop size so the controller can
rotate-then-crop the full-resolution protein image.
"""

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QDialog, QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QListWidget,
                             QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

from core.crop_anchor_finder import (CropAnchorFinder, brightfield_binarize,
                                     infer_tile_size)
from core.image_utils import numpy_to_qimage

logger = logging.getLogger(__name__)

_OVERVIEW_LONG_SIDE = 900
_PREVIEW_SIDE = 260


def _to_pixmap(gray_u8: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(np.ascontiguousarray(gray_u8)))


class CropAnchorDialog(QDialog):
    """Overview + ranked-boxes approval UI for the crop anchor."""

    transform_ready = pyqtSignal(dict)

    def __init__(self, reference_img, moving_img, ref_shape, pad=1.1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Crop Anchor")
        self.resize(1100, 720)

        self.reference_img = reference_img
        self.moving_img = moving_img
        self.ref_h = int(ref_shape[0])
        self.ref_w = int(ref_shape[1])
        self.pad = float(pad)
        # Square crop, default 1.1x the reference's larger side.
        self._default_crop_px = int(round(self.pad * max(self.ref_h, self.ref_w)))

        self.candidates: list[dict] = []
        self.selected_index = -1
        self._order_ncc: list[int] = []
        self._finder = None

        # Overview of the (large) protein image, downscaled for display.
        self._ov_ds = min(
            1.0, _OVERVIEW_LONG_SIDE / float(max(moving_img.shape[:2]))
        )
        self._overview_u8 = self._make_overview(moving_img)

        self._build_ui()
        self._show_overview()
        self._show_reference_patch()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # Left: parameters + candidate list + actions.
        left = QVBoxLayout()
        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMaximumWidth(280)

        left.addWidget(QLabel("Anchor patch size (px)"))
        self.patch_size = QSpinBox()
        self.patch_size.setRange(0, 20000)
        self.patch_size.setSpecialValueText("Auto")
        self.patch_size.setSingleStep(100)
        self.patch_size.setValue(0)
        self.patch_size.valueChanged.connect(lambda: self._show_reference_patch())
        left.addWidget(self.patch_size)

        left.addWidget(QLabel("Number of candidates"))
        self.num_candidates = QSpinBox()
        self.num_candidates.setRange(1, 20)
        self.num_candidates.setValue(5)
        left.addWidget(self.num_candidates)

        left.addWidget(QLabel("ORB features"))
        self.n_features = QSpinBox()
        self.n_features.setRange(1000, 260000)
        self.n_features.setSingleStep(5000)
        self.n_features.setValue(50000)
        left.addWidget(self.n_features)

        left.addWidget(QLabel("Crop size (px, square)"))
        self.crop_size_px = QSpinBox()
        self.crop_size_px.setRange(1, 200000)
        self.crop_size_px.setSingleStep(100)
        self.crop_size_px.setValue(self._default_crop_px)
        left.addWidget(self.crop_size_px)

        self.find_button = QPushButton("Find candidates")
        self.find_button.clicked.connect(self._start_find)
        left.addWidget(self.find_button)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        left.addWidget(self.progress)

        left.addWidget(QLabel("Ranked Candidates (NCC)"))
        self.list_ncc = QListWidget()
        self.list_ncc.currentRowChanged.connect(
            lambda r: self._on_list_selected(self.list_ncc, self._order_ncc, r)
        )
        left.addWidget(self.list_ncc, stretch=1)

        self.approve_button = QPushButton("Approve && Crop")
        self.approve_button.setEnabled(False)
        self.approve_button.clicked.connect(self._approve)
        left.addWidget(self.approve_button)

        root.addWidget(left_panel)

        # Center: protein overview with overlaid candidate boxes.
        self.overview_scene = QGraphicsScene(self)
        self._overview_pixmap_item = QGraphicsPixmapItem()
        self.overview_scene.addItem(self._overview_pixmap_item)
        self.overview_view = QGraphicsView(self.overview_scene)
        self.overview_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.overview_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.overview_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._ov_zoom = 1.0
        # Install wheel event filter for zooming
        self.overview_view.viewport().installEventFilter(self)
        root.addWidget(self.overview_view, stretch=1)

        # Right: reference patch (top) and crop preview (bottom).
        right = QVBoxLayout()
        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setMaximumWidth(300)
        right.addWidget(QLabel("Reference top-left"))
        self.ref_label = QLabel()
        self.ref_label.setFixedSize(_PREVIEW_SIDE, _PREVIEW_SIDE)
        self.ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.ref_label)
        right.addWidget(QLabel("Crop preview"))
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(_PREVIEW_SIDE, _PREVIEW_SIDE)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.preview_label)
        right.addStretch(1)
        root.addWidget(right_panel)

    # -- data prep ---------------------------------------------------------
    def _make_overview(self, mov):
        small = cv2.resize(
            mov,
            (
                max(1, int(mov.shape[1] * self._ov_ds)),
                max(1, int(mov.shape[0] * self._ov_ds)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return brightfield_binarize(small)

    def _show_overview(self):
        self._overview_pixmap_item.setPixmap(_to_pixmap(self._overview_u8))
        self.overview_scene.setSceneRect(
            self._overview_pixmap_item.boundingRect()
        )
        self.overview_view.fitInView(
            self._overview_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
        )

    def _show_reference_patch(self):
        val = self.patch_size.value()
        if val <= 0:
            val = infer_tile_size(self.reference_img)
        s = min(val, self.ref_h, self.ref_w)
        patch = brightfield_binarize(self.reference_img[:s, :s])
        pix = _to_pixmap(patch).scaled(
            _PREVIEW_SIDE,
            _PREVIEW_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.ref_label.setPixmap(pix)

    # -- finder ------------------------------------------------------------
    def _start_find(self):
        if self._finder is not None and self._finder.isRunning():
            return
        self._show_reference_patch()
        self.find_button.setEnabled(False)
        self.approve_button.setEnabled(False)
        self._clear_lists()
        self.candidates = []
        self.selected_index = -1
        self._clear_boxes()
        self.progress.setValue(0)

        self._finder = CropAnchorFinder(
            self.reference_img,
            self.moving_img,
            patch_size=self.patch_size.value(),
            num_candidates=self.num_candidates.value(),
            n_features=self.n_features.value(),
        )
        self._finder.progress.connect(lambda p, _msg: self.progress.setValue(p))
        self._finder.candidates_ready.connect(self._on_candidates_ready)
        self._finder.error.connect(self._on_error)
        self._finder.finished.connect(lambda: self.find_button.setEnabled(True))
        self._finder.start()

    def _clear_lists(self):
        self.list_ncc.blockSignals(True)
        self.list_ncc.clear()
        self.list_ncc.blockSignals(False)

    def _on_error(self, msg):
        self.find_button.setEnabled(True)
        self.progress.setValue(0)
        logger.error("Crop anchor search failed: %s", msg)
        self.list_ncc.addItem(f"Error: {msg}")

    def _on_candidates_ready(self, candidates):
        self.candidates = candidates or []
        self._draw_boxes()
        
        self._order_ncc = self._rank_by(lambda c: c["score"])
        self._populate_list(self.list_ncc, self._order_ncc)
        
        if self.candidates:
            self.list_ncc.setCurrentRow(0)  # triggers selection + preview
            self.approve_button.setEnabled(True)

    def _rank_by(self, key):
        return sorted(
            range(len(self.candidates)),
            key=lambda i: key(self.candidates[i]),
            reverse=True,
        )

    def _populate_list(self, widget, order):
        widget.blockSignals(True)
        widget.clear()
        for rank, idx in enumerate(order):
            c = self.candidates[idx]
            widget.addItem(
                f"#{rank + 1}  ncc={c['score']:.3f}  inl={c.get('inliers', 0)}  "
                f"ratio={c.get('inlier_ratio', 0):.2f}  blob={c.get('blob_fraction', 0):.2f}  "
                f"resid={c.get('residual', 0):.1f}  ang={c['angle']:.1f}°"
            )
        widget.blockSignals(False)

    # -- overlay boxes -----------------------------------------------------
    def _crop_size(self):
        size = float(self.crop_size_px.value())
        return size, size

    def _box_polygon(self, cand) -> QPolygonF:
        """Crop rectangle (reference rect under T) projected into overview coords."""
        T = np.asarray(cand["T"], dtype=np.float64).reshape(2, 3)
        cw, ch = self._crop_size()
        corners = np.array(
            [[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64
        )
        poly = QPolygonF()
        for cx, cy in corners:
            px = (T[0, 0] * cx + T[0, 1] * cy + T[0, 2]) * self._ov_ds
            py = (T[1, 0] * cx + T[1, 1] * cy + T[1, 2]) * self._ov_ds
            poly.append(QPointF(px, py))
        return poly

    def _clear_boxes(self):
        self._box_items = getattr(self, "_box_items", [])
        for item in self._box_items:
            self.overview_scene.removeItem(item)
        self._box_items = []

    def _draw_boxes(self):
        self._clear_boxes()
        for i, cand in enumerate(self.candidates):
            selected = i == self.selected_index
            pen = QPen(QColor("#ff3b3b") if selected else QColor("#ffd23b"))
            pen.setWidth(0)
            poly_item = self.overview_scene.addPolygon(
                self._box_polygon(cand), pen, QBrush(Qt.BrushStyle.NoBrush)
            )
            poly_item.setZValue(10)
            self._box_items.append(poly_item)
            text = self.overview_scene.addText(str(i + 1))
            text.setDefaultTextColor(pen.color())
            first = self._box_polygon(cand).first()
            text.setPos(first.x(), first.y())
            text.setZValue(11)
            self._box_items.append(text)

    # -- selection / preview ----------------------------------------------
    def _on_list_selected(self, active_widget, order, row):
        if row < 0 or row >= len(order):
            return
        self.selected_index = order[row]
        self._draw_boxes()
        self._show_preview(self.candidates[self.selected_index])

    def _show_preview(self, cand):
        """Rotate-then-crop preview built from the downscaled overview (fast)."""
        T = np.asarray(cand["T"], dtype=np.float64).reshape(2, 3)
        cw, ch = self._crop_size()
        # Work in overview coords: scale anchor by _ov_ds (rotation is scale-free).
        A = T[:, :2]
        t = T[:, 2] * self._ov_ds
        inv_a = np.linalg.inv(A)
        M = np.hstack([inv_a, (-inv_a @ t).reshape(2, 1)]).astype(np.float32)
        out_w = max(1, int(cw * self._ov_ds))
        out_h = max(1, int(ch * self._ov_ds))
        warped = cv2.warpAffine(self._overview_u8, M, (out_w, out_h))
        pix = _to_pixmap(warped).scaled(
            _PREVIEW_SIDE,
            _PREVIEW_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    # -- approve -----------------------------------------------------------
    def _approve(self):
        if self.selected_index < 0 or self.selected_index >= len(self.candidates):
            return
        cand = self.candidates[self.selected_index]
        cw, ch = self._crop_size()
        self.transform_ready.emit(
            {
                "T": np.asarray(cand["T"], dtype=np.float64).reshape(2, 3),
                "crop_w": int(round(cw)),
                "crop_h": int(round(ch)),
                "angle": cand["angle"],
                "anchor": cand["anchor"],
            }
        )
        self.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._overview_pixmap_item.pixmap().width():
            self.overview_view.fitInView(
                self._overview_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
            )
            if self._ov_zoom != 1.0:
                self.overview_view.scale(self._ov_zoom, self._ov_zoom)

    def eventFilter(self, source, event):
        if source == self.overview_view.viewport() and event.type() == event.Type.Wheel:
            # Zoom logic for the overview view
            zooming_out = event.angleDelta().y() < 0
            if self._ov_zoom <= 0.1 and zooming_out:
                return True
            if self._ov_zoom >= 100 and not zooming_out:
                return True

            zoom_factor = 0.85 if zooming_out else 1.15
            self._ov_zoom *= zoom_factor
            self.overview_view.scale(zoom_factor, zoom_factor)
            return True
        return super().eventFilter(source, event)
