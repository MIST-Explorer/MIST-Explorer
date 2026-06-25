import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QToolTip,
)

from core.dataframe_utils import get_marker_columns


def calculate_weighted_centroids(data, proteins, signal_threshold, custom_thresholds):
    """Calculate weighted centroid for each protein."""
    centroids = {}
    for protein in proteins:
        threshold = custom_thresholds.get(protein, signal_threshold)
        protein_cells = data[data[protein] >= threshold]
        if len(protein_cells) > 0:
            weights = protein_cells[protein]
            avg_x = np.average(protein_cells["Global X"], weights=weights)
            avg_y = np.average(protein_cells["Global Y"], weights=weights)
            centroids[protein] = (avg_x, avg_y)
    return centroids


def compute_distance_matrix(centroids, proteins):
    """Compute pairwise Euclidean distance matrix."""
    n = len(proteins)
    distances = np.full((n, n), np.nan)
    for i, p1 in enumerate(proteins):
        for j, p2 in enumerate(proteins):
            if p1 in centroids and p2 in centroids:
                x1, y1 = centroids[p1]
                x2, y2 = centroids[p2]
                distances[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distances


class HeatmapWindow(QMainWindow):
    def __init__(
        self,
        data,
        signal_threshold=1000,
        custom_thresholds=None,
        x_min=None,
        y_min=None,
        x_max=None,
        y_max=None,
        parent=None,
    ):
        super().__init__(parent)

        if custom_thresholds is None:
            custom_thresholds = {}

        # Optional ROI filtering
        self._data = data
        if all(v is not None for v in (x_min, y_min, x_max, y_max)):
            self._data = data[
                (data["Global X"] >= x_min)
                & (data["Global X"] <= x_max)
                & (data["Global Y"] >= y_min)
                & (data["Global Y"] <= y_max)
            ]

        self._proteins = list(get_marker_columns(self._data))

        # Track which per-protein spinboxes the user has manually edited
        self._manually_edited = set()
        self._grid_cols = 0  # tracks current column count; 0 forces initial layout

        # Debounce timer for live updates
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._update_heatmap)

        # --- Build UI ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Default threshold row
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("Default Threshold:"))
        self._default_spin = QSpinBox()
        self._default_spin.setRange(0, 50000)
        self._default_spin.setSingleStep(100)
        self._default_spin.setValue(signal_threshold)
        self._default_spin.valueChanged.connect(self._on_default_threshold_changed)
        default_row.addWidget(self._default_spin)
        default_row.addStretch()
        main_layout.addLayout(default_row)

        # Collapsible per-protein section
        self._toggle_btn = QToolButton()
        self._toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setText("Per-Protein Thresholds")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.toggled.connect(self._toggle_protein_section)
        main_layout.addWidget(self._toggle_btn)

        # Scrollable grid of per-protein spinboxes
        self._protein_group = QGroupBox()
        self._protein_group.setFlat(True)
        self._protein_group.setVisible(False)
        grid = QGridLayout(self._protein_group)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        # Build widgets once; _relayout_protein_grid() places them
        self._protein_spins = {}
        self._protein_labels = {}
        for protein in self._proteins:
            label = QLabel(protein)
            spin = QSpinBox()
            spin.setRange(0, 50000)
            spin.setSingleStep(100)
            initial = custom_thresholds.get(protein, signal_threshold)
            spin.setValue(initial)
            if protein in custom_thresholds:
                self._manually_edited.add(protein)
            spin.valueChanged.connect(
                lambda _v, p=protein: self._on_protein_threshold_changed(p)
            )
            self._protein_spins[protein] = spin
            self._protein_labels[protein] = label

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._protein_group)
        self._scroll_area = scroll
        scroll.setVisible(False)
        scroll.viewport().installEventFilter(self)

        # Matplotlib figure + canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        self._canvas = FigureCanvas(self.figure)
        self._canvas.mpl_connect("motion_notify_event", self.on_hover)
        self._canvas.mpl_connect("figure_leave_event", lambda e: QToolTip.hideText())

        # Splitter lets the user drag to resize the threshold list vs the heatmap
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(scroll)
        self._splitter.addWidget(self._canvas)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([0, 1])  # start collapsed
        main_layout.addWidget(self._splitter, stretch=1)

        # Initial draw
        self._update_heatmap()

    # --- Signal handlers ---

    def _on_default_threshold_changed(self, value):
        # Propagate to per-protein spinboxes that haven't been manually customized
        for protein, spin in self._protein_spins.items():
            if protein not in self._manually_edited:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._debounce.start()

    def _on_protein_threshold_changed(self, protein):
        self._manually_edited.add(protein)
        self._debounce.start()

    _PROTEIN_PANEL_DEFAULT_HEIGHT = 160

    def _toggle_protein_section(self, checked):
        arrow = Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        self._toggle_btn.setArrowType(arrow)
        self._scroll_area.setVisible(checked)
        if checked:
            total = self._splitter.height()
            panel = min(self._PROTEIN_PANEL_DEFAULT_HEIGHT, total // 2)
            self._splitter.setSizes([panel, total - panel])
            self._relayout_protein_grid()
        else:
            self._splitter.setSizes([0, self._splitter.height()])

    # --- Responsive protein grid ---

    _PAIR_MIN_WIDTH = 200  # px; minimum width allocated per label+spinbox pair

    def eventFilter(self, obj, event):
        if obj is self._scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self._relayout_protein_grid()
        return super().eventFilter(obj, event)

    def _relayout_protein_grid(self):
        vp_w = self._scroll_area.viewport().width()
        cols = max(1, vp_w // self._PAIR_MIN_WIDTH) if vp_w > 0 else 2
        if cols == self._grid_cols:
            return
        self._grid_cols = cols

        grid = self._protein_group.layout()
        while grid.count():
            grid.takeAt(0)
        for c in range(grid.columnCount()):
            grid.setColumnStretch(c, 0)

        # Layout per row: [label | spin | <stretch> | label | spin | <stretch> | ...]
        # Every 3rd column is a stretch spacer so label↔spin stay tight.
        for c in range(cols):
            grid.setColumnStretch(c * 3 + 2, 1)

        for idx, protein in enumerate(self._proteins):
            row, col = divmod(idx, cols)
            grid.addWidget(self._protein_labels[protein], row, col * 3)
            grid.addWidget(self._protein_spins[protein], row, col * 3 + 1)

    # --- Heatmap computation + render ---

    def _update_heatmap(self):
        signal_threshold = self._default_spin.value()
        custom_thresholds = {}
        for protein, spin in self._protein_spins.items():
            val = spin.value()
            if val != signal_threshold:
                custom_thresholds[protein] = val

        centroids = calculate_weighted_centroids(
            self._data, self._proteins, signal_threshold, custom_thresholds
        )
        distances = compute_distance_matrix(centroids, self._proteins)
        # Normalize
        if np.all(np.isnan(distances)):
            distances_normalized = np.full_like(distances, np.nan)
        elif np.nanmax(distances) == np.nanmin(distances):
            distances_normalized = np.zeros_like(distances)
        else:
            distances_normalized = (distances - np.nanmin(distances)) / (
                np.nanmax(distances) - np.nanmin(distances)
            )

        median_distance = np.nanmedian(distances_normalized)
        vmax_value = max(2 * median_distance, 0.01)

        self.distances = distances
        self.distances_normalized = distances_normalized

        # Redraw on existing axes
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        sns.heatmap(
            distances_normalized,
            cmap="coolwarm_r",
            xticklabels=self._proteins,
            yticklabels=self._proteins,
            vmin=0,
            vmax=vmax_value,
            cbar_kws={"label": "Normalized Distance"},
            ax=self.ax,
        )
        self.ax.set_xticklabels(
            self.ax.get_xticklabels(),
            rotation=90,
            fontsize=10,
        )
        self.ax.set_yticklabels(
            self.ax.get_yticklabels(),
            fontsize=10,
        )
        self.ax.set_title(
            "Normalized Protein Spatial Distance Heatmap",
            fontsize=13,
        )
        # apply_matplotlib_theme(self.figure, self.ax)
        self._canvas.draw_idle()

    def on_hover(self, event):
        if not hasattr(self, "distances") or self.distances is None or event.inaxes != self.ax:
            QToolTip.hideText()
            return

        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            p = len(self._proteins)
            col_idx = int(np.floor(x))
            row_idx = int(np.floor(y))
            if 0 <= col_idx < p and 0 <= row_idx < p:
                p1 = self._proteins[row_idx]
                p2 = self._proteins[col_idx]
                raw_dist = self.distances[row_idx, col_idx]
                norm_dist = self.distances_normalized[row_idx, col_idx]

                raw_dist_str = f"{raw_dist:.2f} µm" if not np.isnan(raw_dist) else "N/A"
                norm_dist_str = f"{norm_dist:.4f}" if not np.isnan(norm_dist) else "N/A"

                tooltip_text = (
                    f"<b>Protein A:</b> {p1}<br/>"
                    f"<b>Protein B:</b> {p2}<br/>"
                    f"<b>Distance:</b> {raw_dist_str}<br/>"
                    f"<b>Normalized Dist:</b> {norm_dist_str}"
                )
                QToolTip.showText(QCursor.pos(), tooltip_text, self._canvas)
                return

        QToolTip.hideText()
