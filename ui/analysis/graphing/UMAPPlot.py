import logging
import os
import sys
import traceback

os.environ["QT_API"] = "PyQt6"

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import scanpy as sc
import tifffile
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.analysis.graphing.UMAPDataModel import DataModel
from ui.analysis.graphing.UMAPLocale import Locale
from ui.analysis.graphing.UMAPWorkers import AnalysisWorker, ClusteringWorker
from utils import resource_path

matplotlib.use("QtAgg")

# --- 1. Debugging Setup ---

logger = logging.getLogger("UMAP_App")


def excepthook(exc_type, exc_value, exc_tb):
    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
    app = QApplication.instance()
    if app:
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("An unexpected error occurred.")
        msg.setInformativeText(str(exc_value))
        msg.setDetailedText(tb_str)
        msg.exec()


# ==========================================
# 0. AESTHETICS ENGINE
# ==========================================


class ThemeManager:
    """Legacy bridge — delegates to the app-wide ThemeManager in ui.theme."""

    from ui.theme import THEMES
    from ui.theme import ThemeManager as _AppTheme

    @classmethod
    def get_current(cls):
        try:
            return cls._AppTheme.instance().get_current()
        except RuntimeError:
            return cls.THEMES["DARK"]

    @classmethod
    def toggle(cls):
        try:
            cls._AppTheme.instance().toggle()
        except RuntimeError:
            pass
        return cls.get_current()

    @property
    def current_mode(self):
        try:
            return self._AppTheme.instance().current_mode
        except RuntimeError:
            return "DARK"

    # Class-level access for code that reads ThemeManager.current_mode
    @classmethod
    def _get_current_mode(cls):
        try:
            return cls._AppTheme.instance().current_mode
        except RuntimeError:
            return "DARK"

    @classmethod
    def get_stylesheet(cls):
        c = cls.get_current()

        # Resolve icon path (use resource_path for PyInstaller compatibility)
        icons_dir = resource_path(
            os.path.join("assets", "icons")
        ).replace("\\", "/")

        # Select icon based on theme
        tick_icon = (
            "checkbox_tick_dark.svg"
            if cls._get_current_mode() == "DARK"
            else "checkbox_tick_light.svg"
        )
        tick_path = f"{icons_dir}/{tick_icon}"

        return f"""
        /* GLOBAL RESET */
        QWidget {{
            background-color: {c["bg_dark"]};
            color: {c["text_main"]};
            font-family: "Consolas", "Menlo", "Monaco", monospace;
            font-size: 10pt;
            selection-background-color: {c["accent"]};
            selection-color: {c["bg_dark"]};
        }}

        /* TABS */
        QTabWidget::pane {{
            border: 2px solid {c["grid"]};
            background: {c["bg_dark"]};
            margin-top: -1px; 
        }}

        QTabBar::tab {{
            background: {c["bg_panel"]};
            color: {c["text_main"]};
            padding: 8px 20px;
            border: 1px solid {c["grid"]};
            border-bottom: none;
            margin-right: 2px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        QTabBar::tab:selected {{
            background: {c["bg_dark"]};
            color: {c["accent"]};
            border: 2px solid {c["accent"]};
            border-bottom: 2px solid {c["bg_dark"]};
        }}

        /* INPUTS */
        QLineEdit, QComboBox, QSpinBox {{
            background-color: {c["bg_panel"]};
            border: 1px solid {c["grid"]};
            padding: 4px;
            color: {c["accent"]}; /* Distinctive colored text input */
            font-weight: bold;
        }}

        QLineEdit:focus, QComboBox:focus {{
            border: 2px solid {c["accent"]};
            background-color: {c["bg_dark"]};
        }}

        /* LISTS */
        QListWidget {{
            background-color: {c["bg_panel"]};
            border: 2px solid {c["grid"]};
            outline: none;
        }}
        QListWidget::item {{
            padding: 5px;
            border-bottom: 1px solid {c["grid"]};
            border-left: 4px solid transparent;
            color: {c["text_main"]};
        }}
        QListWidget::item:selected {{
            background-color: {c["bg_dark"]};
            color: {c["text_main"]};
            border-left: 4px solid {c["accent"]};
        }}

        /* BUTTONS */
        QPushButton {{
            background-color: {c["bg_panel"]};
            border: 1px solid {c["accent_dim"]};
            color: {c["accent"]};
            padding: 8px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        QPushButton:hover {{
            background-color: {c["accent"]};
            color: {c["bg_dark"] if cls._get_current_mode() == "DARK" else "#ffffff"};
            border: 1px solid {c["accent"]};
        }}
        QPushButton:pressed {{
            background-color: {c["alert"]};
            color: #ffffff;
            border: 1px solid {c["alert"]};
        }}

        /* GROUPBOX */
        QGroupBox {{
            border: 1px solid {c["grid"]};
            margin-top: 1.5em;
            padding-top: 10px;
            font-weight: bold;
            color: {c["accent_dim"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 10px;
            background-color: {c["bg_dark"]};
            color: {c["text_main"]};
            font-size: 11pt;
        }}

        /* SPLITTER */
        QSplitter::handle {{
            background-color: {c["grid"]};
        }}
        QSplitter::handle:hover {{
            background-color: {c["accent"]};
        }}

        /* SLIDERS */
        QSlider::groove:horizontal {{
            border: 1px solid {c["grid"]};
            height: 6px;
            background: {c["bg_panel"]};
            margin: 2px 0;
        }}
        QSlider::handle:horizontal {{
            background: {c["accent"]};
            border: 1px solid {c["accent"]};
            width: 14px;
            height: 14px;
            margin: -6px 0;
            border-radius: 0px; /* Sharp edges */
        }}

        /* SCROLLBARS */
        QScrollBar:vertical {{
            border: none;
            background: {c["bg_panel"]};
            width: 12px;
        }}
        QScrollBar::handle:vertical {{
            background: {c["accent_dim"]};
            min-height: 20px;
        }}

        /* CHECKBOX INDICATORS */
        QCheckBox::indicator, QListWidget::indicator {{
            width: 14px;
            height: 14px;
            background-color: {c["bg_panel"]};
            border: 1px solid {c["text_main"]};
            border-radius: 3px;
        }}
        QCheckBox::indicator:checked, QListWidget::indicator:checked {{
            background-color: {c["accent"]};
            border: 1px solid {c["accent"]};
            image: url({tick_path});
        }}
        QCheckBox::indicator:hover, QListWidget::indicator:hover {{
            border: 1px solid {c["accent"]};
        }}
        
        /* SPECIFIC OVERRIDES */
        QLabel#h1 {{
            font-size: 14pt;
            color: {c["accent"]};
            font-weight: 900;
        }}
        
        /* MAIN ACTION BUTTON OVERRIDE */
        QPushButton#MainRunBtn {{
            background-color: {c["bg_panel"]};
            border: 3px solid {c["accent"]};
            color: {c["accent"]};
            font-size: 11pt;
            letter-spacing: 2px;
        }}
        QPushButton#MainRunBtn:hover {{
            background-color: {c["accent"]};
            color: {c["bg_dark"] if cls._get_current_mode() == "DARK" else "#ffffff"};
        }}
        """


def apply_matplotlib_theme(figure, ax=None):
    """Force Matplotlib to respect the current theme.

    Delegates to ui.theme.apply_matplotlib_theme which uses the central
    ThemeManager.  Falls back to local colors if the central manager
    is not yet initialised.
    """
    try:
        from ui.theme import apply_matplotlib_theme as _apply

        _apply(figure, ax)
    except RuntimeError:
        c = ThemeManager.get_current()
        figure.patch.set_facecolor(c["bg_dark"])
        if ax:
            ax.set_facecolor(c["bg_dark"])
            ax.tick_params(colors=c["text_main"], which="both")
            for spine in ax.spines.values():
                spine.set_edgecolor(c["accent_dim"])
                spine.set_linewidth(1.5)
            ax.xaxis.label.set_color(c["accent"])
            ax.yaxis.label.set_color(c["accent"])
            ax.title.set_color(c["accent"])


def to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


sys.excepthook = excepthook


# ==========================================
# 1. UI COMPONENTS
# ==========================================


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        self.base_title = title
        self.is_collapsed = False
        super().__init__(f"▼ {title}", parent)
        self.setMouseTracking(True)

    def _get_display_title(self):
        arrow = "▶" if self.is_collapsed else "▼"
        return f"{arrow} {self.base_title}"

    def mouseMoveEvent(self, event):
        if event.pos().y() < 30:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.pos().y() < 30:
            self.toggle_collapse()
            event.accept()
        else:
            super().mousePressEvent(event)

    def toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        super().setTitle(self._get_display_title())
        layout = self.layout()
        if layout:
            self._set_layout_visible(layout, not self.is_collapsed)
        
        if self.is_collapsed:
            self.setMaximumHeight(30)
        else:
            self.setMaximumHeight(16777215)

    def _set_layout_visible(self, layout, visible):
        if not layout:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w:
                w.setVisible(visible)
            elif item.layout():
                self._set_layout_visible(item.layout(), visible)


class ThresholdDialog(QDialog):
    def __init__(self, model, features, parent=None):
        super().__init__(parent)
        self.model = model
        self.features = features
        self.original_threshold_mode = self.model.threshold_mode
        self.original_thresholds = self.model.thresholds.copy()
        self.setWindowTitle("Adjust Protein Thresholds")
        self.resize(600, 680)
        self.setStyleSheet(ThemeManager.get_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        label_info = QLabel(
            "Set the threshold below which the value of each protein is set to 0.\n"
            "If a cell has all zeros across selected proteins post-threshold, it will be filtered from UMAP/Clustering."
        )
        label_info.setWordWrap(True)
        label_info.setStyleSheet("color: #bbbbbb; font-size: 10pt; margin-bottom: 5px;")
        layout.addWidget(label_info)

        # Checkbox for pre-normalization thresholding mode
        self.chk_pre_norm = QCheckBox("Apply thresholds to raw counts (pre-normalization)")
        self.chk_pre_norm.setChecked(self.model.threshold_mode == "pre")
        self.chk_pre_norm.setStyleSheet("font-weight: bold; font-size: 10pt; margin-bottom: 10px;")
        self.chk_pre_norm.toggled.connect(self.on_mode_toggled)
        layout.addWidget(self.chk_pre_norm)

        # Scroll area for proteins
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: 1px solid {ThemeManager.get_current()['grid']};")
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)

        self.controls_data = {}

        # Get the normalized values from the model
        var_names = list(self.model.adata.var_names)
        is_pre = self.model.threshold_mode == "pre"

        # Custom stylesheets to match opacity sliders
        tc = ThemeManager.get_current()
        _edit_style = (
            f"QLineEdit {{"
            f"  font-size: 11px; font-weight: bold;"
            f"  color: {tc['accent']};"
            f"  background: {tc['bg_panel']};"
            f"  border: 1px solid {tc['grid']};"
            f"  border-radius: 2px; padding: 2px 4px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border: 1px solid {tc['accent']};"
            f"  background-color: {tc['bg_dark']};"
            f"}}"
        )

        for idx, feat in enumerate(features):
            feat_idx = var_names.index(feat)
            
            # Select reference values depending on threshold mode
            if is_pre:
                feat_values = self.model.raw_pre_threshold[:, feat_idx]
            else:
                feat_values = self.model.normalized_pre_threshold[:, feat_idx]
                
            min_val = float(np.min(feat_values))
            max_val = float(np.max(feat_values))

            # Pre-compute statistics
            p50 = float(np.percentile(feat_values, 50))
            p90 = float(np.percentile(feat_values, 90))
            p95 = float(np.percentile(feat_values, 95))

            nz_mask = feat_values > 0.0
            nz_pct = float(np.mean(nz_mask) * 100)
            nz_vals = feat_values[nz_mask]
            mean_nz = float(np.mean(nz_vals)) if len(nz_vals) > 0 else 0.0
            std_nz = float(np.std(nz_vals)) if len(nz_vals) > 0 else 0.0

            # Create widgets
            lbl_name = QLabel(feat)
            lbl_name.setStyleSheet("font-weight: bold; font-size: 10pt; min-width: 80px;")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)

            edit = QLineEdit()
            edit.setFixedWidth(60)
            edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            edit.setStyleSheet(_edit_style)

            lbl_subtext = QLabel()
            lbl_subtext.setStyleSheet("color: #888888; font-size: 9pt; margin-top: 2px;")
            lbl_subtext.setWordWrap(True)

            # Initialize values
            curr_thresh = self.model.thresholds.get(feat, 1.0)
            curr_thresh = max(min_val, min(max_val, curr_thresh))

            # Store control widgets and data
            self.controls_data[feat] = {
                "min": min_val,
                "max": max_val,
                "values": feat_values,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "nz_pct": nz_pct,
                "mean_nz": mean_nz,
                "std_nz": std_nz,
                "slider": slider,
                "edit": edit,
                "subtext": lbl_subtext
            }

            # Connect slider and text edits
            slider.valueChanged.connect(lambda v, f=feat: self.on_slider_changed(f, v))
            edit.editingFinished.connect(lambda f=feat: self.on_edit_finished(f))

            # Set the initial UI values
            self.set_ui_value(feat, curr_thresh)

            # Sub-grid layout for aligning Name, Slider, and TextField
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setColumnStretch(1, 1)
            grid.addWidget(lbl_name, 0, 0)
            grid.addWidget(slider, 0, 1)
            grid.addWidget(edit, 0, 2)
            grid.addWidget(lbl_subtext, 1, 0, 1, 3)

            scroll_layout.addLayout(grid)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_reset = QPushButton("Reset All to 1.0")
        btn_reset.setAutoDefault(False)
        btn_reset.setDefault(False)
        btn_reset.clicked.connect(self.reset_to_default)
        btn_layout.addWidget(btn_reset)

        btn_layout.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.setDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Save")
        btn_save.setAutoDefault(False)
        btn_save.setDefault(False)
        btn_save.clicked.connect(self.save_thresholds)
        btn_save.setStyleSheet(f"border: 2px solid {ThemeManager.get_current()['accent']}; font-weight: bold;")
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def on_mode_toggled(self, checked):
        # Update mode
        new_mode = "pre" if checked else "post"
        self.model.threshold_mode = new_mode

        # Parent is UMAPControls, we can find normalization from combo_norm
        parent_controls = self.parent()
        if parent_controls and hasattr(parent_controls, "combo_norm"):
            normalization = parent_controls.combo_norm.currentText()
        else:
            normalization = self.model.normalization_method

        try:
            self.model.reprocess_data(normalization, self.features)
        except Exception as e:
            logger.error(f"Failed to reprocess data on mode toggle: {e}")

        self.refresh_ui()

    def refresh_ui(self):
        var_names = list(self.model.adata.var_names)
        is_pre = self.model.threshold_mode == "pre"

        for feat in self.features:
            feat_idx = var_names.index(feat)

            if is_pre:
                feat_values = self.model.raw_pre_threshold[:, feat_idx]
            else:
                feat_values = self.model.normalized_pre_threshold[:, feat_idx]

            min_val = float(np.min(feat_values))
            max_val = float(np.max(feat_values))

            # Recalculate stats
            p50 = float(np.percentile(feat_values, 50))
            p90 = float(np.percentile(feat_values, 90))
            p95 = float(np.percentile(feat_values, 95))

            nz_mask = feat_values > 0.0
            nz_pct = float(np.mean(nz_mask) * 100)
            nz_vals = feat_values[nz_mask]
            mean_nz = float(np.mean(nz_vals)) if len(nz_vals) > 0 else 0.0
            std_nz = float(np.std(nz_vals)) if len(nz_vals) > 0 else 0.0

            curr_thresh = self.model.thresholds.get(feat, 1.0)
            curr_thresh = max(min_val, min(max_val, curr_thresh))

            self.controls_data[feat].update({
                "min": min_val,
                "max": max_val,
                "values": feat_values,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "nz_pct": nz_pct,
                "mean_nz": mean_nz,
                "std_nz": std_nz
            })

            # Update ui value
            self.set_ui_value(feat, curr_thresh)

    def set_ui_value(self, feat, val):
        data = self.controls_data[feat]
        min_val = data["min"]
        max_val = data["max"]

        data["slider"].blockSignals(True)
        data["edit"].blockSignals(True)

        data["edit"].setText(f"{val:.3f}")

        if max_val > min_val:
            slider_val = int((val - min_val) / (max_val - min_val) * 1000)
            data["slider"].setValue(slider_val)
        else:
            data["slider"].setValue(0)

        data["slider"].blockSignals(False)
        data["edit"].blockSignals(False)

        self.update_subtext(feat, val)

    def on_slider_changed(self, feat, slider_val):
        data = self.controls_data[feat]
        min_val = data["min"]
        max_val = data["max"]

        if max_val > min_val:
            val = min_val + (slider_val / 1000.0) * (max_val - min_val)
        else:
            val = min_val

        data["edit"].blockSignals(True)
        data["edit"].setText(f"{val:.3f}")
        data["edit"].blockSignals(False)

        self.update_subtext(feat, val)

    def on_edit_finished(self, feat):
        data = self.controls_data[feat]
        min_val = data["min"]
        max_val = data["max"]

        try:
            val = float(data["edit"].text())
        except ValueError:
            slider_val = data["slider"].value()
            if max_val > min_val:
                val = min_val + (slider_val / 1000.0) * (max_val - min_val)
            else:
                val = min_val

        val = max(min_val, min(max_val, val))
        self.set_ui_value(feat, val)

    def update_subtext(self, feat, val):
        data = self.controls_data[feat]
        feat_values = data["values"]

        # Calculate dynamic % of cells below threshold
        pct_below = float(np.mean(feat_values < val) * 100)

        subtext = (
            f"Percentiles: 50th={data['p50']:.2f}, 90th={data['p90']:.2f}, 95th={data['p95']:.2f} | "
            f"Non-zero Fraction: {data['nz_pct']:.1f}% | "
            f"Mean±SD (Non-zero): {data['mean_nz']:.2f}±{data['std_nz']:.2f}\n"
            f"Cells below threshold: {pct_below:.1f}%"
        )
        data["subtext"].setText(subtext)

    def reset_to_default(self):
        for feat in self.features:
            min_val = self.controls_data[feat]["min"]
            max_val = self.controls_data[feat]["max"]
            val = max(min_val, min(max_val, 1.0))
            self.set_ui_value(feat, val)

    def save_thresholds(self):
        for feat in self.features:
            try:
                val = float(self.controls_data[feat]["edit"].text())
            except ValueError:
                min_val = self.controls_data[feat]["min"]
                max_val = self.controls_data[feat]["max"]
                slider_val = self.controls_data[feat]["slider"].value()
                val = min_val + (slider_val / 1000.0) * (max_val - min_val)

            min_val = self.controls_data[feat]["min"]
            max_val = self.controls_data[feat]["max"]
            val = max(min_val, min(max_val, val))
            self.model.thresholds[feat] = val
        self.accept()

    def reject(self):
        # Restore original settings
        self.model.threshold_mode = self.original_threshold_mode
        self.model.thresholds = self.original_thresholds.copy()

        # Reprocess data back to original state
        parent_controls = self.parent()
        if parent_controls and hasattr(parent_controls, "combo_norm"):
            normalization = parent_controls.combo_norm.currentText()
        else:
            normalization = self.model.normalization_method

        try:
            self.model.reprocess_data(normalization, self.features)
        except Exception as e:
            logger.error(f"Failed to restore data on cancel: {e}")

        super().reject()


class UMAPControls(QWidget):
    apply_data_signal = pyqtSignal(str, list)
    run_signal = pyqtSignal(dict)
    resolution_change_signal = pyqtSignal(float)
    recalc_pca_signal = pyqtSignal()

    def __init__(self, model, max_pca_components=50, all_features=[]):
        super().__init__()
        self.model = model
        self.res_debounce_timer = QTimer()
        self.res_debounce_timer.setSingleShot(True)
        self.res_debounce_timer.setInterval(400)
        self.res_debounce_timer.timeout.connect(self.emit_resolution_change)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # === 0. Data Setup ===
        data_group = CollapsibleGroupBox(Locale.get("GRP_DATA"))
        data_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel(Locale.get("LBL_NORM")))
        self.combo_norm = QComboBox()
        self.combo_norm.addItems(
            self.model.get_normalization_methods()
            if hasattr(self.model, "get_normalization_methods")
            else ["Log1p", "Z-Score"]
        )
        if hasattr(self.model, "normalization_method"):
            self.combo_norm.setCurrentText(self.model.normalization_method)
        # on change emit
        self.combo_norm.currentIndexChanged.connect(self.emit_apply_data)
        data_layout.addWidget(self.combo_norm)

        input_row = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText(Locale.get("PLACEHOLDER_SEARCH"))
        self.search_bar.textChanged.connect(self.filter_feature_list)
        input_row.addWidget(self.search_bar)
        data_layout.addLayout(input_row)

        btn_row = QHBoxLayout()
        btn_all = QPushButton(Locale.get("BTN_ALL"))
        btn_all.clicked.connect(self.select_all)
        btn_none = QPushButton(Locale.get("BTN_CLR"))
        btn_none.clicked.connect(self.select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        data_layout.addLayout(btn_row)

        self.feature_list = QListWidget()
        self.feature_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.feature_list.setMinimumHeight(200)
        self.populate_features(all_features)
        data_layout.addWidget(self.feature_list)
        self.feature_list.itemChanged.connect(self.emit_apply_data)

        self.btn_adjust_thresholds = QPushButton("Adjust Thresholds")
        self.btn_adjust_thresholds.clicked.connect(self.show_threshold_dialog)
        data_layout.addWidget(self.btn_adjust_thresholds)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group, 1)

        # === 1. PCA ===
        pca_group = CollapsibleGroupBox(Locale.get("GRP_PCA"))
        pca_layout = QVBoxLayout()
        self.lbl_pca = QLabel("")
        self.slider_pca = QSlider(Qt.Orientation.Horizontal)
        self.slider_pca.setRange(2, max_pca_components)
        self.slider_pca.setValue(5)
        self.slider_pca.valueChanged.connect(self.update_pca_label)
        pca_layout.addWidget(self.lbl_pca)
        pca_layout.addWidget(self.slider_pca)
        
        self.btn_recalc_pca = QPushButton(Locale.get("BTN_RECALC_PCA"))
        self.btn_recalc_pca.clicked.connect(self.recalc_pca_signal.emit)
        pca_layout.addWidget(self.btn_recalc_pca)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group)
        self.update_pca_label()

        # === 2. UMAP ===
        umap_group = CollapsibleGroupBox(Locale.get("GRP_UMAP"))
        umap_layout = QVBoxLayout()
        self.lbl_neighbors = QLabel(Locale.get("LBL_NEIGHBORS", 15))
        self.slider_neighbors = QSlider(Qt.Orientation.Horizontal)
        self.slider_neighbors.setRange(2, 200)
        self.slider_neighbors.setValue(15)
        self.slider_neighbors.valueChanged.connect(
            lambda v: self.lbl_neighbors.setText(Locale.get("LBL_NEIGHBORS", v))
        )
        umap_layout.addWidget(self.lbl_neighbors)
        umap_layout.addWidget(self.slider_neighbors)

        self.lbl_dist = QLabel(Locale.get("LBL_MINDIST", 0.10))
        self.slider_dist = QSlider(Qt.Orientation.Horizontal)
        self.slider_dist.setRange(1, 99)
        self.slider_dist.setValue(10)
        self.slider_dist.valueChanged.connect(
            lambda v: self.lbl_dist.setText(Locale.get("LBL_MINDIST", v / 100.0))
        )
        umap_layout.addWidget(self.lbl_dist)
        umap_layout.addWidget(self.slider_dist)
        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group)

        # === 3. Clustering ===
        cluster_group = CollapsibleGroupBox(Locale.get("GRP_CLUSTER"))
        cluster_layout = QVBoxLayout()
        self.lbl_res = QLabel(Locale.get("LBL_RES", 0.30))
        self.slider_res = QSlider(Qt.Orientation.Horizontal)
        self.slider_res.setRange(1, 200)
        self.slider_res.setValue(30)
        self.slider_res.valueChanged.connect(self.on_res_slider_change)
        cluster_layout.addWidget(self.lbl_res)
        cluster_layout.addWidget(self.slider_res)
        cluster_group.setLayout(cluster_layout)

        self.btn_run = QPushButton(Locale.get("BTN_EXECUTE"))
        self.btn_run.setObjectName("MainRunBtn")  # ID for specific styling
        self.btn_run.setMinimumHeight(50)
        self.btn_run.clicked.connect(self.emit_run)
        layout.addWidget(self.btn_run)
        layout.addStretch()
        layout.addWidget(cluster_group)

    def on_res_slider_change(self, value):
        self.lbl_res.setText(Locale.get("LBL_RES", value / 100.0))
        self.res_debounce_timer.start()

    def emit_resolution_change(self):
        val = self.slider_res.value() / 100.0
        self.resolution_change_signal.emit(val)

    def populate_features(self, features):
        self.feature_list.clear()
        for f in features:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.feature_list.addItem(item)

    def filter_feature_list(self, text):
        if "," in text:
            tokens = [t.strip().lower() for t in text.split(",")]
            tokens = [t for t in tokens if t]
            if not tokens:
                for i in range(self.feature_list.count()):
                    self.feature_list.item(i).setHidden(False)
            else:
                for i in range(self.feature_list.count()):
                    item = self.feature_list.item(i)
                    item_text = item.text().lower()
                    item.setHidden(not any(token in item_text for token in tokens))
        else:
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                item.setHidden(text.lower() not in item.text().lower())

    def select_all(self, _=None):
        self.feature_list.blockSignals(True)
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)
        self.feature_list.blockSignals(False)
        self.emit_apply_data()

    def select_none(self, _=None):
        self.feature_list.blockSignals(True)
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)
        self.feature_list.blockSignals(False)
        self.emit_apply_data()

    def get_selected_features(self):
        selected = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def emit_apply_data(self):
        self.apply_data_signal.emit(
            self.combo_norm.currentText(), self.get_selected_features()
        )

    def emit_run(self):
        self.run_signal.emit(
            {
                "n_components": self.slider_pca.value(),
                "n_neighbors": self.slider_neighbors.value(),
                "min_dist": self.slider_dist.value() / 100.0,
                "random_state": 42,
                "resolution": self.slider_res.value() / 100.0,
            }
        )

    def update_pca_label(self, v=None):
        if v is None:
            v = self.slider_pca.value()
        try:
            var_ratios = self.model.get_pca_variance_ratio()
            captured = float(np.sum(var_ratios[:v]) * 100)
            self.lbl_pca.setText(f"Principal Components [{v:02d}] (Captured: {captured:.1f}%)")
        except Exception:
            self.lbl_pca.setText(Locale.get("LBL_COMPONENTS", v))

    def set_max_pca(self, max_val):
        self.slider_pca.setMaximum(max_val)
        self.update_pca_label()

    def show_threshold_dialog(self):
        normalization = self.combo_norm.currentText()
        selected_features = self.get_selected_features()
        if len(selected_features) < 1:
            QMessageBox.warning(self, "Input Error", "Please select at least one protein.")
            return

        try:
            self.model.reprocess_data(normalization, selected_features)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to normalize data: {e}")
            return

        dialog = ThresholdDialog(self.model, selected_features, parent=self)
        if dialog.exec():
            try:
                self.model.reprocess_data(normalization, selected_features)
                self.update_pca_label()
                self.apply_data_signal.emit(normalization, selected_features)

                total_cells = len(self.model.adata)
                filtered_cells = int(np.sum(self.model.adata.obs["is_filtered"]))
                pct_filtered = (filtered_cells / total_cells * 100) if total_cells > 0 else 0.0

                QMessageBox.information(
                    self,
                    "Filtering Results",
                    f"Thresholds applied successfully.\n\n"
                    f"Cells filtered: {filtered_cells:,} / {total_cells:,} ({pct_filtered:.2f}%)"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply thresholds: {e}")



class PlotView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 5), dpi=100)
        apply_matplotlib_theme(self.figure)

        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, "NO_SIGNAL", ha="center", va="center")
        self.ax.axis("off")
        self.current_adata = None
        self.current_key = None
        self.current_hide_hidden = False

    def refresh_theme(self):
        """Called when theme toggles"""
        apply_matplotlib_theme(self.figure, self.ax)
        if self.current_adata and self.current_key:
            self.update_plot(
                self.current_adata, self.current_key, self.current_hide_hidden
            )
        else:
            self.canvas.draw()

    def update_plot(self, adata, color_key, hide_hidden=False):
        self.current_adata = adata
        self.current_key = color_key
        self.current_hide_hidden = hide_hidden

        self.figure.clear()
        apply_matplotlib_theme(self.figure)
        self.ax = self.figure.add_subplot(111)
        apply_matplotlib_theme(self.figure, self.ax)

        c = ThemeManager.get_current()
        try:
            # Exclude 'Filtered' cells and cells with NaN coordinates from the UMAP plot
            mask_valid = adata.obs[color_key].notna() & (adata.obs[color_key] != "Filtered")
            if "X_umap" in adata.obsm:
                mask_valid = mask_valid & (~np.isnan(adata.obsm["X_umap"]).any(axis=1))
            adata_plot = adata[mask_valid]

            if len(adata_plot) > 0:
                sc.pl.umap(
                    adata_plot,
                    color=color_key,
                    ax=self.ax,
                    show=False,
                    title=f"{color_key.upper().split('_')[0]} UMAP",
                    legend_loc="on data",
                    frameon=False,
                )
            else:
                self.ax.text(0.5, 0.5, "No non-filtered cells to plot", ha="center", va="center", color=c["text_main"])
            for text_obj in list(self.ax.texts):
                if text_obj.get_text() in ("NA", "NaN", "nan"):
                    text_obj.remove()
            for text_obj in self.ax.texts:
                text_obj.set_fontfamily("monospace")
                text_obj.set_fontsize(10)
                text_obj.set_weight("bold")

                fg_color = (
                    "#ffffff"
                    if ThemeManager._get_current_mode() == "DARK"
                    else "#000000"
                )
                text_obj.set_color(fg_color)

                text_obj.set_path_effects(
                    [PathEffects.withStroke(linewidth=4, foreground=c["bg_dark"])]
                )
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            logger.exception("UMAP_PLOT_FAIL")
            logger.debug(f"Error: {e}")
            self.ax.text(
                0.5, 0.5, Locale.get("PLOT_NO_SIGNAL"), ha="center", color=c["alert"]
            )
            self.canvas.draw()


class HeatmapView(QWidget):
    def __init__(self):
        super().__init__()
        # Main layout holds a QSplitter
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left: Controls Panel
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(15)

        # Config GroupBox
        self.config_group = QGroupBox("Heatmap Configuration")
        self.config_layout = QVBoxLayout(self.config_group)
        self.config_layout.setContentsMargins(15, 20, 15, 15)
        self.config_layout.setSpacing(12)

        # Scaling Option
        self.config_layout.addWidget(QLabel("Color Scaling:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems([
            "No Scaling",
            "Scale per Protein (Z-Score)",
            "Scale per Protein (0-1)",
            "Scale per Cell (0-1)"
        ])
        self.combo_scale.setToolTip(
            "Color Scaling Options:\n\n"
            "• No Scaling: Plot raw log1p expression values directly.\n"
            "• Scale per Protein (Z-Score): Standardize each protein's expression to mean=0, std=1 across cells.\n"
            "• Scale per Protein (0-1): Min-max scale each protein to [0,1] across cells.\n"
            "• Scale per Cell (0-1): Min-max scale each cell's expression across selected proteins to [0,1]."
        )
        self.combo_scale.currentIndexChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.combo_scale)

        # Colormap Option
        self.config_layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems([
            "Default (Theme-Based)",
            "magma",
            "viridis",
            "plasma",
            "inferno",
            "cividis",
            "coolwarm",
            "RdBu_r",
            "seismic",
            "Blues",
            "Reds"
        ])
        self.combo_cmap.setToolTip(
            "Select the color palette for the heatmap.\n"
            "Diverging palettes (like coolwarm or RdBu_r) are recommended for Z-Score scaled data."
        )
        self.combo_cmap.currentIndexChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.combo_cmap)

        # Feature Sort Checkbox
        self.chk_sort_y = QCheckBox("Sort Y (Features)")
        self.chk_sort_y.stateChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.chk_sort_y)

        # Equiwidth Checkbox
        self.chk_equiwidth = QCheckBox("Equiwidth")
        self.chk_equiwidth.stateChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.chk_equiwidth)

        # Show Dendrogram Checkbox
        self.chk_dendrogram = QCheckBox("Show Dendrogram")
        self.chk_dendrogram.setToolTip("Show dendrogram hierarchically clustering features along the Y-axis.")
        self.chk_dendrogram.stateChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.chk_dendrogram)

        # Swap Axes Checkbox
        self.chk_swap_axes = QCheckBox("Swap Axes")
        self.chk_swap_axes.setChecked(True)
        self.chk_swap_axes.setToolTip("Swap X and Y axes. If checked, features are on Y-axis and clusters are on X-axis.")
        self.chk_swap_axes.stateChanged.connect(self.on_ui_change)
        self.config_layout.addWidget(self.chk_swap_axes)

        self.config_layout.addStretch()
        self.controls_layout.addWidget(self.config_group)

        # Export Button at the bottom of the left sidebar
        self.btn_export = QPushButton("EXPORT HEATMAP")
        self.btn_export.clicked.connect(self.export_heatmap)
        self.btn_export.setToolTip("Save the current heatmap to a PNG, PDF, or SVG file.")
        self.controls_layout.addWidget(self.btn_export)

        # Right: Plot Panel
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        apply_matplotlib_theme(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        # Add widgets to Splitter
        self.splitter.addWidget(self.controls_widget)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)

        # Store last params to allow re-drawing on theme switch
        self.last_params = None

    def on_ui_change(self):
        if self.last_params:
            self.update_heatmap(*self.last_params)

    def refresh_theme(self):
        if self.last_params:
            self.update_heatmap(*self.last_params)
        else:
            self.figure.clear()
            apply_matplotlib_theme(self.figure)
            self.canvas.draw()

    def export_heatmap(self):
        if not self.figure:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Heatmap",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches="tight")
                QMessageBox.information(self, "Export Success", f"Heatmap successfully exported to:\n{file_path}")
            except Exception as e:
                logger.exception("HEATMAP_EXPORT_FAIL")
                QMessageBox.critical(self, "Export Error", f"Failed to export heatmap:\n{str(e)}")

    def update_heatmap(
        self,
        adata,
        cluster_key,
        selected_features,
        visible_categories=None,
    ):
        self.last_params = (
            adata,
            cluster_key,
            selected_features,
            visible_categories,
        )

        # Cleanup old canvas
        try:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        except Exception:
            pass

        c = ThemeManager.get_current()

        def show_msg(text):
            self.figure = Figure()
            apply_matplotlib_theme(self.figure)
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, text, ha="center", color=c["alert"])
            ax.axis("off")
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)

        if adata is None:
            show_msg(Locale.get("HM_WAITING"))
            return

        try:
            # Drop cells where cluster_key is nan or 'Filtered'
            mask_cells = adata.obs[cluster_key].notna() & (adata.obs[cluster_key] != "Filtered")
            if visible_categories is not None:
                mask_cells = mask_cells & adata.obs[cluster_key].isin(visible_categories)
            if np.sum(mask_cells) == 0:
                show_msg(Locale.get("HM_VOID"))
                return

            sub_adata = adata[mask_cells, :][:, list(selected_features)].copy()

            # Initialize heatmap_data with raw log1p expression values to avoid pre-scaled data
            if adata.raw is not None:
                raw_subset = adata.raw[mask_cells, list(selected_features)]
                X_data = raw_subset.X.copy() if hasattr(raw_subset.X, "copy") else raw_subset.X
                if hasattr(X_data, "toarray"):
                    X_data = X_data.toarray()
                heatmap_data = sc.AnnData(
                    X=X_data,
                    obs=sub_adata.obs.copy(),
                    var=sub_adata.var.copy(),
                    uns=sub_adata.uns.copy(),
                    obsm=sub_adata.obsm.copy() if sub_adata.obsm is not None else None
                )
            else:
                heatmap_data = sub_adata

            # Apply scaling/normalization based on user selection
            scale_mode = self.combo_scale.currentText()
            standard_scale = None

            if scale_mode == "Scale per Protein (Z-Score)":
                try:
                    sc.pp.scale(heatmap_data, max_value=3)
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}")
            elif scale_mode == "Scale per Protein (0-1)":
                standard_scale = "var"
            elif scale_mode == "Scale per Cell (0-1)":
                standard_scale = "obs"

            # Sort Features (Y)
            final_features = list(selected_features)
            if self.chk_sort_y.isChecked():
                final_features.sort()

            # Handle Group Sorting (X)
            heatmap_data.obs["display_label"] = heatmap_data.obs[cluster_key].copy()
            final_categories = list(heatmap_data.obs["display_label"].cat.categories)
            if f"{cluster_key}_colors" in heatmap_data.uns:
                heatmap_data.uns["display_label_colors"] = heatmap_data.uns[
                    f"{cluster_key}_colors"
                ]

            # Apply Equiwidth Resampling
            if self.chk_equiwidth.isChecked():
                try:
                    target_n = 200  # Fixed width per cluster
                    resampled_indices = []

                    # We use the categories we just set
                    for cat in final_categories:
                        # Find integer indices in the current heatmap_data
                        cat_mask = heatmap_data.obs["display_label"] == cat
                        cat_indices = np.where(cat_mask)[0]

                        if len(cat_indices) > 0:
                            # Interpolate indices to stretch/shrink to target_n
                            # linspace gives us N evenly spaced numbers between 0 and len-1
                            sample_pos = np.linspace(0, len(cat_indices) - 1, target_n)
                            # Round to nearest integer index
                            sample_idx = np.round(sample_pos).astype(int)
                            # Map back to original indices
                            resampled_indices.append(cat_indices[sample_idx])

                    if resampled_indices:
                        all_indices = np.concatenate(resampled_indices)
                        heatmap_data = heatmap_data[all_indices].copy()
                except Exception as e:
                    logger.warning(f"Equiwidth resampling failed: {e}")

            # Compute dendrogram if enabled
            show_dendrogram = False
            if self.chk_dendrogram.isChecked():
                try:
                    # sc.tl.dendrogram calculates hierarchical clustering of categories
                    sc.tl.dendrogram(heatmap_data, groupby="display_label", use_rep="X")
                    show_dendrogram = True
                except Exception as e:
                    logger.warning(f"Could not compute dendrogram: {e}")
                    show_dendrogram = False

            # Determine colormap
            cmap_choice = self.combo_cmap.currentText()
            if cmap_choice == "Default (Theme-Based)":
                cmap = "magma" if ThemeManager._get_current_mode() == "DARK" else "viridis"
            else:
                cmap = cmap_choice

            # Generate Plot
            plt.close("all")
            # Force Matplotlib RC for this plot
            with plt.rc_context(
                {
                    "axes.facecolor": c["bg_dark"],
                    "figure.facecolor": c["bg_dark"],
                    "text.color": c["text_main"],
                    "axes.labelcolor": c["accent"],
                    "xtick.color": c["text_main"],
                    "ytick.color": c["text_main"],
                }
            ):
                ax_dict = sc.pl.heatmap(
                    heatmap_data,
                    var_names=final_features,
                    groupby="display_label",
                    swap_axes=self.chk_swap_axes.isChecked(),
                    show=False,
                    use_raw=False,
                    standard_scale=standard_scale,
                    cmap=cmap,
                    dendrogram=show_dendrogram,
                )

            # Extract Figure
            new_fig = None
            if isinstance(ax_dict, dict):
                new_fig = (
                    ax_dict["heatmap"].figure
                    if "heatmap" in ax_dict
                    else list(ax_dict.values())[0].figure
                )
            elif isinstance(ax_dict, list):
                new_fig = ax_dict[0].figure
            else:
                new_fig = ax_dict.figure

            # Remove titles from all axes and the main figure
            if new_fig:
                new_fig.suptitle("")
                for ax in new_fig.axes:
                    ax.set_title("")
                    if ax.get_xlabel() == "display_label":
                        ax.set_xlabel("")
                    if ax.get_ylabel() == "display_label":
                        ax.set_ylabel("")

            apply_matplotlib_theme(new_fig)
            self.figure = new_fig
            self.canvas = FigureCanvas(self.figure)
            self.plot_layout.addWidget(self.canvas)

        except Exception as e:
            logger.exception("HEATMAP_FAIL")
            show_msg(f"ERR: Need to rerunning UMAP.\n{str(e)}")


class AspectRatioCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return w


class RankedGenesView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()

        self.cluster_combo = QComboBox()
        self.cluster_combo.setPlaceholderText("Select Cluster")
        self.cluster_combo.currentIndexChanged.connect(self.on_update)

        self.genes_spin = QSpinBox()
        self.genes_spin.setRange(1, 20)
        self.genes_spin.setValue(5)
        self.genes_spin.valueChanged.connect(self.on_update)

        controls_layout.addWidget(QLabel("Cluster:"))
        controls_layout.addWidget(self.cluster_combo)
        controls_layout.addWidget(QLabel("Top Genes:"))
        controls_layout.addWidget(self.genes_spin)
        controls_layout.addWidget(QLabel("Color Scaling:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems([
            "No Scaling",
            "Scale per Protein (Z-Score)",
            "Scale per Protein (0-1)",
            "Scale per Cell (0-1)"
        ])
        self.combo_scale.setToolTip(
            "Color Scaling Options:\n\n"
            "• No Scaling: Plot raw log1p expression values directly.\n"
            "• Scale per Protein (Z-Score): Standardize each protein's expression to mean=0, std=1 across cells.\n"
            "• Scale per Protein (0-1): Min-max scale each protein to [0,1] across cells.\n"
            "• Scale per Cell (0-1): Min-max scale each cell's expression across ranked genes to [0,1]."
        )
        self.combo_scale.currentIndexChanged.connect(self.on_update)
        controls_layout.addWidget(self.combo_scale)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Scroll Area
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        layout.addWidget(self.scrollArea)

        # Container for plots (Grid Layout)
        self.plot_container = QWidget()
        self.plot_layout = QGridLayout(self.plot_container)
        self.plot_layout.setSpacing(10)
        self.scrollArea.setWidget(self.plot_container)

        self.adata = None
        self.key = None
        self.hide_hidden = False

    def refresh_theme(self):
        # Simply regenerating plots picks up the current theme
        self.on_update()

    def set_data(self, adata, key, hide_hidden=False):
        self.adata = adata
        self.key = key
        self.hide_hidden = hide_hidden

        if self.adata and self.key and self.key in self.adata.obs:
            cats = self.adata.obs[self.key].cat.categories
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            for c in cats:
                name = str(c)
                self.cluster_combo.addItem(name, name)
            self.cluster_combo.blockSignals(False)

            # Check validity of existing rank_genes_groups
            recompute = True
            if "rank_genes_groups" in self.adata.uns:
                rgg = self.adata.uns["rank_genes_groups"]
                if rgg.get("params", {}).get("groupby") == self.key:
                    # Check if all current categories are present in the ranked results
                    stored_names = (
                        set(rgg["names"].dtype.names)
                        if "names" in rgg and hasattr(rgg["names"], "dtype")
                        else set()
                    )
                    current_cats = set(str(c) for c in cats)
                    if current_cats.issubset(stored_names):
                        recompute = False

            if recompute:
                try:
                    sc.tl.rank_genes_groups(
                        self.adata, groupby=self.key, method="t-test"
                    )
                except Exception as e:
                    logger.error(f"Failed to rank genes: {e}")

    def update_cluster_names(self, renames):
        """
        Updates the displayed names in the combo box based on the dictionary
        renames: { int_code : str_new_name }
        """
        self.cluster_combo.blockSignals(True)
        for i in range(self.cluster_combo.count()):
            if i in renames:
                self.cluster_combo.setItemText(i, renames[i])
        self.cluster_combo.blockSignals(False)

    def on_update(self):
        if self.adata is None or self.key is None:
            return

        # Use the UserData (original name) for data lookup
        group = self.cluster_combo.currentData()
        if not group:
            # Fallback for safety
            group = self.cluster_combo.currentText()
        n_genes = self.genes_spin.value()

        if not group:
            return

        try:
            dc_cluster_genes = sc.get.rank_genes_groups_df(
                self.adata, group=group
            ).head(n_genes)["names"]
        except KeyError:
            # Fallback: try recomputing if group is missing
            logger.warning(f"Group {group} not found in ranked genes. Recomputing...")
            try:
                sc.tl.rank_genes_groups(self.adata, groupby=self.key, method="t-test")
                dc_cluster_genes = sc.get.rank_genes_groups_df(
                    self.adata, group=group
                ).head(n_genes)["names"]
            except Exception as e:
                logger.error(f"Retry failed: {e}")
                return

        # Clear existing plots
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        genes_list = list(dc_cluster_genes)
        # Extract expression matrix
        if len(genes_list) > 0:
            expr_matrix = np.zeros((self.adata.n_obs, len(genes_list)), dtype=np.float64)
            for j, gene in enumerate(genes_list):
                if self.adata.raw is not None and gene in self.adata.raw.var_names:
                    gene_idx = list(self.adata.raw.var_names).index(gene)
                    col_data = self.adata.raw.X[:, gene_idx]
                    if hasattr(col_data, "toarray"):
                        col_data = col_data.toarray()
                    expr_matrix[:, j] = np.asarray(col_data).ravel()
                else:
                    expr_matrix[:, j] = sc.get.obs_df(self.adata, keys=[gene])[gene].values

            # Apply scaling on the entire matrix based on user selection
            scale_mode = self.combo_scale.currentText()
            if scale_mode == "Scale per Protein (Z-Score)":
                means = expr_matrix.mean(axis=0)
                stds = expr_matrix.std(axis=0)
                stds[stds == 0] = 1.0
                expr_matrix = (expr_matrix - means) / stds
                expr_matrix = np.clip(expr_matrix, -3, 3)
            elif scale_mode == "Scale per Protein (0-1)":
                mins = expr_matrix.min(axis=0)
                maxs = expr_matrix.max(axis=0)
                diffs = maxs - mins
                diffs[diffs == 0] = 1.0
                expr_matrix = (expr_matrix - mins) / diffs
            elif scale_mode == "Scale per Cell (0-1)":
                mins = expr_matrix.min(axis=1, keepdims=True)
                maxs = expr_matrix.max(axis=1, keepdims=True)
                diffs = maxs - mins
                diffs[diffs == 0] = 1.0
                expr_matrix = (expr_matrix - mins) / diffs
        else:
            expr_matrix = np.empty((self.adata.n_obs, 0))

        try:
            colors = [self.key, *genes_list]
            ncols = 3
            c = ThemeManager.get_current()

            for i, color_key in enumerate(colors):
                # Create individual Figure and Canvas
                fig = Figure(figsize=(4, 4), dpi=100)
                apply_matplotlib_theme(fig)
                canvas = AspectRatioCanvas(fig)
                canvas.setMinimumSize(350, 350)  # Ensure minimum size per plot

                ax = fig.add_subplot(111)

                is_categorical = color_key == self.key

                visible_mask = (
                    self.adata.obs[self.key].notna() & (self.adata.obs[self.key] != "Filtered")
                    if self.hide_hidden
                    else (self.adata.obs[self.key] != "Filtered").values
                )
                x = self.adata.obsm["X_umap"][:, 0][visible_mask]
                y = self.adata.obsm["X_umap"][:, 1][visible_mask]

                ax.set_title(color_key)
                ax.axis("off")
                ax.set_aspect("equal", "box")

                if is_categorical:
                    cats = self.adata.obs[color_key][visible_mask]
                    cat_colors = self.adata.uns.get(f"{color_key}_colors")
                    if cat_colors is None:
                        import matplotlib.colors as mcolors
                        import matplotlib.pyplot as plt

                        cmap = plt.get_cmap("tab20")
                        base_colors = (
                            cmap.colors
                            if hasattr(cmap, "colors")
                            else [cmap(i) for i in range(cmap.N)]
                        )
                        cat_colors = [
                            mcolors.to_hex(base_colors[i % len(base_colors)])
                            for i in range(len(cats.cat.categories))
                        ]

                    color_map_dict = dict(zip(cats.cat.categories, cat_colors))
                    c_values = [
                        color_map_dict.get(v, "#cccccc") if pd.notna(v) else "#cccccc"
                        for v in cats
                    ]
                    ax.scatter(x, y, c=c_values, s=5, alpha=0.8, edgecolors="none")

                    for cat in cats.cat.categories:
                        cat_mask = cats == cat
                        if cat_mask.any():
                            cat_x = x[cat_mask].mean()
                            cat_y = y[cat_mask].mean()
                            ax.text(cat_x, cat_y, str(cat), ha="center", va="center")
                else:
                    c_values = expr_matrix[:, i - 1][visible_mask]
                    sort_idx = np.argsort(c_values)
                    ax.scatter(
                        x[sort_idx],
                        y[sort_idx],
                        c=c_values[sort_idx],
                        cmap=c["plot_cmap"],
                        s=5,
                        alpha=0.8,
                        edgecolors="none",
                    )

                # --- COLOR SETTINGS ---
                fg_color = (
                    "#ffffff"
                    if ThemeManager._get_current_mode() == "DARK"
                    else "#000000"
                )
                ax.title.set_color(fg_color)
                ax.title.set_fontfamily("monospace")
                ax.title.set_weight("bold")

                for text_obj in ax.texts:
                    text_obj.set_fontfamily("monospace")
                    text_obj.set_fontsize(8)
                    text_obj.set_weight("bold")
                    text_obj.set_color(fg_color)
                    text_obj.set_path_effects(
                        [PathEffects.withStroke(linewidth=3, foreground=c["bg_dark"])]
                    )

                # Legend styling
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_color(fg_color)

                # Manually add colorbar for continuous data
                if not is_categorical and ax.collections:
                    mappable = ax.collections[-1]
                    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
                    
                    if len(c_values) > 0:
                        c_min = float(c_values.min())
                        c_max = float(c_values.max())
                        c_med = float(np.median(c_values))
                        
                        unique_ticks = np.unique([c_min, c_med, c_max])
                        cbar.set_ticks(unique_ticks)
                        
                        labels_dict = {
                            c_min: f"Min\n{c_min:.2f}",
                            c_med: f"Med\n{c_med:.2f}",
                            c_max: f"Max\n{c_max:.2f}"
                        }
                        cbar.set_ticklabels([labels_dict[t] for t in unique_ticks])
                    
                    cbar.ax.yaxis.set_tick_params(colors=fg_color, labelsize=7)
                    cbar.outline.set_edgecolor(c["accent_dim"])
                else:
                    ax.axis("off")

                fig.tight_layout()

                # Add to Grid Layout
                row = i // ncols
                col = i % ncols
                self.plot_layout.addWidget(canvas, row, col)

            # Add a vertical spacer to push the rows to the top if they don't fill the vertical space
            if len(colors) > 0:
                last_row = (len(colors) - 1) // ncols
                spacer = QWidget()
                spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
                self.plot_layout.addWidget(spacer, last_row + 1, 0, 1, ncols)

        except Exception as e:
            logger.error(f"Error updating ranked genes view: {e}")
            # Show error in a label if plotting fails
            lbl = QLabel(f"Error: {str(e)}")
            lbl.setStyleSheet(f"color: {c['alert']}")
            self.plot_layout.addWidget(lbl, 0, 0)


class SegmentationView(QWidget):
    cluster_state_changed = pyqtSignal(list, dict)

    def __init__(self, segmentation_data=None, view_tab=None):
        super().__init__()
        self.view_tab = view_tab
        self.adata = None
        self.cluster_key = None
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.seg_data = segmentation_data
        if self.seg_data is not None:
            self.seg_data = self.seg_data.astype(int)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image
        left_w = QWidget()
        l_layout = QVBoxLayout(left_w)
        l_layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel(Locale.get("SEG_STATUS_OFF"))
        self.info_label.setStyleSheet("font-weight: bold;")
        l_layout.addWidget(self.info_label)

        self.pg_view = pg.ImageView()
        self.pg_view.ui.roiBtn.hide()
        self.pg_view.ui.menuBtn.hide()
        self.pg_view.ui.histogram.hide()
        pg.setConfigOption("background", ThemeManager.get_current()["pg_bg"])
        l_layout.addWidget(self.pg_view)
        self.splitter.addWidget(left_w)

        # Right: Controls
        right_w = QWidget()
        r_layout = QVBoxLayout(right_w)

        r_layout.addWidget(QLabel(Locale.get("SEG_CLUSTER_FILTER")))

        self.cluster_list = QListWidget()
        self.cluster_list.itemChanged.connect(self.on_item_changed)
        r_layout.addWidget(self.cluster_list)

        btn_row = QHBoxLayout()
        self.btn_all = QPushButton(Locale.get("BTN_ALL"))
        self.btn_all.clicked.connect(self.select_all)
        self.btn_none = QPushButton(Locale.get("BTN_CLR"))
        self.btn_none.clicked.connect(self.select_none)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        r_layout.addLayout(btn_row)

        self.chk_hide_hidden = QCheckBox("HIDE FILTERED ON UMAP")
        self.chk_hide_hidden.stateChanged.connect(
            lambda: self.cluster_state_changed.emit(*self.get_state())
        )
        r_layout.addWidget(self.chk_hide_hidden)

        self.btn_export_to_view = QPushButton(Locale.get("BTN_EXPORT_VIEW"))
        self.btn_export_to_view.setEnabled(False)
        if self.view_tab is None:
            self.btn_export_to_view.setVisible(False)
        self.btn_export_to_view.clicked.connect(self.export_to_view_tab)
        r_layout.addWidget(self.btn_export_to_view)

        self.splitter.addWidget(right_w)
        self.splitter.setStretchFactor(0, 4)
        main_layout.addWidget(self.splitter)

        if self.seg_data is not None:
            self.pg_view.setImage(self.seg_data)

        self.npy_id_to_cat_idx = None
        self.npy_cat_idx_to_rgb = None
        self._last_visible_set = set()

    def refresh_theme(self):
        c = ThemeManager.get_current()
        pg.setConfigOption("background", c["pg_bg"])
        self.info_label.setStyleSheet(f"color: {c['alert']}; font-weight: bold;")

    def get_state(self):
        visible = []
        renames = {}
        for i in range(self.cluster_list.count()):
            item = self.cluster_list.item(i)
            original_idx = item.data(Qt.ItemDataRole.UserRole)
            renames[original_idx] = item.text()
            if item.checkState() == Qt.CheckState.Checked:
                visible.append(original_idx)
        return visible, renames

    def render_clusters(self, adata, cluster_key):
        if self.seg_data is None:
            return

        self.adata = adata
        self.cluster_key = cluster_key
        self.refresh_theme()
        categories = adata.obs[cluster_key].cat.categories
        if f"{cluster_key}_colors" not in adata.uns or len(adata.uns[f"{cluster_key}_colors"]) != len(categories):
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt

            c = ThemeManager.get_current()
            palette_name = c.get("plot_palette", "tab20")
            cmap = plt.get_cmap(palette_name)
            base_colors = (
                cmap.colors
                if hasattr(cmap, "colors")
                else [cmap(i) for i in range(cmap.N)]
            )
            colors = [
                mcolors.to_hex(base_colors[i % len(base_colors)])
                for i in range(len(categories))
            ]
            adata.uns[f"{cluster_key}_colors"] = np.array(colors)
        cluster_colors = adata.uns[f"{cluster_key}_colors"]
        self.npy_cat_idx_to_rgb = np.array([to_rgb(c) for c in cluster_colors])

        if self.seg_data is not None:
            valid_ids = adata.obs["cell_id"].values.astype(int)
            max_id = max(self.seg_data.max(), valid_ids.max())
            self.npy_id_to_cat_idx = np.full(max_id + 1, -1, dtype=np.int32)
            self.npy_id_to_cat_idx[valid_ids] = adata.obs[cluster_key].cat.codes.values

        self.cluster_list.blockSignals(True)
        self.cluster_list.clear()
        for cat_name, cat_code in zip(categories, range(len(categories))):
            item = QListWidgetItem(str(cat_name))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEditable
            )
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, int(cat_code))
            self.cluster_list.addItem(item)
        self.regenerate_image(force=True)
        self.cluster_list.blockSignals(False)
        self.btn_export_to_view.setEnabled(True)
        self.cluster_state_changed.emit(*self.get_state())

    def on_item_changed(self, item):
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def select_all(self):
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            self.cluster_list.item(i).setCheckState(Qt.CheckState.Checked)
        self.cluster_list.blockSignals(False)
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def select_none(self):
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            self.cluster_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.cluster_list.blockSignals(False)
        self.regenerate_image()
        self.cluster_state_changed.emit(*self.get_state())

    def regenerate_image(self, force=False):
        if self.seg_data is None or self.npy_id_to_cat_idx is None:
            return
        active_palette = np.zeros_like(self.npy_cat_idx_to_rgb)
        name_to_target = {}
        count = 0
        self.cluster_list.blockSignals(True)
        for i in range(self.cluster_list.count()):
            item = self.cluster_list.item(i)
            orig = item.data(Qt.ItemDataRole.UserRole)
            txt = item.text()
            target = name_to_target.get(txt, orig)
            name_to_target[txt] = target

            if 0 <= target < len(self.npy_cat_idx_to_rgb):
                rgb = self.npy_cat_idx_to_rgb[target]
                pix = QPixmap(12, 12)
                pix.fill(QColor.fromRgbF(*rgb))
                item.setIcon(QIcon(pix))

            if item.checkState() == Qt.CheckState.Checked:
                count += 1
                if 0 <= orig < len(active_palette):
                    active_palette[orig] = self.npy_cat_idx_to_rgb[target]
        self.cluster_list.blockSignals(False)
        safe_seg = self.seg_data.copy()
        mask_overflow = safe_seg >= len(self.npy_id_to_cat_idx)
        safe_seg[mask_overflow] = 0
        pixel_cat = self.npy_id_to_cat_idx[safe_seg]
        mask_valid = pixel_cat >= 0
        final = np.zeros(self.seg_data.shape + (3,), dtype=np.float32)
        final[mask_valid] = active_palette[pixel_cat[mask_valid]]
        self.pg_view.setImage(
            final.transpose(1, 0, 2),
            autoRange=False,
            autoLevels=False,
            levels=(0, 1),
            axes={"x": 0, "y": 1, "c": 2},
        )
        self.info_label.setText(Locale.get("SEG_ACTIVE_COUNT", count))

    def export_to_view_tab(self):
        if self.view_tab is None:
            QMessageBox.warning(self, "Export Warning", "View tab reference is not available.")
            return

        if self.seg_data is None:
            QMessageBox.warning(self, "Export Warning", "No segmentation data loaded.")
            return

        if not hasattr(self, "adata") or self.adata is None or not hasattr(self, "cluster_key") or self.cluster_key is None:
            QMessageBox.warning(self, "Export Warning", "No clustering data available. Please run UMAP clustering first.")
            return

        visible, renames = self.get_state()
        cluster_colors = self.adata.uns.get(f"{self.cluster_key}_colors", [])

        try:
            self.view_tab.add_cluster_layer(
                self.adata,
                self.cluster_key,
                cluster_colors,
                visible,
                renames
            )
            QMessageBox.information(
                self,
                "Export Success",
                "Successfully exported cluster labels to the View Tab as a new layer."
            )
        except Exception as e:
            logger.exception("EXPORT_TO_VIEW_TAB_FAIL")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export cluster labels to view tab:\n{str(e)}"
            )


class UMAPVisualizer(QMainWindow):
    def __init__(self, df, segmentation=None, parent=None, view_tab=None):
        super().__init__(parent)
        self.view_tab = view_tab
        self.setWindowTitle(Locale.get("APP_TITLE"))
        self.resize(1300, 900)
        self.setStyleSheet(ThemeManager.get_stylesheet())
        self.is_running_full_pipeline = False
        self.analysis_worker = None
        self.clustering_worker = None

        try:
            self.model = DataModel(df, normalization="none")
            self.segmentation = segmentation
        except Exception as e:
            logger.critical(f"DataModel init failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "UMAP Error",
                f"Failed to initialize data model:\n\n{e}\n\nSee log for full traceback.",
            )
            return

        try:
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            # === TOOLBAR & THEME TOGGLE ===
            top_bar = QHBoxLayout()
            header = QLabel(Locale.get("HEADER"))
            header.setObjectName("h1")
            top_bar.addWidget(header)
            top_bar.addStretch()

            self.btn_theme = QPushButton(Locale.get("THEME_BTN"))
            self.btn_theme.setFixedWidth(200)
            self.btn_theme.clicked.connect(self.toggle_theme)
            top_bar.addWidget(self.btn_theme)

            self.btn_export = QPushButton(Locale.get("EXPORT_BTN"))
            self.btn_export.setFixedWidth(200)
            self.btn_export.setEnabled(False)
            self.btn_export.clicked.connect(self.export_cluster_labels)
            top_bar.addWidget(self.btn_export)

            main_layout.addLayout(top_bar)

            # === TABS ===
            self.tabs = QTabWidget()
            main_layout.addWidget(self.tabs)

            self.umap_tab = QWidget()
            self.setup_umap_tab()
            self.tabs.addTab(self.umap_tab, Locale.get("TAB_UMAP"))

            self.seg_view = SegmentationView(segmentation, view_tab=self.view_tab)
            self.seg_view.cluster_state_changed.connect(self.trigger_heatmap_update)
            self.tabs.addTab(self.seg_view, Locale.get("TAB_SEG"))

            self.heatmap_view = HeatmapView()
            self.tabs.addTab(self.heatmap_view, Locale.get("TAB_HEATMAP"))

            self.ranked_genes_view = RankedGenesView()
            self.tabs.addTab(self.ranked_genes_view, "RANKED GENES")

            self.tabs.currentChanged.connect(self._on_tab_changed)

            # === STATUS BAR ===
            self.status_bar = self.statusBar()
            self.progress = QProgressBar()
            self.progress.setVisible(False)
            self.progress.setMaximumWidth(200)
            self.status_bar.addPermanentWidget(self.progress)
            self.status_label = QLabel(Locale.get("STATUS_READY"))
            self.status_bar.addWidget(self.status_label)

            self.update_status_style()
        except Exception as e:
            logger.critical(f"UMAP UI setup failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "UMAP Error",
                f"Failed to set up UMAP window:\n\n{e}\n\nSee log for full traceback.",
            )

    def setup_umap_tab(self):
        layout = QHBoxLayout(self.umap_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        all_feats = self.model.get_all_features()
        max_pcs = self.model.get_max_pcs()
        self.controls = UMAPControls(
            max_pca_components=max_pcs, all_features=all_feats, model=self.model
        )

        self.controls.run_signal.connect(self.start_full_analysis)
        self.controls.apply_data_signal.connect(self.apply_data_changes)
        self.controls.resolution_change_signal.connect(self.start_clustering_only)
        self.controls.recalc_pca_signal.connect(self.recalculate_pca)

        self.plot_view = PlotView()
        splitter.addWidget(self.controls)
        splitter.addWidget(self.plot_view)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    def trigger_heatmap_update(self, visible=None, renames=None):
        if self.model.adata is None or "leiden" not in self.model.adata.obs:
            return
        if visible is None or renames is None:
            visible, renames = self.seg_view.get_state()

        hide_hidden = self.seg_view.chk_hide_hidden.isChecked()

        adata = self.model.adata
        base_key = "leiden"
        view_key = f"{base_key}_view"

        orig_categories = list(adata.obs[base_key].cat.categories)

        if hide_hidden:
            present_codes = sorted([c for c in renames.keys() if c in visible])
        else:
            present_codes = sorted(list(renames.keys()))

        unique_new_labels = []
        for c in present_codes:
            lbl = renames.get(c, str(orig_categories[c]))
            if lbl not in unique_new_labels:
                unique_new_labels.append(lbl)

        new_labels = []
        for code in adata.obs[base_key].cat.codes:
            if code >= 0 and (not hide_hidden or code in visible):
                new_labels.append(renames.get(code, str(orig_categories[code])))
            else:
                new_labels.append(np.nan)

        adata.obs[view_key] = pd.Categorical(new_labels, categories=unique_new_labels)

        if f"{base_key}_colors" in adata.uns:
            orig_colors = adata.uns[f"{base_key}_colors"]

            color_map = {}
            for i, cat_name in enumerate(orig_categories):
                if not hide_hidden or i in visible:
                    new_name = renames.get(i, str(cat_name))
                    if new_name not in color_map:
                        color_map[new_name] = orig_colors[i]

            view_colors = [color_map[cat] for cat in unique_new_labels]
            adata.uns[f"{view_key}_colors"] = np.array(view_colors)

        self.plot_view.update_plot(adata, view_key, hide_hidden)
        self.ranked_genes_view.set_data(adata, view_key, hide_hidden)

        selected_features = self.controls.get_selected_features()
        visible_categories = []
        for c in visible:
            lbl = renames.get(c, str(orig_categories[c]))
            if lbl not in visible_categories:
                visible_categories.append(lbl)

        self.heatmap_view.update_heatmap(
            adata, view_key, selected_features, visible_categories=visible_categories
        )

    def start_full_analysis(self, params):
        features = self.controls.get_selected_features()
        if len(features) < 1:
            QMessageBox.warning(self, "INPUT_ERR", "MIN_FEATURES_REQUIRED: 1")
            return

        self.set_busy_state(True)
        self.is_running_full_pipeline = True

        # Reprocess data with current normalization/features before running pipeline
        normalization = self.controls.combo_norm.currentText()
        try:
            self.model.reprocess_data(normalization, features)
            new_max_pcs = self.model.get_max_pcs()
            self.controls.set_max_pca(new_max_pcs)
        except Exception as e:
            QMessageBox.critical(self, "ERR", str(e))
            self.set_busy_state(False)
            self.is_running_full_pipeline = False
            return

        self.analysis_worker = AnalysisWorker(
            model=self.model,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            n_components=params["n_components"],
            random_state=params["random_state"],
            resolution=params["resolution"],
        )
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def start_clustering_only(self, resolution):
        if self.is_running_full_pipeline:
            return
        if self.model.adata is None or self.model.adata_clean is None or "neighbors" not in self.model.adata_clean.uns:
            self.status_label.setText("ERR: PIPELINE_REQUIRED")
            return

        self.status_label.setText(f"REPARTITIONING [RES: {resolution}]")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self.clustering_worker = ClusteringWorker(self.model, resolution)
        self.clustering_worker.finished.connect(self.on_analysis_finished)
        self.clustering_worker.error.connect(self.on_analysis_error)
        self.clustering_worker.start()

    def _on_tab_changed(self, index):
        """Redraw the UMAP plot when switching back to the UMAP tab."""
        if index == 0 and self.plot_view.current_adata is not None:
            self.plot_view.canvas.draw_idle()

    def on_analysis_finished(self, adata, key):
        self.set_busy_state(False)
        self.is_running_full_pipeline = False
        n_clusters = len(adata.obs[key].unique())
        self.status_label.setText(
            f"COMPLETED: {n_clusters} CLUSTERS IDENTIFIED ({key.upper()})"
        )
        self.seg_view.render_clusters(adata, key)
        self.btn_export.setEnabled(True)

    def on_analysis_error(self, error_msg):
        self.set_busy_state(False)
        self.is_running_full_pipeline = False
        self.status_label.setText("CRITICAL_FAILURE")
        QMessageBox.critical(self, "PROCESS_TERMINATED", error_msg)

    def apply_data_changes(self, normalization, features):
        """Lightweight update when proteins/normalization change — no reprocessing."""
        if len(features) < 1:
            return
        # Max PCs = n_features (PCA limit)
        self.controls.set_max_pca(max(len(features), 2))
        msg = f"SELECTED: {len(features)} FEATS | {normalization} — click Run Analysis to apply"
        self.status_label.setText(msg)

    def recalculate_pca(self):
        """Reprocess data and recalculate PCA with current settings."""
        features = self.controls.get_selected_features()
        if len(features) < 1:
            QMessageBox.warning(self, "INPUT_ERR", "MIN_FEATURES_REQUIRED: 1")
            return

        normalization = self.controls.combo_norm.currentText()
        self.status_label.setText("Recalculating PCA...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.model.reprocess_data(normalization, features)
            new_max_pcs = self.model.get_max_pcs()
            self.controls.set_max_pca(new_max_pcs)
            self.controls.update_pca_label()
            self.status_label.setText("PCA recalculated successfully.")
        except Exception as e:
            logger.error(f"Failed to recalculate PCA: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to recalculate PCA:\n\n{e}")
            self.status_label.setText("PCA recalculation failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def cancel_analysis(self):
        """Cancel any running analysis or clustering worker."""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel()
            self.analysis_worker.finished.disconnect()
            self.analysis_worker.error.disconnect()
        if self.clustering_worker and self.clustering_worker.isRunning():
            self.clustering_worker.cancel()
            self.clustering_worker.finished.disconnect()
            self.clustering_worker.error.disconnect()
        self.is_running_full_pipeline = False
        self.set_busy_state(False)
        self.status_label.setText("Analysis cancelled.")
        logger.info("Analysis cancelled by user.")

    def set_busy_state(self, is_busy):
        if is_busy:
            self.controls.btn_run.setText(Locale.get("BTN_CANCEL"))
            try:
                self.controls.btn_run.clicked.disconnect()
            except TypeError:
                pass
            self.controls.btn_run.clicked.connect(self.cancel_analysis)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        else:
            self.controls.btn_run.setText(Locale.get("BTN_EXECUTE"))
            try:
                self.controls.btn_run.clicked.disconnect()
            except TypeError:
                pass
            self.controls.btn_run.clicked.connect(self.controls.emit_run)
            self.progress.setVisible(False)

    def toggle_theme(self):
        """Switches between Abyssal (Dark) and Sterile (Light)"""
        ThemeManager.toggle()

        # 1. Update Global Stylesheet
        self.setStyleSheet(ThemeManager.get_stylesheet())

        # 2. Update Matplotlib Canvases
        self.plot_view.refresh_theme()
        self.heatmap_view.refresh_theme()
        self.ranked_genes_view.refresh_theme()

        # 3. Update PyQtGraph
        self.seg_view.refresh_theme()

        # 4. Update Status Bar
        self.update_status_style()

        logger.info("THEME SWITCHED TO: %s", ThemeManager._get_current_mode())

    def update_status_style(self):
        c = ThemeManager.get_current()
        self.status_bar.setStyleSheet(
            f"background: {c['bg_panel']}; color: {c['accent']}; border-top: 2px solid {c['grid']}"
        )
        self.progress.setStyleSheet(
            f"""
            QProgressBar {{ border: 1px solid {c["accent_dim"]}; background-color: {c["bg_dark"]}; text-align: center; }}
            QProgressBar::chunk {{ background-color: {c["accent"]}; width: 5px; }}
        """
        )

    def export_cluster_labels(self):
        """Export cluster labels assigned to cell IDs to a CSV file."""
        if self.model.adata is None or "cell_id" not in self.model.adata.obs:
            QMessageBox.warning(self, "Export Warning", "No cell data available to export.")
            return

        cluster_key = "leiden"
        if cluster_key not in self.model.adata.obs:
            QMessageBox.warning(self, "Export Warning", "No clustering data available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Cluster Labels", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            adata = self.model.adata
            view_key = f"{cluster_key}_view"
            
            # Use leiden_view (which has active/renamed labels) if present
            cluster_labels = (
                adata.obs[view_key].values
                if view_key in adata.obs
                else adata.obs[cluster_key].values
            )

            export_df = pd.DataFrame({
                "Cell_ID": adata.obs["cell_id"].values,
                "Cluster": cluster_labels,
                "Original_Cluster": adata.obs[cluster_key].values
            })

            export_df.to_csv(file_path, index=False)
            QMessageBox.information(
                self,
                "Export Success",
                f"Successfully exported cluster labels to:\n{file_path}"
            )
        except Exception as e:
            logger.exception("EXPORT_FAIL")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export cluster labels:\n{str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
    )
    try:
        logger.info("Starting Application...")
        df = pd.read_csv("sp13_2_6.csv")
        segmentation = tifffile.imread("SP13 Stardist.tif")

        window = UMAPVisualizer(df=df, segmentation=segmentation)
        window.show()
        sys.exit(app.exec())
    except FileNotFoundError:
        logger.error("Demo files not found. Exiting.")
        logger.error("Demo files (csv/tif) not found. Please check paths.")
