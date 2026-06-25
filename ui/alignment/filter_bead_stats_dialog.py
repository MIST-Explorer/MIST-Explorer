import logging
import os
import typing

import numpy as np
import pandas as pd
import pyqtgraph as pg
import qtrangeslider
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QCursor, QPainter
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QScrollArea, QSizePolicy,
                             QSlider, QSplitter, QTableWidget,
                             QTableWidgetItem, QToolTip, QVBoxLayout, QWidget)

from core.image_utils import auto_contrast_helper
from ui.theme import ThemeManager
from utils import resource_path

logger = logging.getLogger(__name__)

_SLIDER_STEPS = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30]
_SLIDER_DEFAULT_INDEX = 3  # 1.00

class FilterBeadStatsDialog(QDialog):
    """
    Dialog to visualize cell filtering by bead / protein ratio.
    Displays a multi-channel overlay, stardist segmentation mask highlighting
    passing vs filtered cells, and shows bead counts per protein on mouse hover.
    """

    def __init__(self, payload: dict, parent=None, model=None):
        super().__init__(parent)
        self.model = model
        self.payload = payload

        # Extract data from model and payload
        self.seg_labels = model.segmentation_labels if model is not None else payload.get("segmentation_labels")
        self.bead_data = model.bead_data if model is not None else payload.get("bead_data")
        self.channel_to_color_code = model.channel_to_color_code if model is not None else payload.get("channel_to_color_code", {})
        self.channel_names = list(self.channel_to_color_code.keys()) if self.channel_to_color_code else payload.get("channel_names", [])
        
        self.num_proteins = payload.get("num_proteins", 1)
        self.total_cells = payload.get("total_cells", 0)
        self.distribution = payload.get("distribution", [])
        
        # Get background-subtracted images from model
        self.images_dict = {}
        if self.model is not None:
            try:
                self.images_dict = self.model._get_background_subtracted_images()
            except Exception as e:
                logger.error("Failed to load background-subtracted images: %s", e)

        # Get raw images
        self.raw_images_dict = self._get_raw_images()

        # Precompute unique cell IDs
        if self.seg_labels is not None and self.seg_labels.size > 0:
            self.unique_cell_ids = np.unique(self.seg_labels[self.seg_labels > 0])
        else:
            self.unique_cell_ids = np.array([], dtype=np.int32)

        # Precompute bead counts per cell
        self.cell_bead_counts = {}
        self.precompute_cell_bead_counts()

        # Setup threshold state
        current_threshold = 1.00
        if self.model and "bead_per_protein_threshold" in self.model.params:
            current_threshold = self.model.params["bead_per_protein_threshold"]

        # Find closest index in slider steps
        self.threshold_index = _SLIDER_DEFAULT_INDEX
        min_diff = float('inf')
        for idx, val in enumerate(_SLIDER_STEPS):
            diff = abs(val - current_threshold)
            if diff < min_diff:
                min_diff = diff
                self.threshold_index = idx
        self.threshold = _SLIDER_STEPS[self.threshold_index]

        # Colors for channels
        self.channel_colors = [
            (0, 255, 255),    # Cyan
            (0, 255, 0),      # Green
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Yellow
            (0, 128, 255),    # Electric Blue
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Violet
        ]

        self.image_items = {}
        self.max_intensities = {}
        self.contrast_controls = {}
        
        self.passing_cell_ids = set()
        self.failing_cell_ids = set()

        self.init_ui()

    def make_lut(self, color):
        """Generate a lookup table mapping intensity [0, 255] to a custom color with linear alpha."""
        r, g, b = color
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0] = np.linspace(0, r, 256)
        lut[:, 1] = np.linspace(0, g, 256)
        lut[:, 2] = np.linspace(0, b, 256)
        lut[:, 3] = np.linspace(0, 255, 256)
        return lut

    def _get_raw_images(self):
        try:
            if self.model is None:
                return {}
            uuid = self.model.source_uuid
            if uuid is None:
                uuid = self.model.storage.get_data("canvas_uuid")
                if uuid is None:
                    return {}
                uuid = uuid["value"]
            
            item = self.model.storage.get_data(uuid)
            if item is None:
                return {}
            data = item.get("data", None)
            if data is None:
                return {}

            images = {}
            from core.project_naming import is_segmentation_channel
            for channel_name in self.channel_names:
                channel_num = int(channel_name.split(" ")[-1])
                c = "Channel " + str(channel_num)
                if c in data:
                    wrapper = data[c]
                    if not is_segmentation_channel(wrapper):
                        images[channel_name] = wrapper.data
            return images
        except Exception as e:
            logger.error("Failed to load raw images: %s", e)
            return {}

    def precompute_cell_bead_counts(self):
        self.cell_bead_counts = {}
        for cell_id in self.unique_cell_ids:
            self.cell_bead_counts[cell_id] = {}

        if self.bead_data is None or self.seg_labels is None or self.seg_labels.size == 0:
            return

        coords = self.bead_data[:, 0:2].astype(int)
        x_limit = self.seg_labels.shape[1]
        y_limit = self.seg_labels.shape[0]

        in_bounds = (
            (coords[:, 0] >= 0) & (coords[:, 0] < x_limit)
            & (coords[:, 1] >= 0) & (coords[:, 1] < y_limit)
        )
        valid_coords = coords[in_bounds]
        valid_bead_data = self.bead_data[in_bounds]

        cell_ids = self.seg_labels[valid_coords[:, 1], valid_coords[:, 0]]

        # Build code_to_protein mapping
        code_to_protein = {}
        for channel_name, df in self.channel_to_color_code.items():
            try:
                df_clean = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
                arr = df_clean.to_numpy()
                for row in arr:
                    protein_name = row[0]
                    code = int("".join([str(int(x)) for x in row[1:]]))
                    code_to_protein[code] = protein_name
            except Exception:
                pass

        # Determine decoding cycles
        num_cycles = 3
        if self.channel_to_color_code:
            first_df = next(iter(self.channel_to_color_code.values()))
            num_cycles = first_df.columns.size - 1

        for i in range(len(valid_coords)):
            cell_id = int(cell_ids[i])
            if cell_id == 0:
                continue

            bead_row = valid_bead_data[i]
            try:
                cycle_cols = bead_row[2 : 2 + num_cycles]
                code = int("".join(map(str, map(int, cycle_cols))))
                protein_name = code_to_protein.get(code, f"Code {code}")
            except Exception:
                protein_name = "Unknown"

            if cell_id not in self.cell_bead_counts:
                self.cell_bead_counts[cell_id] = {}
            self.cell_bead_counts[cell_id][protein_name] = self.cell_bead_counts[cell_id].get(protein_name, 0) + 1

    def init_ui(self):
        self.setWindowTitle("Filter Cells by Expression Ratio")
        self.resize(1150, 850)

        tc = ThemeManager.instance().get_current()
        self.bg_color = tc.get("bg_primary", "#0b0c10")
        self.card_color = tc.get("bg_secondary", "#1f2833")
        self.text_color = tc.get("text_primary", "#c5c6c7")
        self.accent_color = tc.get("accent", "#66fcf1")
        self.border_color = tc.get("border", "#2d3436")
        self.danger_color = tc.get("danger", "#e57373")
        self.success_color = "#2ecc71"

        # Set main dialog layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        icons_dir = resource_path(os.path.join("assets", "icons")).replace("\\", "/")
        mode = ThemeManager.instance().current_mode
        tick_icon = "checkbox_tick_dark.svg" if mode == "DARK" else "checkbox_tick_light.svg"
        tick_path = f"{icons_dir}/{tick_icon}"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.bg_color};
            }}
            QCheckBox {{
                color: {self.text_color};
                font-weight: bold;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                background-color: {self.card_color};
                border: 1px solid {self.text_color};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.accent_color};
                border: 1px solid {self.accent_color};
                image: url({tick_path});
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {self.accent_color};
            }}
        """)

        # --- Content splitter ---
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(8)

        # 1. Left canvas panel (PyQtGraph PlotWidget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(tc.get("canvas_bg", "#000000"))
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().showGrid(x=False, y=False)
        self.plot_widget.getViewBox().invertY(True)

        # Add channels to PlotWidget
        active_dict = self.images_dict if self.images_dict else self.raw_images_dict
        for idx, ch_name in enumerate(self.channel_names):
            if ch_name not in active_dict:
                continue

            img_data = active_dict[ch_name]
            image_item = pg.ImageItem(autoDownsample=True)
            image_item.setOpts(smooth=False)
            image_item.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

            # Set color LUT
            color = self.channel_colors[idx % len(self.channel_colors)]
            image_item.setLookupTable(self.make_lut(color))

            # Set levels based on max intensity
            img_max = np.max(img_data)
            slider_max = float(img_max) if img_max > 0 else 1.0
            self.max_intensities[ch_name] = slider_max

            vmin, vmax = auto_contrast_helper(img_data, lower=1.0, upper=99.5)
            if slider_max > 0:
                vmin = (vmin / 255.0) * slider_max
                vmax = (vmax / 255.0) * slider_max

            vmin = max(0.0, float(vmin))
            vmax = min(slider_max, max(vmin + 0.001, float(vmax)))
            image_item.setImage(img_data)
            image_item.setLevels([vmin, vmax])

            self.plot_widget.addItem(image_item)
            self.image_items[ch_name] = image_item

        # 2. Add Cell Overlay ImageItem
        self.cell_overlay_item = pg.ImageItem(autoDownsample=True)
        self.cell_overlay_item.setOpts(smooth=False)
        self.cell_overlay_item.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        self.cell_overlay_item.setZValue(99)  # Draw above channel backdrops
        self.plot_widget.addItem(self.cell_overlay_item)

        # Setup lookup table for cell overlays
        # 0: background (transparent)
        # 1: passing cells (green)
        # 2: failing cells (red)
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[1] = [46, 204, 113, 90]   # Semi-transparent Green
        lut[2] = [231, 76, 60, 90]    # Semi-transparent Red
        self.cell_overlay_item.setLookupTable(lut)
        self.cell_overlay_item.setLevels([0, 255])

        # Enable mouse hover signals
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.on_mouse_moved
        )

        content_splitter.addWidget(self.plot_widget)

        # 3. Right Sidebar Control Panel
        controls_panel = QFrame()
        controls_panel.setObjectName("ControlsPanel")
        controls_panel.setMaximumWidth(340)
        controls_panel.setStyleSheet(f"""
            QFrame#ControlsPanel {{
                background-color: {self.card_color};
                border: 1px solid {self.border_color};
                border-radius: 10px;
            }}
        """)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(12)

        # --- Section A: Filter Threshold Control ---
        threshold_title = QLabel("Filter Threshold")
        threshold_title.setStyleSheet(f"color: {self.accent_color}; font-size: 14px; font-weight: bold; border: none;")
        controls_layout.addWidget(threshold_title)

        threshold_box = QFrame()
        threshold_box.setObjectName("ThresholdBox")
        threshold_box.setStyleSheet(f"""
            QFrame#ThresholdBox {{
                background-color: {self.bg_color};
                border: 1px solid {self.border_color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        threshold_box_layout = QVBoxLayout(threshold_box)
        threshold_box_layout.setSpacing(8)

        self.threshold_val_label = QLabel(f"Ratio: {self.threshold:.2f}")
        self.threshold_val_label.setStyleSheet(f"color: {self.text_color}; font-weight: bold; font-size: 13px; border: none;")
        threshold_box_layout.addWidget(self.threshold_val_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, len(_SLIDER_STEPS) - 1)
        self.threshold_slider.setValue(self.threshold_index)
        self.threshold_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {self.border_color};
                height: 5px;
                background: {self.bg_color};
                border-radius: 2.5px;
            }}
            QSlider::handle:horizontal {{
                background: {self.accent_color};
                border: none;
                width: 12px;
                height: 12px;
                margin: -3.5px 0;
                border-radius: 6px;
            }}
        """)
        self.threshold_slider.valueChanged.connect(self.on_slider_changed)
        threshold_box_layout.addWidget(self.threshold_slider)

        controls_layout.addWidget(threshold_box)

        # Separator line
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        sep1.setStyleSheet(f"background-color: {self.border_color}; max-height: 1px; border: none;")
        controls_layout.addWidget(sep1)

        # --- Section B: Layer Controls ---
        sidebar_title = QLabel("Layer Controls")
        sidebar_title.setStyleSheet(f"color: {self.accent_color}; font-size: 14px; font-weight: bold; border: none;")
        controls_layout.addWidget(sidebar_title)

        # Cell overlay toggle
        self.overlay_checkbox = QCheckBox("Show Cell Overlay")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.toggled.connect(self.cell_overlay_item.setVisible)
        controls_layout.addWidget(self.overlay_checkbox)

        # Background subtraction toggle
        self.bg_subtract_checkbox = QCheckBox("Subtract Background")
        self.bg_subtract_checkbox.setChecked(self.images_dict is not None and len(self.images_dict) > 0)
        self.bg_subtract_checkbox.setEnabled(self.images_dict is not None and len(self.images_dict) > 0)
        self.bg_subtract_checkbox.toggled.connect(self.on_bg_subtract_toggled)
        controls_layout.addWidget(self.bg_subtract_checkbox)

        # Auto Contrast All button
        self.auto_all_btn = QPushButton("Auto Contrast All")
        bg_hover = tc.get("bg_hover", "rgba(255, 255, 255, 0.06)")
        bg_pressed = tc.get("bg_pressed", "rgba(255, 255, 255, 0.12)")
        self.auto_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.accent_color};
                border: 1px solid {self.accent_color};
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
            QPushButton:pressed {{
                background-color: {bg_pressed};
            }}
        """)
        self.auto_all_btn.clicked.connect(self.auto_contrast_all)
        controls_layout.addWidget(self.auto_all_btn)

        # Separator line
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        sep2.setStyleSheet(f"background-color: {self.border_color}; max-height: 1px; border: none;")
        controls_layout.addWidget(sep2)

        # --- Scroll Area for Channel Sliders ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        edit_style = (
            f"QLineEdit {{"
            f"  font-size: 11px; font-weight: bold;"
            f"  color: {self.accent_color};"
            f"  background: {bg_hover};"
            f"  border: none; border-bottom: 1px solid {self.border_color};"
            f"  border-radius: 2px; padding: 1px 2px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-bottom: 2px solid {self.accent_color};"
            f"  background: {bg_pressed};"
            f"}}"
        )

        auto_btn_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 4px;
                padding: 1px 6px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
                color: {self.accent_color};
                border-color: {self.accent_color};
            }}
            QPushButton:pressed {{
                background-color: {bg_pressed};
            }}
        """

        def format_value(v):
            if v >= 100:
                return f"{int(v)}"
            elif v >= 1:
                return f"{v:.1f}"
            else:
                return f"{v:.3f}"

        for idx, ch_name in enumerate(self.channel_names):
            if ch_name not in self.image_items:
                continue

            color = self.channel_colors[idx % len(self.channel_colors)]
            color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"

            channel_card = QFrame()
            channel_card.setObjectName("ChannelCard")
            channel_card.setStyleSheet(f"""
                QFrame#ChannelCard {{
                    background-color: {self.bg_color};
                    border: 1px solid {self.border_color};
                    border-radius: 8px;
                    padding: 6px;
                }}
            """)
            channel_card_layout = QVBoxLayout(channel_card)
            channel_card_layout.setSpacing(4)

            # Row 1: Checkbox, Color Indicator, Auto Button
            row1 = QHBoxLayout()
            color_indicator = QLabel()
            color_indicator.setFixedSize(10, 10)
            color_indicator.setStyleSheet(f"background-color: {color_hex}; border-radius: 5px; border: none;")
            row1.addWidget(color_indicator)

            chk = QCheckBox(ch_name)
            chk.setChecked(True)
            chk.toggled.connect(lambda checked, ch=ch_name: self.image_items[ch].setVisible(checked))
            row1.addWidget(chk)
            row1.addStretch()

            auto_btn = QPushButton("Auto")
            auto_btn.setStyleSheet(auto_btn_style)
            auto_btn.clicked.connect(lambda _, ch=ch_name: self.auto_contrast_channel(ch))
            row1.addWidget(auto_btn)

            channel_card_layout.addLayout(row1)

            levels = self.image_items[ch_name].levels
            vmin_curr, vmax_curr = levels[0], levels[1]
            slider_max = self.max_intensities[ch_name]

            # Row 2: Contrast double slider
            row2 = QHBoxLayout()
            row2.setSpacing(6)

            con_lo_edit = QLineEdit(format_value(vmin_curr))
            con_lo_edit.setFixedWidth(45)
            con_lo_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            con_lo_edit.setStyleSheet(edit_style)

            contrast_slider = qtrangeslider.QDoubleRangeSlider(Qt.Orientation.Horizontal)
            contrast_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            contrast_slider.setMinimum(0.0)
            contrast_slider.setMaximum(slider_max)
            contrast_slider.blockSignals(True)
            contrast_slider.setValue((vmin_curr, vmax_curr))
            contrast_slider.blockSignals(False)
            contrast_slider.setStyleSheet(f"""
                QDoubleRangeSlider::groove:horizontal, QSlider::groove:horizontal {{
                    border: 1px solid {self.border_color};
                    height: 5px;
                    background: {self.bg_color};
                    border-radius: 2.5px;
                }}
                QDoubleRangeSlider::handle:horizontal, QSlider::handle:horizontal {{
                    background: {color_hex};
                    border: none;
                    width: 12px;
                    height: 12px;
                    margin: -3.5px 0;
                    border-radius: 6px;
                }}
            """)

            con_hi_edit = QLineEdit(format_value(vmax_curr))
            con_hi_edit.setFixedWidth(45)
            con_hi_edit.setStyleSheet(edit_style)

            row2.addWidget(con_lo_edit)
            row2.addWidget(contrast_slider)
            row2.addWidget(con_hi_edit)
            channel_card_layout.addLayout(row2)

            scroll_layout.addWidget(channel_card)

            # Store references
            self.contrast_controls[ch_name] = {
                "slider": contrast_slider,
                "lo_edit": con_lo_edit,
                "hi_edit": con_hi_edit,
                "image_item": self.image_items[ch_name],
                "max": slider_max,
            }

            # Setup event handlers for range sliders
            def make_on_contrast_changed(ch):
                def handler(val):
                    ctrl = self.contrast_controls[ch]
                    ctrl["lo_edit"].blockSignals(True)
                    ctrl["hi_edit"].blockSignals(True)
                    ctrl["lo_edit"].setText(format_value(val[0]))
                    ctrl["hi_edit"].setText(format_value(val[1]))
                    ctrl["lo_edit"].blockSignals(False)
                    ctrl["hi_edit"].blockSignals(False)
                    
                    vmin = val[0]
                    vmax = max(val[1], vmin + 0.001)
                    ctrl["image_item"].setLevels([vmin, vmax])
                return handler

            contrast_slider.valueChanged.connect(make_on_contrast_changed(ch_name))

            def make_on_lo_edited(ch):
                def handler():
                    ctrl = self.contrast_controls[ch]
                    slider = ctrl["slider"]
                    curr_val = slider.value()
                    try:
                        v = float(ctrl["lo_edit"].text())
                        v = max(0.0, min(v, curr_val[1] - 0.001))
                    except ValueError:
                        v = curr_val[0]
                    ctrl["lo_edit"].setText(format_value(v))
                    slider.setValue((v, curr_val[1]))
                return handler

            def make_on_hi_edited(ch):
                def handler():
                    ctrl = self.contrast_controls[ch]
                    slider = ctrl["slider"]
                    curr_val = slider.value()
                    try:
                        v = float(ctrl["hi_edit"].text())
                        v = max(curr_val[0] + 0.001, min(v, ctrl["max"]))
                    except ValueError:
                        v = curr_val[1]
                    ctrl["hi_edit"].setText(format_value(v))
                    slider.setValue((curr_val[0], v))
                return handler

            con_lo_edit.editingFinished.connect(make_on_lo_edited(ch_name))
            con_hi_edit.editingFinished.connect(make_on_hi_edited(ch_name))

        scroll.setWidget(scroll_content)
        controls_layout.addWidget(scroll)

        # Separator line
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        sep3.setStyleSheet(f"background-color: {self.border_color}; max-height: 1px; border: none;")
        controls_layout.addWidget(sep3)

        # --- Statistics Dashboard Card ---
        stats_card = QFrame()
        stats_card.setObjectName("StatsCard")
        stats_card.setStyleSheet(f"""
            QFrame#StatsCard {{
                background-color: {self.bg_color};
                border: 1px solid {self.border_color};
                border-radius: 8px;
                padding: 8px;
            }}
            QLabel {{
                color: {self.text_color};
                font-size: 12px;
                border: none;
            }}
        """)
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setSpacing(4)
        stats_layout.setContentsMargins(8, 6, 8, 6)

        self.stats_total_label = QLabel()
        self.stats_passing_label = QLabel()
        self.stats_filtered_label = QLabel()

        stats_layout.addWidget(self.stats_total_label)
        stats_layout.addWidget(self.stats_passing_label)
        stats_layout.addWidget(self.stats_filtered_label)
        controls_layout.addWidget(stats_card)

        # --- Protein Distribution Table ---
        self.dist_table = QTableWidget(len(self.distribution), 2)
        self.dist_table.setHorizontalHeaderLabels(["# Proteins", "% of Cells"])
        self.dist_table.horizontalHeader().setStretchLastSection(True)
        self.dist_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dist_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.dist_table.setMaximumHeight(150)
        self.dist_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {self.bg_color};
                gridline-color: {self.border_color};
                border: 1px solid {self.border_color};
                border-radius: 6px;
                color: {self.text_color};
            }}
            QHeaderView::section {{
                background-color: {self.card_color};
                color: {self.text_color};
                border: 1px solid {self.border_color};
                padding: 2px;
                font-weight: bold;
            }}
        """)

        for row, (label, percentage) in enumerate(self.distribution):
            self.dist_table.setItem(row, 0, QTableWidgetItem(str(label)))
            self.dist_table.setItem(row, 1, QTableWidgetItem(f"{percentage:.2f}%"))
        controls_layout.addWidget(self.dist_table)

        # --- Actions ---
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(6)

        reset_btn = QPushButton("Reset Zoom")
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.border_color};
            }}
        """)
        reset_btn.clicked.connect(self.plot_widget.autoRange)
        actions_layout.addWidget(reset_btn)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid {self.border_color};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.border_color};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply && Exit")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: {tc.get("text_on_accent", "#0b0c10")};
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: {tc.get("accent_hover", "#7ffdfa")};
            }}
            QPushButton:pressed {{
                background-color: {tc.get("accent_dim", "#45a29e")};
            }}
        """)
        apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(apply_btn)

        actions_layout.addLayout(btn_row)
        controls_layout.addLayout(actions_layout)

        content_splitter.addWidget(controls_panel)
        content_splitter.setSizes([830, 320])
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(content_splitter)
        self.setLayout(main_layout)

        # Render cell highlights initially
        self.update_cell_filtering()
        self.plot_widget.autoRange()

    def update_cell_filtering(self):
        """Re-evaluate passing vs filtered cells and update the segmentation overlay image."""
        self.passing_cell_ids.clear()
        self.failing_cell_ids.clear()

        # Group cells by status
        for cell_id in self.unique_cell_ids:
            bead_info = self.cell_bead_counts.get(cell_id, {})
            total_beads = sum(bead_info.values())
            
            ratio = total_beads / self.num_proteins if self.num_proteins > 0 else 0.0
            if ratio >= self.threshold:
                self.passing_cell_ids.add(cell_id)
            else:
                self.failing_cell_ids.add(cell_id)

        # Update stats labels
        passing_cnt = len(self.passing_cell_ids)
        filtered_cnt = len(self.failing_cell_ids)
        pct = (passing_cnt / self.total_cells * 100) if self.total_cells > 0 else 0.0

        self.stats_total_label.setText(f"Total Cells: <b>{self.total_cells:,}</b>")
        self.stats_passing_label.setText(
            f"Passing Cells: <b style='color:{self.success_color};'>{passing_cnt:,} ({pct:.2f}%)</b>"
        )
        self.stats_filtered_label.setText(
            f"Filtered Cells: <b style='color:{self.danger_color};'>{filtered_cnt:,} ({100.0 - pct:.2f}%)</b>"
        )

        # Update the pg.ImageItem
        if self.seg_labels is not None and self.seg_labels.size > 0:
            max_cell_id = int(self.unique_cell_ids.max()) if len(self.unique_cell_ids) > 0 else 0
            map_arr = np.zeros(max_cell_id + 1, dtype=np.uint8)
            
            # Map cell statuses: 1 = Pass, 2 = Fail
            for cid in self.passing_cell_ids:
                map_arr[cid] = 1
            for cid in self.failing_cell_ids:
                map_arr[cid] = 2

            state_img = map_arr[self.seg_labels]
            self.cell_overlay_item.setImage(state_img)
        else:
            self.cell_overlay_item.setImage(np.array([], dtype=np.uint8))

    def on_slider_changed(self, value):
        self.threshold = _SLIDER_STEPS[value]
        self.threshold_index = value
        self.threshold_val_label.setText(f"Filter threshold (beads / proteins): {self.threshold:.2f}")
        self.update_cell_filtering()

    def on_bg_subtract_toggled(self, checked):
        active_dict = self.images_dict if checked else self.raw_images_dict
        for ch_name, image_item in self.image_items.items():
            if ch_name not in active_dict:
                continue
            
            img_data = active_dict[ch_name]
            image_item.setImage(img_data)

            # Recalculate levels & max
            img_max = np.max(img_data)
            slider_max = float(img_max) if img_max > 0 else 1.0
            self.max_intensities[ch_name] = slider_max

            if ch_name in self.contrast_controls:
                ctrl = self.contrast_controls[ch_name]
                ctrl["max"] = slider_max
                ctrl["slider"].blockSignals(True)
                ctrl["slider"].setMaximum(slider_max)
                ctrl["slider"].blockSignals(False)
                self.auto_contrast_channel(ch_name)

    def auto_contrast_channel(self, ch_name):
        if ch_name not in self.contrast_controls:
            return
        ctrl = self.contrast_controls[ch_name]
        
        checked = self.bg_subtract_checkbox.isChecked()
        active_dict = self.images_dict if checked else self.raw_images_dict
        if ch_name not in active_dict:
            return
        img_data = active_dict[ch_name]

        vmin, vmax = auto_contrast_helper(img_data, lower=1.0, upper=99.5)
        slider_max = ctrl["max"]
        if slider_max > 0:
            vmin = (vmin / 255.0) * slider_max
            vmax = (vmax / 255.0) * slider_max

        vmin = max(0.0, float(vmin))
        vmax = min(slider_max, max(vmin + 0.001, float(vmax)))
        ctrl["slider"].setValue((vmin, vmax))

    def auto_contrast_all(self):
        for ch_name in self.channel_names:
            self.auto_contrast_channel(ch_name)

    def selected_threshold(self) -> float:
        return self.threshold

    def on_mouse_moved(self, evt):
        if self.seg_labels is None or self.seg_labels.size == 0:
            QToolTip.hideText()
            return

        pos = evt[0]
        rect = self.plot_widget.sceneBoundingRect()
        try:
            contains = rect.contains(pos)
        except TypeError:
            from PyQt6.QtCore import QPoint
            contains = rect.contains(pos.toPoint())

        if contains:
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            x = int(mouse_point.x())
            y = int(mouse_point.y())

            if 0 <= x < self.seg_labels.shape[1] and 0 <= y < self.seg_labels.shape[0]:
                cell_id = int(self.seg_labels[y, x])
                if cell_id > 0:
                    bead_info = self.cell_bead_counts.get(cell_id, {})
                    total_beads = sum(bead_info.values())
                    passing = cell_id in self.passing_cell_ids

                    status_text = (
                        f"<span style='color:{self.success_color}; font-weight:bold;'>Pass</span>"
                        if passing else
                        f"<span style='color:{self.danger_color}; font-weight:bold;'>Filtered Out</span>"
                    )

                    ratio = total_beads / self.num_proteins if self.num_proteins > 0 else 0.0

                    tooltip_lines = []
                    tooltip_lines.append(f"<b>Cell ID:</b> {cell_id}")
                    tooltip_lines.append(f"<b>Status:</b> {status_text}")
                    tooltip_lines.append(f"<b>Total Beads:</b> {total_beads}")
                    tooltip_lines.append(f"<b>Beads / Proteins Ratio:</b> {ratio:.2f} (Thresh: {self.threshold:.2f})")
                    tooltip_lines.append("<hr style='margin: 3px 0; border: none; border-top: 1px solid #555;'/>")
                    tooltip_lines.append("<b>Bead Counts per Protein:</b>")

                    if bead_info:
                        sorted_beads = sorted(bead_info.items(), key=lambda item: item[1], reverse=True)
                        for protein_name, count in sorted_beads:
                            tooltip_lines.append(f"• {protein_name}: <b>{count}</b>")
                    else:
                        tooltip_lines.append("<i>No beads in this cell</i>")

                    tooltip_text = (
                        f"<div style='font-family: sans-serif; font-size: 12px; line-height: 1.4; color: {self.text_color};'>"
                        + "<br/>".join(tooltip_lines)
                        + "</div>"
                    )
                    QToolTip.showText(QCursor.pos(), tooltip_text, self.plot_widget)
                else:
                    QToolTip.hideText()
            else:
                QToolTip.hideText()
        else:
            QToolTip.hideText()
