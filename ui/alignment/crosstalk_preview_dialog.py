import os
import logging
import numpy as np
import qtrangeslider
from utils import resource_path
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QCursor
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QCheckBox,
    QSlider,
    QLabel,
    QFrame,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolTip,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QProgressBar,
)
from ui.theme import ThemeManager
from core.image_utils import auto_contrast_helper

logger = logging.getLogger(__name__)


class CrosstalkPreviewDialog(QDialog):
    """
    Dialog to preview crosstalk suppression results.
    Overlays multi-channel background-subtracted images in distinctive colors,
    and places red markers on coordinates where bead-channel signals are suppressed.
    """

    def __init__(self, parent, model, images_dict, bead_data, channel_names, ratio):
        super().__init__(parent)
        self.model = model
        self.images_dict = images_dict
        self.bead_data = bead_data
        self.channel_names = channel_names
        self.ratio = ratio

        # Dialog-level cache for suppression masks per ratio
        self.suppressed_mask_cache = {}

        # Initial calculate of crosstalk suppression
        self.suppressed_mask = self.get_suppressed_mask(self.ratio)

        # Pre-calculate bead intensities across channels
        self.calculate_bead_intensities()

        # Load raw (pre-background-subtracted) images
        self.raw_images_dict = self._get_raw_images()

        # Premium distinct color palette for channels (excluding red, which is for suppressed markers)
        self.channel_colors = [
            (0, 255, 255),    # Cyan
            (0, 255, 0),      # Green
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Yellow
            (0, 128, 255),    # Electric Blue
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Violet
        ]

        # Check if segmentation labels are available and compute mask of beads in cells
        self.in_cell_mask = None
        self.seg_labels = None

        if (
            hasattr(self.model, "segmentation_labels")
            and self.model.segmentation_labels is not None
            and self.model.segmentation_labels.size > 0
        ):
            self.seg_labels = self.model.segmentation_labels
        elif (
            hasattr(self.model, "source_uuid")
            and self.model.source_uuid
            and hasattr(self.model, "storage")
            and self.model.storage
        ):
            item = self.model.storage.get_data(self.model.source_uuid)
            if item is not None:
                data = item.get("data", {})
                from core.project_naming import is_segmentation_channel
                for ch_name, wrapper in data.items():
                    if is_segmentation_channel(wrapper):
                        if hasattr(wrapper, "data") and wrapper.data is not None:
                            self.seg_labels = np.asarray(wrapper.data, dtype=np.int32)
                            break

        if self.seg_labels is not None and self.bead_data is not None:
            coords = self.bead_data[:, 0:2].astype(int)
            x_limit, y_limit = self.seg_labels.shape[1], self.seg_labels.shape[0]
            in_bounds = (
                (coords[:, 0] >= 0) & (coords[:, 0] < x_limit)
                & (coords[:, 1] >= 0) & (coords[:, 1] < y_limit)
            )
            self.in_cell_mask = np.zeros(len(self.bead_data), dtype=bool)
            if np.any(in_bounds):
                cell_ids = self.seg_labels[coords[in_bounds, 1], coords[in_bounds, 0]]
                self.in_cell_mask[in_bounds] = cell_ids > 0

        self.image_items = {}
        self.max_intensities = {}
        self.contrast_controls = {}
        self.init_ui()

    def make_lut(self, color):
        """Generate a lookup table mapping intensity [0, 255] to a custom color with linear alpha."""
        r, g, b = color
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0] = np.linspace(0, r, 256)
        lut[:, 1] = np.linspace(0, g, 256)
        lut[:, 2] = np.linspace(0, b, 256)
        lut[:, 3] = np.linspace(0, 255, 256)  # Alpha scales from transparent to opaque
        return lut

    def init_ui(self):
        self.setWindowTitle("Crosstalk Suppression Viewer")
        self.resize(1150, 850)

        tc = ThemeManager.instance().get_current()
        self.bg_color = tc.get("bg_primary", "#0b0c10")
        self.card_color = tc.get("bg_secondary", "#1f2833")
        self.text_color = tc.get("text_primary", "#c5c6c7")
        self.accent_color = tc.get("accent", "#66fcf1")
        self.border_color = tc.get("border", "#2d3436")
        self.danger_color = tc.get("danger", "#e57373")

        # Set main dialog layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Resolve icon path for checkbox (matching UMAPPlot)
        icons_dir = resource_path(os.path.join("assets", "icons")).replace("\\", "/")
        mode = ThemeManager.instance().current_mode
        tick_icon = "checkbox_tick_dark.svg" if mode == "DARK" else "checkbox_tick_light.svg"
        tick_path = f"{icons_dir}/{tick_icon}"

        # Style the dialog and its child checkboxes (using UMAPPlot indicators)
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
            QCheckBox#BeadCheckbox, QCheckBox#CellBeadsCheckbox, QCheckBox#BgSubtractCheckbox {{
                font-size: 13px;
            }}
        """)

        # --- Content splitter (Plot on Left, Controls on Right) ---
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(8)

        # 1. Left canvas panel (PyQtGraph PlotWidget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(tc.get("canvas_bg", "#000000"))
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.getPlotItem().showGrid(x=False, y=False)

        # INVERT Y-AXIS so (0,0) is top-left and coordinates match standard image files
        self.plot_widget.getViewBox().invertY(True)

        # Add channels to PlotWidget
        for idx, ch_name in enumerate(self.channel_names):
            if ch_name not in self.images_dict:
                continue

            img_data = self.images_dict[ch_name]
            
            # Enable automatic downsampling to prevent aliasing when zoomed out, 
            # but keep smooth=False so pixels remain sharp when zoomed in.
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

            # Calculate auto contrast by default (percentile 1.0% to 99.5%)
            vmin, vmax = auto_contrast_helper(img_data, lower=1.0, upper=99.5)
            if slider_max > 0:
                vmin = (vmin / 255.0) * slider_max
                vmax = (vmax / 255.0) * slider_max

            vmin = max(0.0, float(vmin))
            vmax = min(slider_max, max(vmin + 0.001, float(vmax)))
            vmin = np.clip(vmin, 0.0, slider_max)
            vmax = np.clip(vmax, 0.0, slider_max)
            if vmax <= vmin:
                vmax = slider_max
                vmin = 0.0

            image_item.setImage(img_data)
            image_item.setLevels([vmin, vmax])

            # Add to plot and register
            self.plot_widget.addItem(image_item)
            self.image_items[ch_name] = image_item

        # 2. Add Red ScatterPlotItem for suppressed bead centers
        self.scatter_item = pg.ScatterPlotItem(hoverable=True, tip=None)
        self.scatter_item.setZValue(999)  # Overlay on top of all images
        self.scatter_item.sigHovered.connect(self.on_scatter_hovered)
        self.plot_widget.addItem(self.scatter_item)

        content_splitter.addWidget(self.plot_widget)

        # 3. Right Sidebar Control Panel
        controls_panel = QFrame()
        controls_panel.setObjectName("ControlsPanel")
        controls_panel.setMaximumWidth(320)
        controls_panel.setStyleSheet(f"""
            QFrame#ControlsPanel {{
                background-color: {self.card_color};
                border: 1px solid {self.border_color};
                border-radius: 10px;
            }}
        """)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(15)

        # --- Section A: Crosstalk Ratio Threshold Control ---
        threshold_title = QLabel("Crosstalk Ratio Threshold")
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

        row_thresh = QHBoxLayout()
        self.ratio_val_label = QLabel(f"Ratio: {int(self.ratio)}:1")
        self.ratio_val_label.setStyleSheet(f"color: {self.text_color}; font-weight: bold; font-size: 13px; border: none;")
        row_thresh.addWidget(self.ratio_val_label)
        row_thresh.addStretch()
        threshold_box_layout.addLayout(row_thresh)

        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(1, 100)
        self.ratio_slider.setValue(int(self.ratio))
        self.ratio_slider.setStyleSheet(f"""
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
        self.ratio_slider.valueChanged.connect(self.on_ratio_slider_changed)
        threshold_box_layout.addWidget(self.ratio_slider)

        controls_layout.addWidget(threshold_box)

        # --- Precalculate Ratios UI Card ---
        precalc_box = QFrame()
        precalc_box.setObjectName("PrecalcBox")
        precalc_box.setStyleSheet(f"""
            QFrame#PrecalcBox {{
                background-color: {self.bg_color};
                border: 1px solid {self.border_color};
                border-radius: 8px;
                padding: 10px;
                margin-top: 5px;
            }}
        """)
        precalc_layout = QVBoxLayout(precalc_box)
        precalc_layout.setSpacing(8)

        precalc_title = QLabel("Precalculate Ratios")
        precalc_title.setStyleSheet(f"color: {self.accent_color}; font-weight: bold; font-size: 12px; border: none;")
        precalc_layout.addWidget(precalc_title)

        range_layout = QHBoxLayout()
        range_layout.setSpacing(6)

        lbl_from = QLabel("From:")
        lbl_from.setStyleSheet(f"color: {self.text_color}; border: none; font-size: 11px;")
        self.spin_x1 = QSpinBox()
        self.spin_x1.setRange(1, 100)
        self.spin_x1.setValue(1)
        self.spin_x1.setStyleSheet(f"""
            QSpinBox {{
                background-color: {self.card_color};
                border: 1px solid {self.border_color};
                border-radius: 4px;
                padding: 3px;
                color: {self.accent_color};
                font-weight: bold;
                font-size: 11px;
            }}
            QSpinBox:focus {{
                border: 1.5px solid {self.accent_color};
            }}
        """)

        lbl_to = QLabel("To:")
        lbl_to.setStyleSheet(f"color: {self.text_color}; border: none; font-size: 11px;")
        self.spin_x2 = QSpinBox()
        self.spin_x2.setRange(1, 100)
        self.spin_x2.setValue(100)
        self.spin_x2.setStyleSheet(f"""
            QSpinBox {{
                background-color: {self.card_color};
                border: 1px solid {self.border_color};
                border-radius: 4px;
                padding: 3px;
                color: {self.accent_color};
                font-weight: bold;
                font-size: 11px;
            }}
            QSpinBox:focus {{
                border: 1.5px solid {self.accent_color};
            }}
        """)

        range_layout.addWidget(lbl_from)
        range_layout.addWidget(self.spin_x1)
        range_layout.addWidget(lbl_to)
        range_layout.addWidget(self.spin_x2)
        precalc_layout.addLayout(range_layout)

        # Precalculate Button
        self.precalc_btn = QPushButton("Precalculate Range")
        self.precalc_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.accent_color};
                border: 1px solid {self.accent_color};
                border-radius: 6px;
                padding: 5px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {self.accent_color};
                color: {tc.get("text_on_accent", "#0b0c10")};
            }}
            QPushButton:disabled {{
                border-color: {self.border_color};
                color: {self.border_color};
            }}
        """)
        self.precalc_btn.clicked.connect(self.on_precalculate_clicked)
        precalc_layout.addWidget(self.precalc_btn)

        # Progress Bar
        self.precalc_progress = QProgressBar()
        self.precalc_progress.setRange(0, 100)
        self.precalc_progress.setValue(0)
        self.precalc_progress.setTextVisible(True)
        self.precalc_progress.hide()
        self.precalc_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {self.border_color};
                border-radius: 4px;
                text-align: center;
                background-color: {self.bg_color};
                color: {self.text_color};
                font-size: 10px;
                height: 14px;
            }}
            QProgressBar::chunk {{
                background-color: {self.accent_color};
                border-radius: 3px;
            }}
        """)
        precalc_layout.addWidget(self.precalc_progress)

        controls_layout.addWidget(precalc_box)

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

        # Filtered beads toggle layer
        bead_toggle_layout = QHBoxLayout()
        self.bead_checkbox = QCheckBox("Filtered Beads (Red)")
        self.bead_checkbox.setObjectName("BeadCheckbox")
        self.bead_checkbox.setChecked(True)
        self.bead_checkbox.toggled.connect(self.scatter_item.setVisible)
        bead_toggle_layout.addWidget(self.bead_checkbox)

        self.bead_count_badge = QLabel()
        self.bead_count_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {self.danger_color};
                color: {tc.get("badge_fg", "#ffffff")};
                font-weight: bold;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }}
        """)
        bead_toggle_layout.addWidget(self.bead_count_badge)
        bead_toggle_layout.addStretch()
        controls_layout.addLayout(bead_toggle_layout)

        # Only show beads within cells toggle
        self.cell_beads_checkbox = QCheckBox("Only Beads Within Cells")
        self.cell_beads_checkbox.setObjectName("CellBeadsCheckbox")
        
        has_segmentation = self.seg_labels is not None and self.seg_labels.size > 0
        self.cell_beads_checkbox.setChecked(has_segmentation)
        self.cell_beads_checkbox.setEnabled(has_segmentation)
        if not has_segmentation:
            self.cell_beads_checkbox.setToolTip("Stardist segmentation labels not loaded.")
            
        self.cell_beads_checkbox.toggled.connect(self.on_cell_beads_toggled)
        controls_layout.addWidget(self.cell_beads_checkbox)

        # Background subtraction toggle
        self.bg_subtract_checkbox = QCheckBox("Subtract Background")
        self.bg_subtract_checkbox.setObjectName("BgSubtractCheckbox")
        self.bg_subtract_checkbox.setChecked(True)
        self.bg_subtract_checkbox.toggled.connect(self.on_bg_subtract_toggled)
        controls_layout.addWidget(self.bg_subtract_checkbox)

        # Auto Contrast All button
        bg_hover = tc.get("bg_hover", "rgba(255, 255, 255, 0.06)")
        bg_pressed = tc.get("bg_pressed", "rgba(255, 255, 255, 0.12)")
        self.auto_all_btn = QPushButton("Auto Contrast All")
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
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Populate channels slider groups
        bg_hover = tc.get("bg_hover", "rgba(255, 255, 255, 0.06)")
        bg_pressed = tc.get("bg_pressed", "rgba(255, 255, 255, 0.12)")

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
                    padding: 8px;
                }}
            """)
            channel_card_layout = QVBoxLayout(channel_card)
            channel_card_layout.setSpacing(6)

            # Row 1: Checkbox, Color Indicator, and Auto Contrast button
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

            # Small "Auto" contrast button
            auto_btn = QPushButton("Auto")
            auto_btn.setStyleSheet(auto_btn_style)
            auto_btn.clicked.connect(lambda _, ch=ch_name: self.auto_contrast_channel(ch))
            row1.addWidget(auto_btn)

            channel_card_layout.addLayout(row1)

            # Get current levels from the image_item
            levels = self.image_items[ch_name].levels
            vmin_curr, vmax_curr = levels[0], levels[1]
            slider_max = self.max_intensities[ch_name]

            # Row 2: Low Edit, Range Slider, High Edit
            row2 = QHBoxLayout()
            row2.setSpacing(6)

            con_lo_edit = QLineEdit(format_value(vmin_curr))
            con_lo_edit.setFixedWidth(50)
            con_lo_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            con_lo_edit.setToolTip("Click to type low contrast threshold")
            con_lo_edit.setStyleSheet(edit_style)

            contrast_slider = qtrangeslider.QDoubleRangeSlider(Qt.Orientation.Horizontal)
            contrast_slider.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
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
            con_hi_edit.setFixedWidth(50)
            con_hi_edit.setToolTip("Click to type high contrast threshold")
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

            # Setup connections
            # Contrast changed handler
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

        # Separator line above stats
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        sep3.setStyleSheet(f"background-color: {self.border_color}; max-height: 1px; border: none;")
        controls_layout.addWidget(sep3)

        # --- Statistics Dashboard Card (bottom of sidebar, above actions) ---
        stats_card = QFrame()
        stats_card.setObjectName("StatsCard")
        stats_card.setStyleSheet(f"""
            QFrame#StatsCard {{
                background-color: {self.bg_color};
                border: 1px solid {self.border_color};
                border-radius: 8px;
                padding: 10px;
            }}
            QLabel {{
                color: {self.text_color};
                font-size: 12px;
                border: none;
            }}
        """)
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setSpacing(6)
        stats_layout.setContentsMargins(10, 8, 10, 8)

        # Add stats items to card
        self.ratio_label = QLabel()
        self.beads_label = QLabel()
        self.supp_beads_label = QLabel()

        stats_layout.addWidget(self.ratio_label)
        stats_layout.addWidget(self.beads_label)
        stats_layout.addWidget(self.supp_beads_label)

        controls_layout.addWidget(stats_card)

        # --- Sidebar Actions (Reset Zoom, Apply & Exit, Cancel) ---
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)

        # Top row: Reset Zoom
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

        # Bottom row: Cancel / Apply & Exit
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

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

        apply_btn = QPushButton("Apply & Exit")
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

        # Add to splitter and configure stretches
        content_splitter.addWidget(controls_panel)
        content_splitter.setSizes([850, 300])
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(content_splitter)
        self.setLayout(main_layout)

        # Trigger rendering and stats calculations
        self.update_suppressed_layers()
        self.plot_widget.autoRange()

    def update_suppressed_layers(self):
        """Recalculate suppression mask and update scatter points / labels."""
        # Determine active subset of beads and suppressed mask
        if self.bead_data is not None:
            if self.cell_beads_checkbox.isChecked() and self.in_cell_mask is not None:
                active_beads = self.bead_data[self.in_cell_mask]
                active_mask = self.suppressed_mask[self.in_cell_mask] if self.suppressed_mask.size > 0 else self.suppressed_mask
            else:
                active_beads = self.bead_data
                active_mask = self.suppressed_mask
        else:
            active_beads = None
            active_mask = self.suppressed_mask

        total_beads = len(active_beads) if active_beads is not None else 0
        suppressed_beads_mask = np.any(active_mask, axis=1) if active_mask.size > 0 else np.array([], dtype=bool)
        suppressed_beads_count = int(np.sum(suppressed_beads_mask)) if suppressed_beads_mask.size > 0 else 0
        pct_beads = (suppressed_beads_count / total_beads * 100) if total_beads > 0 else 0.0

        # Update Header Card labels
        self.ratio_label.setText(f"Ratio Threshold: <b style='color:{self.accent_color};'>{int(self.ratio)}:1</b>")
        self.beads_label.setText(f"Total Beads: <b>{total_beads:,}</b>")
        self.supp_beads_label.setText(
            f"Filtered Beads: <b style='color:{self.danger_color};'>{suppressed_beads_count:,} ({pct_beads:.1f}%)</b>"
        )

        # Update Sidebar badge
        self.bead_count_badge.setText(f"{suppressed_beads_count} beads")

        # Update Scatter plot coordinates
        if active_beads is not None and suppressed_beads_mask.size > 0:
            suppressed_coords = active_beads[suppressed_beads_mask]
            
            # Map back to original bead indices
            if self.cell_beads_checkbox.isChecked() and self.in_cell_mask is not None:
                active_indices = np.where(self.in_cell_mask)[0]
            else:
                active_indices = np.arange(len(self.bead_data))
            suppressed_orig_indices = active_indices[suppressed_beads_mask]

            if len(suppressed_coords) > 0:
                self.scatter_item.setData(
                    x=suppressed_coords[:, 0],
                    y=suppressed_coords[:, 1],
                    size=7,
                    pen=pg.mkPen(color=(255, 0, 0, 255), width=1.2),
                    brush=pg.mkBrush(255, 0, 0, 160),
                    symbol='o',
                    data=suppressed_orig_indices.tolist(),
                )
            else:
                self.scatter_item.setData(x=np.array([]), y=np.array([]))
        else:
            self.scatter_item.setData(x=np.array([]), y=np.array([]))

    def get_suppressed_mask(self, ratio):
        """Get suppression mask from cache or calculate it."""
        ratio_key = round(float(ratio), 2)
        if ratio_key not in self.suppressed_mask_cache:
            _, mask = self.model.calculate_crosstalk_suppression(ratio)
            self.suppressed_mask_cache[ratio_key] = mask
        return self.suppressed_mask_cache[ratio_key]

    def on_ratio_slider_changed(self, value):
        """Handler for crosstalk suppression ratio threshold slider."""
        self.ratio = float(value)
        self.ratio_val_label.setText(f"Ratio: {value}:1")
        
        # Calculate new suppression mask using cache/model
        self.suppressed_mask = self.get_suppressed_mask(self.ratio)
        
        # Update layers and statistics
        self.update_suppressed_layers()

    def _get_raw_images(self):
        try:
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

    def on_bg_subtract_toggled(self, checked):
        """Handler for toggling background subtraction."""
        self.update_image_source()

    def update_image_source(self):
        """Update the displayed image items and contrast limits based on the background subtraction setting."""
        use_bg_subtoggled = self.bg_subtract_checkbox.isChecked()
        active_dict = self.images_dict if (use_bg_subtoggled or not self.raw_images_dict) else self.raw_images_dict

        for ch_name, image_item in self.image_items.items():
            if ch_name not in active_dict:
                continue
            
            img_data = active_dict[ch_name]
            image_item.setImage(img_data)
            
            # Recalculate max intensity and update contrast controls
            img_max = np.max(img_data)
            slider_max = float(img_max) if img_max > 0 else 1.0
            self.max_intensities[ch_name] = slider_max
            
            if ch_name in self.contrast_controls:
                ctrl = self.contrast_controls[ch_name]
                ctrl["max"] = slider_max
                
                # Block signals to prevent infinite feedback loop
                ctrl["slider"].blockSignals(True)
                ctrl["slider"].setMaximum(slider_max)
                ctrl["slider"].blockSignals(False)
                
                # Perform auto-contrast for this channel
                self.auto_contrast_channel(ch_name)

    def on_precalculate_clicked(self):
        """Precalculate and cache suppression masks for a range of ratio thresholds."""
        x1 = self.spin_x1.value()
        x2 = self.spin_x2.value()

        if x1 > x2:
            x1, x2 = x2, x1
            self.spin_x1.setValue(x1)
            self.spin_x2.setValue(x2)

        # Disable controls
        self.precalc_btn.setEnabled(False)
        self.spin_x1.setEnabled(False)
        self.spin_x2.setEnabled(False)

        self.precalc_progress.setFormat("%p%")
        self.precalc_progress.setValue(0)
        self.precalc_progress.show()

        total_steps = x2 - x1 + 1

        # Prime the model cache first with the first ratio
        self.get_suppressed_mask(float(x1))

        for idx, ratio_val in enumerate(range(x1, x2 + 1)):
            self.get_suppressed_mask(float(ratio_val))
            progress_pct = int((idx + 1) / total_steps * 100)
            self.precalc_progress.setValue(progress_pct)
            # Yield control back to Qt to process GUI updates and keep window responsive
            QCoreApplication.processEvents()

        # Re-enable controls
        self.precalc_btn.setEnabled(True)
        self.spin_x1.setEnabled(True)
        self.spin_x2.setEnabled(True)
        self.precalc_progress.setFormat("Precalculated!")

    def on_cell_beads_toggled(self, checked):
        """Handler for toggling 'Only show beads within cells'."""
        self.update_suppressed_layers()

    def auto_contrast_channel(self, ch_name):
        """Re-calculate and apply auto-contrast for a specific channel."""
        if ch_name not in self.contrast_controls:
            return
        ctrl = self.contrast_controls[ch_name]
        
        use_bg_subtoggled = getattr(self, "bg_subtract_checkbox", None) is None or self.bg_subtract_checkbox.isChecked()
        active_dict = self.images_dict if (use_bg_subtoggled or not self.raw_images_dict) else self.raw_images_dict
        
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
        
        vmin = np.clip(vmin, 0.0, slider_max)
        vmax = np.clip(vmax, 0.0, slider_max)
        if vmax <= vmin:
            vmax = slider_max
            vmin = 0.0
            
        ctrl["slider"].setValue((vmin, vmax))

    def auto_contrast_all(self):
        """Re-calculate and apply auto-contrast for all channels."""
        for ch_name in self.channel_names:
            self.auto_contrast_channel(ch_name)

    def on_contrast_changed(self, ch_name, val, label_widget):
        """Backward-compatible handler for contrast changes (used in tests)."""
        label_widget.setText(f"Contrast: {val}%")
        image_item = self.image_items[ch_name]
        img_max = self.max_intensities[ch_name]
        factor = val / 100.0
        max_level = max(img_max * factor, 0.001)
        image_item.setLevels([0, max_level])
        
        # If the new controls exist, update the slider/edits to match
        if ch_name in self.contrast_controls:
            ctrl = self.contrast_controls[ch_name]
            ctrl["slider"].blockSignals(True)
            ctrl["slider"].setValue((0.0, max_level))
            ctrl["slider"].blockSignals(False)
            
            def format_value(v):
                if v >= 100:
                    return f"{int(v)}"
                elif v >= 1:
                    return f"{v:.1f}"
                else:
                    return f"{v:.3f}"
            ctrl["lo_edit"].setText(format_value(0.0))
            ctrl["hi_edit"].setText(format_value(max_level))

    def calculate_bead_intensities(self):
        """Pre-calculate raw 5x5 median intensities for each bead in each channel."""
        num_beads = len(self.bead_data) if self.bead_data is not None else 0
        num_channels = len(self.channel_names)
        self.bead_intensities = np.zeros((num_beads, num_channels), dtype=float)
        if num_beads > 0 and num_channels > 0:
            bead_xs = self.bead_data[:, 0].astype(int)
            bead_ys = self.bead_data[:, 1].astype(int)
            
            for c_idx, ch_name in enumerate(self.channel_names):
                if ch_name in self.images_dict:
                    img = self.images_dict[ch_name]
                    height, width = img.shape
                    
                    # Pad by 2 to safely handle borders
                    padded_img = np.pad(img, pad_width=2, mode='constant', constant_values=0)
                    
                    # Coordinates in padded space
                    padded_xs = bead_xs + 2
                    padded_ys = bead_ys + 2
                    
                    valid_coords_mask = (bead_xs >= 0) & (bead_xs < width) & (bead_ys >= 0) & (bead_ys < height)
                    
                    # Meshgrid offsets for 5x5 neighborhood
                    dx, dy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
                    dx = dx.flatten()
                    dy = dy.flatten()
                    
                    neighbor_xs = padded_xs[:, np.newaxis] + dx
                    neighbor_ys = padded_ys[:, np.newaxis] + dy
                    
                    neighbor_xs = np.clip(neighbor_xs, 0, width + 3)
                    neighbor_ys = np.clip(neighbor_ys, 0, height + 3)
                    
                    neighborhoods = padded_img[neighbor_ys, neighbor_xs]
                    raw_medians = np.median(neighborhoods, axis=1)
                    
                    self.bead_intensities[:, c_idx] = np.where(valid_coords_mask, raw_medians, 0.0)

    def on_scatter_hovered(self, scatter_item, points, ev):
        """Show a rich tooltip showing coordinates, intensities and filtered status on hover."""
        if points is None or len(points) == 0:
            QToolTip.hideText()
            return

        point = points[0]
        orig_idx = point.data()
        if orig_idx is None:
            return

        x, y = self.bead_data[orig_idx, 0], self.bead_data[orig_idx, 1]

        tooltip_lines = []
        tooltip_lines.append(f"<b>Bead Coordinates:</b> ({x:.1f}, {y:.1f})")
        tooltip_lines.append("<hr style='margin: 3px 0; border: none; border-top: 1px solid #555;'/>")
        tooltip_lines.append("<b>Channel Intensities:</b>")

        for c_idx, ch_name in enumerate(self.channel_names):
            intensity = self.bead_intensities[orig_idx, c_idx]
            is_suppressed = self.suppressed_mask[orig_idx, c_idx]
            
            color = self.channel_colors[c_idx % len(self.channel_colors)]
            color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            status_text = ""
            if is_suppressed:
                status_text = f" <span style='color:{self.danger_color}; font-weight:bold;'>[Filtered]</span>"
            
            tooltip_lines.append(
                f"<span style='color:{color_hex};'>●</span> {ch_name}: <b>{intensity:.1f}</b>{status_text}"
            )

        tooltip_text = (
            f"<div style='font-family: sans-serif; font-size: 12px; line-height: 1.4; color: {self.text_color};'>"
            + "<br/>".join(tooltip_lines)
            + "</div>"
        )
        
        QToolTip.showText(QCursor.pos(), tooltip_text, self.plot_widget)
