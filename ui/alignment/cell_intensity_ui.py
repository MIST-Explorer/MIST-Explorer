"""
Cell intensity UI module.
"""
import os
import typing

import numpy as np
import pandas as pd
# pylint: disable=no-name-in-module
from PyQt6.QtCore import QCoreApplication, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                             QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QSlider, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from core.project_naming import is_segmentation_channel
from ui.processing.segmentation_image_selector import SegmentationImageSelector

_SLIDER_STEPS = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30]
_SLIDER_DEFAULT_INDEX = 3  # 1.00


class ProteinDistributionDialog(QDialog):
    """Modal table showing distribution of bead counts across cells,
    with a threshold slider that filters cells by beads/proteins ratio."""

    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Protein Distribution")
        self.resize(360, 480)

        self._per_cell_counts = np.asarray(payload["per_cell_counts"])
        self._num_proteins = payload["num_proteins"]
        self._total_cells = payload["total_cells"]
        self._selected_threshold = _SLIDER_STEPS[_SLIDER_DEFAULT_INDEX]
        distribution = payload["distribution"]

        layout = QVBoxLayout(self)

        # --- distribution table ---
        table = QTableWidget(len(distribution), 2, self)
        table.setHorizontalHeaderLabels(["# Proteins", "% of Cells"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for row, (label, percentage) in enumerate(distribution):
            table.setItem(row, 0, QTableWidgetItem(str(label)))
            table.setItem(row, 1, QTableWidgetItem(f"{percentage:.2f}%"))

        layout.addWidget(table)

        # --- threshold slider ---
        threshold_label = QLabel(
            f"Filter threshold (beads / proteins): {_SLIDER_STEPS[_SLIDER_DEFAULT_INDEX]:.2f}"
        )
        layout.addWidget(threshold_label)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel(str(_SLIDER_STEPS[0])))
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setMinimum(0)
        slider.setMaximum(len(_SLIDER_STEPS) - 1)
        slider.setValue(_SLIDER_DEFAULT_INDEX)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)
        slider_row.addWidget(slider)
        slider_row.addWidget(QLabel(str(_SLIDER_STEPS[-1])))
        layout.addLayout(slider_row)

        self._remaining_label = QLabel()
        layout.addWidget(self._remaining_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # wire up slider
        slider.valueChanged.connect(
            lambda idx, lbl=threshold_label: self._on_slider_changed(idx, lbl)
        )
        self._on_slider_changed(_SLIDER_DEFAULT_INDEX, threshold_label)

    def selected_threshold(self) -> float:
        return self._selected_threshold

    def _on_slider_changed(self, idx: int, threshold_label: QLabel):
        threshold = _SLIDER_STEPS[idx]
        self._selected_threshold = threshold
        threshold_label.setText(f"Filter threshold (beads / proteins): {threshold:.2f}")

        if self._total_cells == 0 or self._num_proteins == 0:
            self._remaining_label.setText("Cells remaining after filter: —")
            return

        passing = int(np.sum(self._per_cell_counts / self._num_proteins >= threshold))
        pct = passing / self._total_cells * 100
        self._remaining_label.setText(
            f"Cells remaining after filter: {pct:.2f}%  ({passing} / {self._total_cells})"
        )


# pylint: disable=too-many-instance-attributes
class CellIntensityUI(QWidget):
    """
    UI for managing cell intensity analysis parameters.
    """
    errorSignal = pyqtSignal(str)
    emitBeadData = pyqtSignal(np.ndarray)
    emitColorCodes = pyqtSignal(dict)
    generate_cell_data = pyqtSignal()
    requestFilteredStats = pyqtSignal(int)
    thresholdApplied = pyqtSignal(float, int)
    testCrosstalkRequested = pyqtSignal()

    def __init__(self, parent=None, containing_layout: typing.Optional[QVBoxLayout] = None):
        super().__init__(parent)
        self.channel_rows = []
        self.channel_to_color_code = {}
        self.available_channels = []
        self.bead_data_file = None

        # Attributes initialized in setup_ui
        self.cell_intensity_groupbox = None
        self.main_layout = None
        self.components_widget = None
        self.cellintensity_components_vlayout = None
        self.image_selector = None
        self.bead_data_layout = None
        self.bead_data = None
        self.bead_data_label = None
        self.channels_layout = None
        self.add_channel_button = None
        self.radius_fg_layout = None
        self.radius_fg_label = None
        self.radius_fg = None
        self.radius_bg_layout = None
        self.radius_bg_label = None
        self.radius_bg = None
        self.crosstalk_layout = None
        self.crosstalk_label_layout = None
        self.crosstalk_label = None
        self.crosstalk_val_label = None
        self.crosstalk_slider = None
        self.test_crosstalk_button = None
        self.run_button = None
        self.cancel_button = None
        self.save_button = None
        self.filtered_stats_button = None
        self.filter_status_label = None
        self.channel_progress_label = None
        self._channels_total = 0
        self._channels_done = 0

        self.setup_ui(parent, containing_layout)

    def update_channels(self, channels: dict):
        """Update available channels in dropdowns."""
        valid_channels = {}
        for channel_name, wrapper in channels.items():
            if is_segmentation_channel(wrapper):
                continue
            valid_channels[channel_name] = wrapper

        self.available_channels = sorted(
            valid_channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
        )
        for row_widget in self.channel_rows:
            combo_box = row_widget.findChild(QComboBox)
            if combo_box:
                current_text = combo_box.currentText()
                combo_box.clear()
                combo_box.addItems(self.available_channels)
                if current_text in self.available_channels:
                    combo_box.setCurrentText(current_text)

    def setup_ui(self, parent, containing_layout: typing.Optional[QVBoxLayout]):
        """Initialize the UI components."""
        self.cell_intensity_groupbox = QGroupBox(parent)
        self.cell_intensity_groupbox.setObjectName("cell_intensity_groupbox")

        self.main_layout = QVBoxLayout(self.cell_intensity_groupbox)

        self.components_widget = QWidget()
        self.cellintensity_components_vlayout = QVBoxLayout(self.components_widget)

        # Image selector (no channel picker — channels are assigned per color code below)
        self.image_selector = SegmentationImageSelector(self.cell_intensity_groupbox)
        self.image_selector.hide_channel_selector()
        self.cellintensity_components_vlayout.addWidget(self.image_selector)

        # bead data
        self.bead_data_layout = QHBoxLayout()
        self.bead_data = QPushButton(self.cell_intensity_groupbox)
        self.bead_data.clicked.connect(self.load_bead_data)
        self.bead_data_label = QLabel()
        self.bead_data_layout.addWidget(self.bead_data)
        self.bead_data_layout.addWidget(self.bead_data_label)
        self.cellintensity_components_vlayout.addLayout(self.bead_data_layout)

        # Channels layout
        self.channels_layout = QVBoxLayout()
        self.cellintensity_components_vlayout.addLayout(self.channels_layout)

        self.add_channel_button = QPushButton()
        self.add_channel_button.clicked.connect(self.add_channel_row)
        self.cellintensity_components_vlayout.addWidget(self.add_channel_button)

        # radius fg
        self.radius_fg_layout = QHBoxLayout()
        self.radius_fg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_fg_layout.addWidget(self.radius_fg_label)
        self.radius_fg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_fg.setProperty("value", 2)
        self.radius_fg_layout.addWidget(self.radius_fg)
        self.cellintensity_components_vlayout.addLayout(self.radius_fg_layout)

        # radius bg
        self.radius_bg_layout = QHBoxLayout()
        self.radius_bg_label = QLabel(self.cell_intensity_groupbox)
        self.radius_bg_layout.addWidget(self.radius_bg_label)
        self.radius_bg = QSpinBox(self.cell_intensity_groupbox)
        self.radius_bg.setProperty("value", 6)
        self.radius_bg_layout.addWidget(self.radius_bg)
        self.cellintensity_components_vlayout.addLayout(self.radius_bg_layout)

        # crosstalk ratio
        self.crosstalk_layout = QVBoxLayout()
        
        self.crosstalk_label_layout = QHBoxLayout()
        self.crosstalk_label = QLabel(self.cell_intensity_groupbox)
        self.crosstalk_label.hide()
        self.crosstalk_val_label = QLabel("10:1", self.cell_intensity_groupbox)
        self.crosstalk_val_label.hide()
        self.crosstalk_label_layout.addWidget(self.crosstalk_label)
        self.crosstalk_label_layout.addWidget(self.crosstalk_val_label)
        self.crosstalk_layout.addLayout(self.crosstalk_label_layout)
        
        self.crosstalk_slider = QSlider(Qt.Orientation.Horizontal, self.cell_intensity_groupbox)
        self.crosstalk_slider.setRange(1, 100)
        self.crosstalk_slider.setValue(10)
        self.crosstalk_slider.valueChanged.connect(self._handle_crosstalk_ratio_changed)
        self.crosstalk_slider.hide()
        self.crosstalk_layout.addWidget(self.crosstalk_slider)
        
        self.test_crosstalk_button = QPushButton(self.cell_intensity_groupbox)
        self.test_crosstalk_button.clicked.connect(self._handle_test_crosstalk)
        self.crosstalk_layout.addWidget(self.test_crosstalk_button)
        
        self.cellintensity_components_vlayout.addLayout(self.crosstalk_layout)

        # run button
        self.run_button = QPushButton(self.cell_intensity_groupbox)
        self.run_button.clicked.connect(self._handle_generate_cell_data)
        self.run_button.setEnabled(False)
        self.cellintensity_components_vlayout.addWidget(self.run_button)

        # cancel button
        self.cancel_button = QPushButton(self.cell_intensity_groupbox)
        self.cellintensity_components_vlayout.addWidget(self.cancel_button)

        # save button (disabled until a run completes)
        self.save_button = QPushButton(self.cell_intensity_groupbox)
        self.save_button.setEnabled(False)
        self.cellintensity_components_vlayout.addWidget(self.save_button)

        # filtered bead stats button
        self.filtered_stats_button = QPushButton(self.cell_intensity_groupbox)
        self.filtered_stats_button.clicked.connect(self._show_filtered_bead_stats)
        self.cellintensity_components_vlayout.addWidget(self.filtered_stats_button)

        # filter status label (shown after applying threshold)
        self.filter_status_label = QLabel()
        self.filter_status_label.setVisible(False)
        self.cellintensity_components_vlayout.addWidget(self.filter_status_label)

        # channel progress label (shown while running)
        self.channel_progress_label = QLabel()
        self.channel_progress_label.setVisible(False)
        self.cellintensity_components_vlayout.addWidget(self.channel_progress_label)

        self.main_layout.addWidget(self.components_widget)

        if containing_layout is not None:
            containing_layout.addWidget(self.cell_intensity_groupbox)

        self._retranslate_ui()

    def _handle_generate_cell_data(self):
        if self.bead_data_file is not None:
            self.filter_status_label.setVisible(False)
            self._start_run(len(self.channel_to_color_code))
            self.emitBeadData.emit(self.bead_data_file)
            self.emitColorCodes.emit(self.channel_to_color_code)
            self.generate_cell_data.emit()
        else:
            self.errorSignal.emit("Please load bead data first.")

    def _start_run(self, total_channels: int):
        self._channels_total = total_channels
        self._channels_done = 0
        self.save_button.setEnabled(False)
        self.channel_progress_label.setText(f"0 / {total_channels} channels processed")
        self.channel_progress_label.setVisible(True)

    def on_channel_done(self, done: int, total: int):
        self._channels_done = done
        self._channels_total = total
        self.channel_progress_label.setText(f"{done} / {total} channels processed")
        if done >= total:
            self.save_button.setEnabled(True)

    def add_channel_row(self):
        """Add a new row for channel color code mapping."""
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)

        channel_combo = QComboBox()
        channel_combo.setEditable(True)
        if self.available_channels:
            channel_combo.addItems(self.available_channels)
        else:
            channel_combo.addItems([f"Channel {i+1}" for i in range(10)])

        load_button = QPushButton("Load Color Code")
        remove_button = QPushButton("Remove")

        row_layout.addWidget(channel_combo)
        row_layout.addWidget(load_button)
        row_layout.addWidget(remove_button)

        self.channels_layout.addWidget(row_widget)
        self.channel_rows.append(row_widget)

        load_button.clicked.connect(
            lambda: self.load_color_code_for_channel(channel_combo, load_button)
        )
        remove_button.clicked.connect(lambda: self.remove_channel_row(row_widget))

    def remove_channel_row(self, row_widget):
        """Remove a channel row."""
        channel_combo = row_widget.findChild(QComboBox)
        if channel_combo:
            channel_name = channel_combo.currentText()
            if channel_name in self.channel_to_color_code:
                del self.channel_to_color_code[channel_name]

        if row_widget in self.channel_rows:
            self.channel_rows.remove(row_widget)

        row_widget.deleteLater()

    def load_color_code_for_channel(self, channel_combo, button):
        """Load color code from file for a specific channel."""
        channel_name = channel_combo.currentText()
        if not channel_name:
            self.errorSignal.emit("Please specify a channel name.")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Open Color Code for {channel_name}",
            "",
            "Color Code(*.csv *.xlsx);;All Files (*)",
        )
        if file_name:
            try:
                if file_name.endswith(".csv"):
                    color_code = pd.read_csv(file_name)
                else:
                    color_code = pd.read_excel(file_name)
                self.channel_to_color_code[channel_name] = color_code

                base_name = os.path.basename(file_name)
                stem, ext = os.path.splitext(base_name)
                if len(stem) > 10:
                    button_text = f"{stem[:5]}...{stem[-5:]}{ext}"
                else:
                    button_text = base_name
                button.setText(button_text)
            except (UnicodeDecodeError, pd.errors.ParserError):
                self.errorSignal.emit("Please select a valid file type")

    def load_bead_data(self):
        """Load bead data from file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Bead Data", "", "Bead Data(*.csv *.xlsx);;All Files (*)"
        )
        if file_name:
            try:
                self.bead_data_file = (
                    pd.read_csv(file_name).to_numpy().astype("uint16")
                )
                self.bead_data_label.setText(os.path.basename(file_name))
                self.emitBeadData.emit(self.bead_data_file)
            except UnicodeDecodeError:
                self.errorSignal.emit("Please select a valid file type")

    def _show_filtered_bead_stats(self):
        if self.bead_data_file is None:
            self.errorSignal.emit("Please load bead data first.")
            return
        if not self.channel_to_color_code:
            self.errorSignal.emit("Please load at least one color code file first.")
            return
        num_proteins = max(len(df) for df in self.channel_to_color_code.values())
        self.emitBeadData.emit(self.bead_data_file)
        self.requestFilteredStats.emit(num_proteins)

    def set_generation_enabled(self, enabled: bool):
        """Enable generation only when a source image is selected in the image manager."""
        self.run_button.setEnabled(bool(enabled))

    def set_processing_buttons_enabled(self, enabled: bool):
        """Enable/disable filtered stats button during generation. Save is controlled by on_channel_done."""
        self.filtered_stats_button.setEnabled(bool(enabled))

    def show_protein_distribution(self, payload: object):
        dialog = ProteinDistributionDialog(payload, self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            threshold = dialog.selected_threshold()
            num_proteins = payload["num_proteins"]
            total_cells = payload["total_cells"]
            per_cell_counts = np.asarray(payload["per_cell_counts"])

            if total_cells > 0 and num_proteins > 0:
                passing = int(np.sum(per_cell_counts / num_proteins >= threshold))
            else:
                passing = 0

            self.filter_status_label.setText(
                f"{passing} cells remaining / {total_cells} total cells"
            )
            self.filter_status_label.setVisible(True)
            self.thresholdApplied.emit(threshold, num_proteins)

    def _handle_crosstalk_ratio_changed(self, value):
        self.crosstalk_val_label.setText(f"{value}:1")

    def _handle_test_crosstalk(self):
        self.testCrosstalkRequested.emit()

    def _retranslate_ui(self):
        _translate = QCoreApplication.translate
        self.cell_intensity_groupbox.setTitle(
            _translate("MainWindow", "Generate Protein Data of Cells")
        )

        self.add_channel_button.setText(_translate("MainWindow", "+ Add Channel"))
        self.bead_data_label.setText(_translate("MainWindow", "none selected"))
        self.bead_data.setText(_translate("MainWindow", "Open Bead Data"))
        self.radius_fg_label.setText(_translate("MainWindow", "Radius fg"))
        self.radius_bg_label.setText(_translate("MainWindow", "Radius bg"))
        self.crosstalk_label.setText(_translate("MainWindow", "Crosstalk Ratio Threshold"))
        self.test_crosstalk_button.setText(_translate("MainWindow", "Reduce Crosstalk"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
        self.filtered_stats_button.setText(_translate("MainWindow", "Filtered Bead Stats"))
