"""
Cell layer alignment UI module.
"""

# pylint: disable=no-name-in-module
from PyQt6.QtCore import QCoreApplication, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import CellLayerAligner, ImageStorage
from core.image_utils import window_image_by_contrast
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog
from ui.processing.segmentation_image_selector import SegmentationImageSelector


# pylint: disable=too-many-instance-attributes
class CellLayerAlignmentUI(QWidget):
    """
    UI for managing cell layer alignment parameters.
    """

    errorSignal = pyqtSignal(str)
    alignmentCompleteSignal = pyqtSignal(object, str)
    loadOnCanvasSignal = pyqtSignal(dict, bool, str)
    channelChanged = pyqtSignal(int)
    progress = pyqtSignal(int, str)

    def __init__(
        self,
        containing_layout: QVBoxLayout,
        storage: ImageStorage,
        parent=None,
    ):
        super().__init__(parent)

        self.target_image = None
        self.target_uuid = ""
        self.target_name = "not loaded"

        self.unaligned_image = None
        self.unaligned_uuid = ""
        self.unaligned_name = "not loaded"

        self.aligner = CellLayerAligner()
        self.storage = storage

        # Attributes initialized in setup methods
        self.alignment_groupbox = None
        self.main_layout = None
        self._target_groupbox = None
        self.target_image_selector = None
        self._unaligned_groupbox = None
        self.unaligned_image_selector = None
        self.unaligned_spacing_row = None
        self.unaligned_spacing_label = None
        self.unaligned_spacing_x = None
        self.unaligned_spacing_y = None
        self.fine_alignment_scale_row = None
        self.fine_alignment_scale_label = None
        self.fine_alignment_scale_input = None
        self.memory_estimate_label = None
        self.checkbox_layout = None
        self.need_centering = None
        self.need_gradient_descent = None
        self.target_use_contrasted_checkbox = None
        self.unaligned_use_contrasted_checkbox = None
        self.button_layout = None
        self.register_button = None
        self.manually_align_button = None

        self._setup_ui(parent, containing_layout)
        self._setup_connections()

    def _setup_ui(self, parent, containing_layout: QVBoxLayout):
        """Initialize the UI components."""
        self.alignment_groupbox = QGroupBox(parent)
        self.main_layout = QVBoxLayout(self.alignment_groupbox)

        self._setup_image_layouts()
        self._setup_spacing_layouts()
        self._setup_checkboxes()
        self._setup_buttons()

        self.main_layout.addWidget(self._target_groupbox)
        self.main_layout.addWidget(self._unaligned_groupbox)
        self.main_layout.addLayout(self.unaligned_spacing_row)
        self.main_layout.addLayout(self.fine_alignment_scale_row)
        self.main_layout.addWidget(self.memory_estimate_label)
        self.main_layout.addLayout(self.checkbox_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.button_layout)

        if containing_layout:
            containing_layout.addWidget(self.alignment_groupbox)

        self._retranslate_ui()

    def _setup_image_layouts(self):
        """Setup image selectors for target and moving images."""
        self._target_groupbox = QGroupBox("Protein Array Image", self)
        target_layout = QVBoxLayout(self._target_groupbox)
        target_layout.setContentsMargins(4, 4, 4, 4)
        self.target_image_selector = SegmentationImageSelector(self)
        target_layout.addWidget(self.target_image_selector)
        self.target_use_contrasted_checkbox = QCheckBox(
            "Use contrasted image (toolbar sliders)"
        )
        self.target_use_contrasted_checkbox.setChecked(False)
        self.target_use_contrasted_checkbox.setToolTip(
            "Use toolbar contrast settings for this image when computing the transform. "
            "The output channel always uses original pixel values."
        )
        target_layout.addWidget(self.target_use_contrasted_checkbox)

        self._unaligned_groupbox = QGroupBox("Aligned by this Image", self)
        unaligned_layout = QVBoxLayout(self._unaligned_groupbox)
        unaligned_layout.setContentsMargins(4, 4, 4, 4)
        self.unaligned_image_selector = SegmentationImageSelector(self)
        unaligned_layout.addWidget(self.unaligned_image_selector)
        self.unaligned_use_contrasted_checkbox = QCheckBox(
            "Use contrasted image (toolbar sliders)"
        )
        self.unaligned_use_contrasted_checkbox.setChecked(False)
        self.unaligned_use_contrasted_checkbox.setToolTip(
            "Use toolbar contrast settings for this image when computing the transform. "
            "The output channel always uses original pixel values."
        )
        unaligned_layout.addWidget(self.unaligned_use_contrasted_checkbox)

    def _setup_spacing_layouts(self):
        """Setup layouts for scaling factor input."""
        self.unaligned_spacing_row = QHBoxLayout()
        self.unaligned_spacing_label = QLabel("Scaling Factor (x, y):")
        self.unaligned_spacing_x = QLineEdit("1.0")
        self.unaligned_spacing_y = QLineEdit("1.0")
        self.unaligned_spacing_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_label)
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_x)
        self.unaligned_spacing_row.addWidget(QLabel(","))
        self.unaligned_spacing_row.addWidget(self.unaligned_spacing_y)

        self.fine_alignment_scale_row = QHBoxLayout()
        self.fine_alignment_scale_label = QLabel("Fine Alignment Scale:")
        self.fine_alignment_scale_input = QLineEdit("0.5")
        self.fine_alignment_scale_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.fine_alignment_scale_row.addWidget(self.fine_alignment_scale_label)
        self.fine_alignment_scale_row.addWidget(self.fine_alignment_scale_input)

        self.memory_estimate_label = QLabel(
            "Estimated extra memory during alignment: select images to calculate"
        )
        self.memory_estimate_label.setWordWrap(True)
        self.memory_estimate_label.setStyleSheet("color: gray;")

    def _setup_checkboxes(self):
        """Setup checkboxes for alignment options."""
        self.checkbox_layout = QHBoxLayout()
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.need_centering = QCheckBox("Skip Centering Images")
        self.need_centering.setChecked(True)
        self.need_centering.setToolTip(
            "Check this to center the images before alignment."
        )
        self.need_gradient_descent = QCheckBox("Skip Gradient Descent")
        self.need_gradient_descent.setChecked(False)
        self.need_gradient_descent.setToolTip(
            "Check this to skip the gradient descent step."
        )
        self.checkbox_layout.addWidget(self.need_centering)
        self.checkbox_layout.addWidget(self.need_gradient_descent)

    def _setup_buttons(self):
        """Setup registration buttons."""
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.register_button = QPushButton("Register Images")
        self.register_button.setEnabled(False)
        self.register_button.clicked.connect(self.register_images)
        self.register_button.setMinimumHeight(30)
        self.register_button.setStyleSheet(self._get_button_style())

        self.manually_align_button = QPushButton("Manually Register")
        self.manually_align_button.setEnabled(False)
        self.manually_align_button.setMinimumHeight(30)
        self.manually_align_button.setStyleSheet(self._get_button_style())
        self.manually_align_button.clicked.connect(self.manually_align_images)

        self.button_layout.addWidget(self.manually_align_button)
        self.button_layout.addWidget(self.register_button)

    def _get_button_style(self):
        """Return stylesheet for registration buttons."""
        from ui.theme import ThemeManager

        c = ThemeManager.instance().get_current()
        return f"""
            QPushButton {{
                background-color: {c["success"]};
                color: {c["text_on_accent"]};
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:disabled {{
                background-color: {c["bg_tertiary"]};
                color: {c["text_secondary"]};
            }}
            QPushButton:hover {{
                background-color: {c["success_hover"]};
            }}
        """

    def _setup_connections(self):
        """Set up the connections to the aligner thread"""
        self.aligner.progress.connect(self._handle_progress)
        self.aligner.error.connect(self._handle_error)

        self.target_image_selector.image_changed.connect(self._on_target_image_changed)
        self.target_image_selector.channel_changed.connect(
            lambda _channel: self._update_memory_estimate()
        )
        self.unaligned_image_selector.image_changed.connect(
            self._on_unaligned_image_changed
        )
        self.unaligned_image_selector.channel_changed.connect(
            lambda _channel: self._update_memory_estimate()
        )

        self.aligner.error.connect(self.errorSignal)

        self.unaligned_spacing_x.textChanged.connect(self._update_memory_estimate)
        self.unaligned_spacing_y.textChanged.connect(self._update_memory_estimate)
        self.fine_alignment_scale_input.textChanged.connect(
            self._update_memory_estimate
        )

        self.need_centering.stateChanged.connect(
            lambda state: self.aligner.skip_coarse_alignment(
                state == Qt.CheckState.Checked.value
            )
        )
        self.need_centering.stateChanged.connect(
            lambda _state: self._update_memory_estimate()
        )

        self.need_gradient_descent.stateChanged.connect(
            lambda state: self.aligner.skip_gradient_descent(
                state == Qt.CheckState.Checked.value
            )
        )
        self.need_gradient_descent.stateChanged.connect(
            lambda _state: self._update_memory_estimate()
        )

    def _on_target_image_changed(self, uuid: str):
        """Called when user picks a different reference image in the selector."""
        item = self.storage.get_data(uuid)
        if item is None:
            return
        self.target_uuid = uuid
        self.target_image = item["data"]
        self.target_name = item["name"]
        self.target_image_selector.set_channels(item["data"])
        self._check_can_register()
        self._update_memory_estimate()

    def _on_unaligned_image_changed(self, uuid: str):
        """Called when user picks a different moving image in the selector."""
        item = self.storage.get_data(uuid)
        if item is None:
            return
        self.unaligned_uuid = uuid
        self.unaligned_image = item["data"]
        self.unaligned_name = item["name"]
        self.unaligned_image_selector.set_channels(item["data"])
        self._check_can_register()
        self._update_memory_estimate()

    def _retranslate_ui(self):
        """Retranslate UI components."""
        _translate = QCoreApplication.translate
        self.alignment_groupbox.setTitle(
            _translate("MainWindow", "Register and Replace Selected Reference Channel")
        )
        self.register_button.setText(_translate("MainWindow", "Register Images"))
        self.manually_align_button.setText(
            _translate("MainWindow", "Manually Register")
        )

    # pylint: disable=unused-argument
    def set_target_image(self, item_uuid, is_leaf, channel):
        """Set the Reference image for alignment (called by context menu)."""
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        obj, name = item["data"], item["name"]
        self.target_uuid = item_uuid
        self.target_image = obj
        self.target_name = name
        # Update selector silently so context-menu picks don't re-trigger image_changed
        self.target_image_selector.set_selected_uuid_silent(item_uuid)
        self.target_image_selector.set_channels(obj)
        self.target_image_selector.set_current_channel_silent(f"Channel {channel + 1}")
        self._check_can_register()
        self._update_memory_estimate()

    # pylint: disable=unused-argument
    def set_unaligned_image(self, item_uuid, is_leaf, channel):
        """Set the unaligned image that will be registered to the target (called by context menu)."""
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        obj, name = item["data"], item["name"]
        self.unaligned_uuid = item_uuid
        self.unaligned_image = obj
        self.unaligned_name = name
        self.unaligned_image_selector.set_selected_uuid_silent(item_uuid)
        self.unaligned_image_selector.set_channels(obj)
        self.unaligned_image_selector.set_current_channel_silent(
            f"Channel {channel + 1}"
        )
        self._check_can_register()
        self._update_memory_estimate()

    def _check_can_register(self):
        """Check if both images are loaded and enable/disable register button"""
        can_register = (
            self.target_image is not None and self.unaligned_image is not None
        )
        self.register_button.setEnabled(can_register)
        self.manually_align_button.setEnabled(can_register)

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes using binary units for a compact UI estimate."""
        value = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TiB"

    def _selected_channel_array(self, image_dict, selector):
        """Return the currently selected channel array, or None if unavailable."""
        if image_dict is None:
            return None
        channel_name = selector.current_channel()
        wrapper = image_dict.get(channel_name)
        if wrapper is None:
            return None
        return wrapper.data

    def _update_memory_estimate(self):
        """Refresh the estimated peak memory subtext."""
        target_array = self._selected_channel_array(
            self.target_image, self.target_image_selector
        )
        moving_array = self._selected_channel_array(
            self.unaligned_image, self.unaligned_image_selector
        )
        if target_array is None or moving_array is None:
            self.memory_estimate_label.setText(
                "Estimated peak memory during alignment: select images to calculate"
            )
            return

        try:
            scale = (
                float(self.unaligned_spacing_x.text().strip()),
                float(self.unaligned_spacing_y.text().strip()),
            )
            fine_scale = float(self.fine_alignment_scale_input.text().strip())
            if min(*scale, fine_scale) <= 0:
                raise ValueError
        except ValueError:
            self.memory_estimate_label.setText(
                "Estimated peak memory during alignment: enter valid positive scale values"
            )
            return

        estimated_bytes = self.aligner.estimate_peak_memory_bytes(
            target_shape=target_array.shape,
            moving_shape=moving_array.shape,
            target_dtype=target_array.dtype,
            moving_dtype=moving_array.dtype,
            scale=scale,
            fine_scale=fine_scale,
            coarse_scale=self.aligner.coarse_scale,
            use_gradient_descent=not self.need_gradient_descent.isChecked(),
            skip_coarse_alignment=self.need_centering.isChecked(),
        )
        self.memory_estimate_label.setText(
            f"Estimated peak memory during alignment: ~{self._format_bytes(estimated_bytes)}"
        )

    def prepare_aligner(self):
        """Prepare the aligner with the current settings"""
        if self.target_image is None or self.unaligned_image is None:
            QMessageBox.warning(
                self,
                "Alignment Error",
                "Please load both target and unaligned images before alignment.",
            )
            return

        self.register_button.setEnabled(False)
        self.manually_align_button.setEnabled(False)

        try:
            scale = (
                float(self.unaligned_spacing_x.text().strip()),
                float(self.unaligned_spacing_y.text().strip()),
            )
            fine_scale = float(self.fine_alignment_scale_input.text().strip())
            if fine_scale <= 0:
                raise ValueError("fine_scale")
            self.aligner.set_scale(scale)
            self.aligner.set_fine_scale(fine_scale)
        except ValueError:
            self._handle_error(
                "Invalid scale values. Please enter numeric values for x, y, and fine alignment scale. Fine alignment scale must be greater than 0."
            )
            return

        target_ch_name = self.target_image_selector.current_channel()
        unaligned_ch_name = self.unaligned_image_selector.current_channel()

        self.aligner.set_target_image(
            self.target_image[target_ch_name].data,
            target_ch_name,
            self.target_uuid,
        )
        self.aligner.set_unaligned_image(
            self.unaligned_image[unaligned_ch_name].data,
            unaligned_ch_name,
            self.unaligned_uuid,
        )

        target_wrapper = self.target_image[target_ch_name]
        moving_wrapper = self.unaligned_image[unaligned_ch_name]
        target_reg = (
            window_image_by_contrast(
                target_wrapper.data,
                target_wrapper.contrast_min,
                target_wrapper.contrast_max,
            )
            if self.target_use_contrasted_checkbox.isChecked()
            else None
        )
        moving_reg = (
            window_image_by_contrast(
                moving_wrapper.data,
                moving_wrapper.contrast_min,
                moving_wrapper.contrast_max,
            )
            if self.unaligned_use_contrasted_checkbox.isChecked()
            else None
        )
        self.aligner.set_registration_images(target_reg, moving_reg)

    def register_images(self):
        """Start the image registration process"""
        if self.target_image is not None and self.unaligned_image is not None:
            self.prepare_aligner()
            self.aligner.progress.emit(0, "Starting alignment...")
            self.aligner.setStackSize(8 * 1024 * 1024)
            self.aligner.start()

    def manually_align_images(self):
        """Open a dialog for manual alignment of images"""
        if self.target_image is None or self.unaligned_image is None:
            QMessageBox.warning(
                self,
                "Alignment Error",
                "Please load both target and unaligned images before manual alignment.",
            )
            return
        self.prepare_aligner()
        target_ch_name = self.target_image_selector.current_channel()
        unaligned_ch_name = self.unaligned_image_selector.current_channel()

        target_wrapper = self.target_image[target_ch_name]
        moving_wrapper = self.unaligned_image[unaligned_ch_name]
        target_display = (
            window_image_by_contrast(
                target_wrapper.data,
                target_wrapper.contrast_min,
                target_wrapper.contrast_max,
            )
            if self.target_use_contrasted_checkbox.isChecked()
            else target_wrapper.data
        )
        moving_display = (
            window_image_by_contrast(
                moving_wrapper.data,
                moving_wrapper.contrast_min,
                moving_wrapper.contrast_max,
            )
            if self.unaligned_use_contrasted_checkbox.isChecked()
            else moving_wrapper.data
        )

        def progress_cb(value, message):
            self.progress.emit(value, message)
            QCoreApplication.processEvents()

        preview_dialog = AlignmentPreviewDialog(
            {
                "target_image": target_display,
                "aligned_image": moving_display,
            },
            progress_callback=progress_cb,
        )
        preview_dialog.transformation_ready.connect(self._on_alignment_ready)
        preview_dialog.exec()
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)

    def _on_alignment_ready(self, payload: dict) -> None:
        """Route transformation_ready payload from AlignmentPreviewDialog to the aligner."""
        matrix = payload["matrix"]
        action = payload.get("action", "add_layer")
        self.aligner.manually_align(matrix, action)

    def _handle_progress(self, value, message):
        """Handle progress updates from the aligner thread"""
        self.progress.emit(value, message)
        if value >= 100:
            self._handle_finished()

    def _handle_error(self, error_message):
        """Handle error messages from the aligner thread"""
        QMessageBox.critical(self, "Alignment Error", error_message)
        self.progress.emit(100, "Error occurred during alignment.")

    def _handle_finished(self):
        """Handle when the alignment thread finishes"""
        self.register_button.setEnabled(True)
        self.manually_align_button.setEnabled(True)
