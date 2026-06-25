"""Class to handle signal connections"""

import copy
import json
import logging
import os
import typing
import uuid

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import (
    CellIntensity,
    ImageGraphicsView,
    ImageStorage,
    ImageWrapper,
    ReferenceGraphicsView,
    Register,
    StarDist,
)
from core.project_manager import ProjectManager
from core.project_naming import is_segmentation_channel, is_temp_project_name
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog
from ui.alignment.alignment_view_dialog import AlignmentViewDialog
from ui.alignment.crop_anchor_dialog import CropAnchorDialog
from ui.alignment.crosstalk_preview_dialog import CrosstalkPreviewDialog
from ui.stardist.crop_experiment_dialog import CropExperimentDialog

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ui.app import MainWindow


# pylint: disable=too-many-instance-attributes
class Controller:
    """
    Controller class implements the Singleton pattern for managing the MIST-Explorer application.
    This class serves as the central controller for the image analysis application, coordinating
    between the UI, image processing models, and data storage. It handles image loading, saving,
    alignment operations, and user interactions.
    Attributes:
        _instance (Controller): Singleton instance of the Controller class
        image_count (int): Counter for loaded images, not really used in the current implementation
        model_canvas (ImageGraphicsView): Main image display canvas
        model_stardist (StarDist): StarDist model for image analysis
        model_register (Register): Registration model for image alignment
        model_cellIntensity (CellIntensity): Cell intensity analysis model
        reference_view (ReferenceGraphicsView): Reference image display view
        view (Ui_MainWindow): Main application UI window
        openFilesDialog (QFileDialog): File dialog for opening images
        signal_manager (SignalConnectionManager): Manages signal connections
        storage: Image data storage system
    Methods:
        __new__(cls, app): Creates or returns the singleton instance
        __init__(app): Initializes the controller with the main application window (app)
        handleError(error_message): Displays error messages to the user
        pixmap_to_image(pixmap): Converts QPixmap to numpy array for image processing
        controlSave(): Handles saving the current canvas image to file
        openFileDialog(viewer): Opens file dialog for image selection
        on_action_reference_triggered(): Handles reference image selection
        on_actionOpen_triggered(): Handles main image opening
        handle_new_image(data, file_name): Processes newly loaded images
        handle_new_reference_image(data, file_name): Processes newly loaded reference images
        _handle_aligned_image(aligned_data, target_small, aligned_small): Processes aligned images
        _show_preview_dialog(target_small, aligned_small): Shows alignment preview dialog
    Note:
        This class uses the Singleton pattern to ensure only one controller instance exists
        throughout the application lifecycle. It connects most UI interactions to function calls,
        there are some exceptions where the UI directly calls methods on the model. Perhaps those
        instances should also be handled by the controller.
    """

    _instance = None

    @classmethod
    def init(cls, app):
        """Initializes the controller."""
        if cls._instance is None:
            cls._instance = cls(app)

    @classmethod
    def get(cls):
        """Returns the singleton instance of the controller."""
        if cls._instance is None:
            raise RuntimeError("Controller not initialized")
        return cls._instance

    def __init__(self, app: "MainWindow"):
        self._initialized = True
        self.image_count = 0
        self.prev_tab_index = 0
        self.model_canvas = ImageGraphicsView(self)
        self.model_stardist = StarDist()
        self.model_register = Register()
        self.model_cell_intensity = CellIntensity()
        self.model_reference_canvas = ReferenceGraphicsView()
        self.view = app
        self.open_files_dialog = None
        self.view.images_tab.set_model_canvas(self.model_canvas)
        self.view.images_tab.set_model_stardist(self.model_stardist)
        self.view.images_tab.set_model_reference_canvas(self.model_reference_canvas)
        self.view.images_tab.finalize_badge_connections()
        self.signal_manager = SignalConnectionManager(self)
        self.signal_manager.setup_all_connections()
        self.storage = self.view.images_tab.storage
        self._apply_project_context()

        # Handle initial arguments
        initial_args = app.args
        if initial_args.image is not None:
            self.model_canvas.add_to_canvas(initial_args.image)
        if initial_args.reference is not None:
            self.model_reference_canvas.add_to_canvas(initial_args.reference)

        # self.need_preview_alignment.connect(self._handle_aligned_image)
        aligner = self.view.cell_layer_alignment.aligner
        aligner.manual_ready.connect(self._handle_manual_registration)
        aligner.aligned_image_signal.connect(self._handle_registration_image)
        self.model_register.alignment_complete.connect(self._handle_alignment_image)
        self.model_register.error.connect(self.handle_error)

    def handle_error(self, error_message):
        """Displays an error message."""
        QMessageBox.critical(self.view, "Error", error_message)

    def _apply_project_context(self):
        project_path = getattr(self.view, "current_project_path", None)
        project_name = None
        is_temp_project = False
        if project_path is not None:
            metadata = ProjectManager.load_project(project_path)
            if metadata is not None and metadata.name:
                project_name = metadata.name
                is_temp_project = bool(metadata.is_temporary) or is_temp_project_name(
                    project_name
                )
            else:
                project_name = project_path.stem
                is_temp_project = is_temp_project_name(project_name)
        self.model_stardist.set_project_context(project_name, is_temp_project)
        self.model_cell_intensity.set_project_context(project_name, is_temp_project)

    def pixmap_to_image(self, pixmap: QPixmap):
        """Converts a QPixmap to a numpy array."""
        if pixmap is None:
            raise ValueError("No pixmap provided")
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape((height, width, 4))  # 4 for RGBA

        # Convert from BGRA to RGB by dropping alpha channel and reversing BGR
        if arr.shape[2] == 4:  # If the image has an alpha channel
            arr = arr[:, :, :3]  # Remove the alpha channel

        # Convert BGR to RGB (OpenCV uses BGR, but most other libraries use RGB)
        arr = arr[:, :, ::-1]

        return arr

    def control_save(self):
        """Saves the current canvas image."""
        img = self.model_canvas.image_wrapper.data
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # qimage = pm.toImage()
        if img is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
            )
            if file_name:
                logger.debug(file_name)
                Image.fromarray(img).save(file_name)
                return True

            return False

        self.handle_error("No image in canvas, please load image")
        return False

    def open_file_dialog(self, viewer):
        """Opens a file dialog to select an image."""
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)"
        )
        if file_name:
            viewer.add_to_canvas(file_name)

    def on_action_reference_triggered(self):
        """Handles the reference image open action."""
        self.open_file_dialog(self.model_reference_canvas)

    def on_action_open_triggered(self):
        """Handles the main image open action."""
        self.open_file_dialog(self.model_canvas)

    def _store_uploaded_image(self, data, file_name, metadata=None) -> str:
        """Persist a newly uploaded image in storage/sidebar and return its UUID."""
        storage_item = {}
        storage_item["name"] = os.path.basename(file_name)
        storage_item["original_filename"] = file_name
        storage_item["metadata"] = metadata
        self.image_count += 1

        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.view.images_tab.add_item(my_uuid)
        # Auto-save a path reference whenever the user uploads a real file
        if os.path.isfile(file_name):
            self.view.images_tab.save_all_images(copy_data=False)
        return my_uuid

    # add new image to storage
    def handle_new_image(self, data, file_name, metadata=None):
        """Handles a new image by adding it to storage."""
        my_uuid = Controller._store_uploaded_image(self, data, file_name, metadata)
        # Set the active canvas UUID after selectors have refreshed from the new tree row.
        self.model_canvas.set_uuid(my_uuid)

    def handle_new_reference_image(self, data, file_name):
        """Handles a new reference image by adding it to storage."""
        my_uuid = Controller._store_uploaded_image(self, data, file_name)
        self.model_reference_canvas.set_uuid(my_uuid)
        # Reference uploads do not emit a dedicated UUID-changed signal, so
        # update the selector directly after the new image exists in the model.
        self.view.register_groupbox.reference_selector.follow_canvas_uuid(str(my_uuid))

    @staticmethod
    def _apply_matrix_to_all_layers(
        matrix: np.ndarray, item: dict, target_shape
    ) -> dict:
        """Apply a 2x3 affine matrix to every channel of item["data"]."""
        _cv2_ok = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
        results = {}
        for layer_key, wrapper in item["data"].items():
            img = wrapper.data
            original_dtype = img.dtype
            if img.dtype not in _cv2_ok:
                img = img.astype(np.float32)
            out_h, out_w = target_shape[:2] if target_shape is not None else img.shape[:2]
            # Controller receives a forward map (Moving→Target) from UI/Aligner,
            # which is what cv2.warpAffine expects by default.
            warped = cv2.warpAffine(img, matrix, (out_w, out_h))
            results[layer_key] = (
                warped.astype(original_dtype) if warped.dtype != original_dtype else warped
            )
        return results

    def _commit_registration(
        self, aligned_data: dict, matrix: np.ndarray, action: str
    ) -> None:
        """Apply matrix and add the result to canvas. Shared by manual and auto paths."""
        item = self.storage.get_data(aligned_data["uuid"])
        assert item is not None, "Aligned image data not found in storage"
        warped_layers = Controller._apply_matrix_to_all_layers(
            matrix, item, aligned_data.get("target_shape")
        )
        if action == "replace_channel":
            target_uuid = aligned_data.get("target_uuid", "")
            target_channel = aligned_data.get("target_channel", "")
            target_item = self.storage.get_data(target_uuid) if target_uuid else None
            if target_item is None or not target_channel:
                return  # target not found — nothing to do
            out_data = copy.deepcopy(target_item["data"])
            out_data[target_channel] = ImageWrapper(
                warped_layers.get(aligned_data.get("layer", "")), target_channel
            )
            self.model_canvas.add_to_canvas(
                out_data, True, "Registered_" + target_item["name"]
            )
            return
        layer = aligned_data["layer"]
        data = copy.deepcopy(item["data"])
        data[layer].data = warped_layers.get(layer, next(iter(warped_layers.values())))
        self.model_canvas.add_to_canvas(data, True, "Registered_" + item["name"])

    def _handle_manual_registration(self, aligned_data: dict) -> None:
        """Handle manual registration: apply matrix directly, no dialog."""
        self._commit_registration(
            aligned_data, aligned_data["matrix"], aligned_data.get("action", "add_layer")
        )

    def _handle_registration_image(
        self, aligned_data: dict, target_small: np.ndarray, aligned_small: np.ndarray
    ) -> None:
        """Handle auto cell-layer alignment: show dialog, compose matrices, then commit."""
        snapshot = {"target_image": target_small, "aligned_image": aligned_small}
        pre_matrix = aligned_data["matrix"]

        def on_accepted(payload):
            fine_matrix = payload["matrix"]
            pre_3x3 = np.vstack([pre_matrix, [0, 0, 1]])
            fine_3x3 = np.vstack([fine_matrix, [0, 0, 1]])
            combined = (fine_3x3 @ pre_3x3)[:2, :]
            self._commit_registration(aligned_data, combined, payload.get("action", "add_layer"))

        def progress_cb(value, message):
            self.view.update_progress_bar(value, message)
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

        preview_dialog = AlignmentPreviewDialog(snapshot, progress_callback=progress_cb)
        preview_dialog.transformation_ready.connect(on_accepted)
        preview_dialog.exec()

    def _handle_alignment_image(
        self, aligned_data: dict, target_small: np.ndarray, aligned_small: np.ndarray
    ) -> None:
        """Handle registered-arrays result: show AlignmentViewDialog for confirmation."""
        snapshot = {"target_image": target_small, "aligned_image": aligned_small}
        layer = aligned_data["layer"]  # list of channel keys

        def on_accepted(_payload):
            item = self.storage.get_data(aligned_data["uuid"])
            assert item is not None, "Aligned image data not found in storage"
            assert len(aligned_data["data"].keys()) == len(layer), (
                "Aligned data keys do not match the expected layers"
            )
            out_data = {k: ImageWrapper(aligned_data["data"][k], k) for k in layer}
            self.model_canvas.add_to_canvas(out_data, True, "Registered_" + item["name"])

        def progress_cb(value, message):
            self.view.update_progress_bar(value, message)
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

        preview_dialog = AlignmentViewDialog(snapshot, progress_callback=progress_cb)
        preview_dialog.moving_image_changed.connect(on_accepted)
        preview_dialog.exec()

    def _show_preview_dialog(self, target_small, aligned_small):
        """Show the preview dialog with red/green overlay"""
        # Use the downscaled images for the preview dialog
        snapshot = {
            "target_image": target_small,
            "aligned_image": aligned_small,
            "metadata": {
                "preview_target_shape": target_small.shape,
                "preview_aligned_shape": aligned_small.shape,
            },
        }
        def progress_cb(value, message):
            self.view.update_progress_bar(value, message)
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()

        preview_dialog = AlignmentPreviewDialog(snapshot, progress_callback=progress_cb)
        result = preview_dialog.exec()
        return result == 1 and preview_dialog.result_accepted

    def need_canvas_change(self, new_index):
        """Handles the canvas change when the tab is changed."""
        if self.prev_tab_index != 0 and new_index == 0:
            # self.view.canvas.pixmap_item.show()
            self.view.canvas.show_images_tab_image()
            if self.model_canvas.uuid:
                logger.debug("swapping channel")
                self.model_canvas.swap_channel(self.model_canvas.current_channel)
            else:
                logger.debug("clearing canvas")
                self.model_canvas.clear_canvas()
        elif self.prev_tab_index == 0 and new_index != 0:
            self.view.canvas.show_view_tab_image()
            self.view.view_tab.process_images()
            # self.view.canvas.pixmap_item.hide()

    def handle_tab_change(self, index):
        """Handles the tab change."""
        self.need_canvas_change(index)
        self.prev_tab_index = index

        # Update ROI button visibility based on tab
        # Show on View (1) and Analyze (2) tabs, hide on Extract (0)
        if hasattr(self.view, "canvas"):
            should_show_buttons = index in (1, 2)
            self.view.canvas.set_buttons_visible(should_show_buttons)

    def update_small_view_visibility(self, *args):
        """Updates the visibility of the small view (reference view)."""
        # args can be ignored (either index from tab change or pixmap from update_reference)
        index = self.view.stacked_widget.currentIndex()
        is_images_tab = index == 0
        has_content = not self.view.small_view.is_empty()
        self.view.small_view.setVisible(is_images_tab and has_content)

    def handle_cancel_registration(self):
        """Cancels the registration."""
        self.model_register.cancel()


# pylint: disable=too-few-public-methods
class SignalConnectionManager:
    """Manages signal connections"""

    def __init__(self, controller: Controller):
        self.c = controller
        self.blur_timer = None
        self._seg_source_uuid = None
        self._seg_source_channel = None

    @staticmethod
    def _next_available_channel_key(channel_map: dict) -> str:
        next_index = 0
        for key in channel_map:
            if not isinstance(key, str) or not key.startswith("Channel "):
                continue
            try:
                idx = int(key.replace("Channel ", ""))
            except ValueError:
                continue
            next_index = max(next_index, idx)
        return f"Channel {next_index + 1}"

    def _resolve_stardist_channel_key(self, channel_map: dict) -> str:
        for channel_key, wrapper in channel_map.items():
            if is_segmentation_channel(wrapper):
                return channel_key
        return self._next_available_channel_key(channel_map)

    def _persist_stardist_result_to_source(
        self, source_uuid: str, stardist_wrapper: ImageWrapper, label_name: str
    ) -> bool:
        item = self.c.storage.get_data(str(source_uuid))
        if item is None:
            return False
        channel_map = item.get("data")
        if not isinstance(channel_map, dict):
            return False

        target_channel = self._resolve_stardist_channel_key(channel_map)
        wrapped = ImageWrapper(
            stardist_wrapper.data,
            name=label_name,
            cmap="label_image",
        )
        wrapped.contrast_min = 0
        wrapped.contrast_max = 255
        self.c.storage.update_data(
            str(source_uuid),
            target_channel,
            wrapped,
        )
        return True

    def _on_overlay_build_finished(self):
        """If the overlay failed (still disabled after build), uncheck the button."""
        if not self.c.model_canvas.stardist_overlay_enabled:
            btn = self.c.view.stardist_groupbox.overlay_toggle_button
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            # Manually sync button text since blockSignals prevented toggled from firing
            btn.setText("Show Overlay")

    def _handle_stardist_done(
        self, stardist_wrapper: ImageWrapper, _success: bool, label_name: str
    ):
        source_uuid = self.c.model_stardist.last_result_source_uuid
        current_uuid = self.c.model_canvas.uuid

        if source_uuid is None or str(current_uuid) == str(source_uuid):
            self.c.model_canvas.load_segmentation_labels(stardist_wrapper, label_name)
            self.c.model_cell_intensity.load_segmentation_labels(stardist_wrapper)
            self.c.view.stardist_groupbox.overlay_toggle_button.setChecked(False)
            self._auto_save_image_to_project(str(current_uuid))
            return

        persisted = self._persist_stardist_result_to_source(
            str(source_uuid), stardist_wrapper, label_name
        )
        if not persisted:
            logger.warning(
                "Discarded StarDist result for source UUID %s; source image no longer exists.",
                source_uuid,
            )
            return
        self.c.view.stardist_groupbox.overlay_toggle_button.setChecked(False)
        self._auto_save_image_to_project(str(source_uuid))

    def _auto_save_image_to_project(self, item_uuid: str):
        images_tab = self.c.view.images_tab
        if not images_tab._can_save_project():
            return
        item = self.c.storage.get_data(item_uuid)
        if item is None:
            return
        images_tab._save_image_to_project(item_uuid, item)

    def setup_all_connections(self):
        """Set up all signal-slot connections with full IntelliSense support"""
        self._setup_alignment_connections()
        self._setup_menubar_connections()
        self._setup_toolbar_connections()
        self._setup_img_handling_conns()
        self._setup_canvas_connections()
        self._setup_crop_connections()
        self._setup_transform_connections()
        self._setup_stardist_connections()
        self._setup_registration_connections()
        self._setup_cell_intensity_conns()
        self._setup_img_broadcast_conns()
        self._setup_misc_connections()
        self._setup_segmentation_selector_connections()
        self._setup_cell_layer_alignment_selector_connections()

    def _setup_alignment_connections(self):
        """Alignment section signal connections"""
        self.c.view.cell_layer_alignment.alignmentCompleteSignal.connect(
            self.c.handle_new_image
        )
        self.c.view.cell_layer_alignment.loadOnCanvasSignal.connect(
            self.c.model_canvas.add_to_canvas
        )
        self.c.view.cell_layer_alignment.progress.connect(
            self.c.view.update_progress_bar
        )

    def _setup_menubar_connections(self):
        """Menu bar signal connections"""
        self.c.view.menu_bar.open_reference.triggered.connect(
            self.c.on_action_reference_triggered
        )
        self.c.view.menu_bar.open_image.triggered.connect(
            self.c.on_action_open_triggered
        )

    def _setup_toolbar_connections(self):
        """Toolbar signal connections"""
        self.c.view.tool_bar.actionReset.triggered.connect(
            self.c.model_canvas.reset_image
        )
        self.c.view.tool_bar.channelChanged.connect(self.c.model_canvas.swap_channel)
        self.c.view.tool_bar.contrast_slider.valueChanged.connect(
            self.c.model_canvas.update_contrast
        )
        self.c.view.tool_bar.cmapChanged.connect(self.c.model_canvas.update_image)
        self.c.view.tool_bar.auto_contrast_button.clicked.connect(
            self.c.model_canvas.auto_contrast
        )
        self.c.view.tool_bar.tabChanged.connect(
            self.c.view.stacked_widget.setCurrentIndex
        )
        self.c.view.tool_bar.tabChanged.connect(self.c.handle_tab_change)

    def _setup_img_handling_conns(self):
        """Image loading and display connections"""
        self.c.view.canvas.image_dropped.connect(self.c.model_canvas.add_to_canvas)
        self.c.view.small_view.image_dropped.connect(
            self.c.model_reference_canvas.add_to_canvas
        )
        self.c.view.images_tab.image_tree_view.image_dropped.connect(
            self.c.model_canvas.add_to_canvas
        )
        self.c.model_reference_canvas.update_reference.connect(
            self.c.view.small_view.display
        )
        self.c.model_reference_canvas.update_reference.connect(
            self.c.update_small_view_visibility
        )
        self.c.view.small_view.channel_changed.connect(
            self.c.model_reference_canvas.on_channel_changed
        )
        self.c.model_canvas.update_canvas.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.update_channel.connect(
            self.c.view.tool_bar.setChannelSelector
        )

        self.c.view.images_tab.storage.data_changed.connect(
            self.c.view.images_tab.set_channel_icon
        )
        self.c.view.view_tab.change_pix.connect(
            self.c.view.canvas.update_view_tab_canvas
        )
        self.c.view.view_tab.update_contrast_sig.connect(
            self.c.view.canvas.update_layer_levels
        )
        self.c.view.view_tab.update_layer_cmap_sig.connect(
            self.c.view.canvas.update_layer_cmap
        )

        self.c.view.view_tab.update_layer_opacity_sig.connect(
            self.c.view.canvas.update_layer_opacity
        )
        self.c.view.view_tab.update_layer_visible_sig.connect(
            self.c.view.canvas.update_layer_visibility
        )
        self.c.view.view_tab.reset_view_tab.connect(self.c.view.canvas.reset_view_tab)

        self.c.view.view_tab.export_png_sig.connect(self.c.view.canvas.save_as_png)
        self.c.view.view_tab.export_tif_sig.connect(self.c.view.canvas.save_as_tif)
        # self.c.model_canvas.canvas_updated.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.update_manager.connect(self.c.handle_new_image)
        self.c.model_reference_canvas.update_manager.connect(
            self.c.handle_new_reference_image
        )

    def _setup_canvas_connections(self):
        """Canvas-related signal connections"""
        self.c.model_canvas.update_progress.connect(self.c.view.update_progress_bar)
        self.c.model_canvas.error_signal.connect(self.c.handle_error)
        self.c.view.canvas.show_crop.connect(self.c.model_canvas.crop)
        self.c.model_canvas.update_cmap.connect(
            self.c.view.tool_bar.update_cmap_selector
        )
        self.c.model_canvas.change_slider.connect(
            self.c.view.tool_bar.update_contrast_slider
        )
        self.c.model_canvas.fill_metadata.connect(self.c.view.get_metadata)

    def _setup_crop_connections(self):
        """Crop operation connections"""
        self.c.view.crop_groupbox.crop_button.triggered.connect(
            self.c.view.canvas.start_crop_mode
        )
        self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
            self.c.view.canvas.cancel_crop_mode
        )

    def _setup_transform_connections(self):
        """Image transformation connections"""
        # Flip operations
        self.c.view.canvas.horizontal_flip.connect(self.c.model_canvas.flip_horizontal)
        self.c.view.canvas.vertical_flip.connect(self.c.model_canvas.flip_vertical)

        # Rotation
        self.c.view.rotate_groupbox.rotate_confirm.pressed.connect(
            lambda: self.c.model_canvas.rotate_image(
                self.c.view.rotate_groupbox.rotate_line_edit.text()
            )
        )

        # Gaussian blur
        self.c.view.gaussian_blur.confirm.clicked.connect(
            lambda: self.c.model_canvas.blur_layer(
                self.c.view.gaussian_blur.slider.value(),
                confirm=True,
                source_uuid=self._seg_source_uuid,
                source_channel=self._seg_source_channel,
            )
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.view.gaussian_blur.update_slider_label
        )
        self.blur_timer = QTimer()
        self.blur_timer.setSingleShot(True)
        self.blur_timer.setInterval(300)  # ms of inactivity before applying

        # Connect the timer timeout to the actual blur update
        self.blur_timer.timeout.connect(
            lambda: self.c.model_canvas.blur_layer(
                self.c.view.gaussian_blur.slider.value(),
                source_uuid=self._seg_source_uuid,
                source_channel=self._seg_source_channel,
            )
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            lambda _: self.blur_timer.start()
        )

    def _setup_stardist_connections(self):
        """StarDist-related connections"""
        # Parameter connections
        self.c.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(
            self.c.model_stardist.set_channel
        )
        self.c.view.stardist_groupbox.segmentation_method_selector.currentTextChanged.connect(
            self.c.model_stardist.set_segmentation_method
        )
        self.c.model_stardist.cell_image_set.connect(
            self.c.view.stardist_groupbox.set_groupbox_title
        )
        self.c.model_stardist.cell_channel.connect(
            self.c.view.stardist_groupbox.stardist_channel_selector.setCurrentText
        )
        self.c.view.stardist_groupbox.stardist_pretrained_models.currentTextChanged.connect(
            self.c.model_stardist.set_model
        )
        self.c.view.stardist_groupbox.percentile_high.valueChanged.connect(
            self.c.model_stardist.set_percentile_high
        )
        self.c.view.stardist_groupbox.percentile_low.valueChanged.connect(
            self.c.model_stardist.set_percentile_low
        )
        self.c.view.stardist_groupbox.prob_threshold.valueChanged.connect(
            self.c.model_stardist.set_prob_thresh
        )
        self.c.view.stardist_groupbox.nms_threshold.valueChanged.connect(
            self.c.model_stardist.set_nms_thresh
        )
        self.c.view.stardist_groupbox.scale_factor.valueChanged.connect(
            self.c.model_stardist.set_scale
        )
        self.c.view.stardist_groupbox.n_tiles.valueChanged.connect(
            self.c.model_stardist.set_num_tiles
        )
        self.c.view.stardist_groupbox.enable_dilation.toggled.connect(
            self.c.model_stardist.set_enable_dilation
        )
        self.c.view.stardist_groupbox.radius.valueChanged.connect(
            self.c.model_stardist.set_dialation_radisu
        )
        self.c.view.stardist_groupbox.use_contrasted_checkbox.toggled.connect(
            self.c.model_stardist.set_use_contrasted_image
        )
        self.c.view.stardist_groupbox.min_size.valueChanged.connect(
            self.c.model_stardist.set_min_size
        )
        self.c.view.stardist_groupbox.max_size.valueChanged.connect(
            self.c.model_stardist.set_max_size
        )
        self.c.view.stardist_groupbox.overlay_toggle_button.toggled.connect(
            self.c.model_canvas.set_stardist_overlay_enabled
        )
        self.c.model_canvas.overlay_build_started.connect(
            lambda: self.c.view.stardist_groupbox.overlay_toggle_button.setEnabled(
                False
            )
        )
        self.c.model_canvas.overlay_build_finished.connect(
            lambda: self.c.view.stardist_groupbox.overlay_toggle_button.setEnabled(True)
        )
        self.c.model_canvas.overlay_build_finished.connect(
            self._on_overlay_build_finished
        )

        # CellProfiler-like advanced settings
        cp = self.c.view.stardist_groupbox.cp_advanced
        cp.threshold_method.currentTextChanged.connect(
            self.c.model_stardist.set_threshold_method
        )
        cp.threshold_scope.currentTextChanged.connect(
            self.c.model_stardist.set_threshold_scope
        )
        cp.threshold_correction_factor.valueChanged.connect(
            self.c.model_stardist.set_threshold_correction_factor
        )
        cp.threshold_smoothing_scale.valueChanged.connect(
            self.c.model_stardist.set_threshold_smoothing_scale
        )
        cp.threshold_lower_bound.valueChanged.connect(
            self.c.model_stardist.set_threshold_lower_bound
        )
        cp.threshold_upper_bound.valueChanged.connect(
            self.c.model_stardist.set_threshold_upper_bound
        )
        cp.manual_threshold.valueChanged.connect(
            self.c.model_stardist.set_manual_threshold
        )
        cp.otsu_class.currentTextChanged.connect(
            self.c.model_stardist.set_two_class_otsu
        )
        cp.otsu_middle.currentTextChanged.connect(
            self.c.model_stardist.set_assign_middle_to_foreground
        )
        cp.object_fraction.valueChanged.connect(
            self.c.model_stardist.set_object_fraction
        )
        cp.lower_outlier_fraction.valueChanged.connect(
            self.c.model_stardist.set_lower_outlier_fraction
        )
        cp.upper_outlier_fraction.valueChanged.connect(
            self.c.model_stardist.set_upper_outlier_fraction
        )
        cp.averaging_method.currentTextChanged.connect(
            self.c.model_stardist.set_averaging_method
        )
        cp.variance_method.currentTextChanged.connect(
            self.c.model_stardist.set_variance_method
        )
        cp.number_of_deviations.valueChanged.connect(
            self.c.model_stardist.set_number_of_deviations
        )
        cp.adaptive_window_size.valueChanged.connect(
            self.c.model_stardist.set_adaptive_window_size
        )
        cp.fill_holes_thresholding.toggled.connect(
            self.c.model_stardist.set_fill_holes_after_thresholding
        )
        cp.fill_holes_declumping.toggled.connect(
            self.c.model_stardist.set_fill_holes_after_declumping
        )
        cp.automatic_smoothing.toggled.connect(
            self.c.model_stardist.set_automatic_smoothing
        )
        cp.smoothing_filter_size.valueChanged.connect(
            self.c.model_stardist.set_smoothing_filter_size
        )
        cp.automatic_maxima_suppression.toggled.connect(
            self.c.model_stardist.set_automatic_maxima_suppression
        )
        cp.maxima_suppression_size.valueChanged.connect(
            self.c.model_stardist.set_maxima_suppression_size
        )
        cp.low_res_maxima.toggled.connect(self.c.model_stardist.set_low_res_maxima)
        cp.exclude_border_objects.toggled.connect(
            self.c.model_stardist.set_exclude_border_objects
        )
        cp.invert_image.toggled.connect(self.c.model_stardist.set_invert_image)
        cp.crop_experiment_button.clicked.connect(self._open_crop_experiment)
        cp.save_preset_button.clicked.connect(self._save_cp_preset)
        cp.load_preset_button_adv.clicked.connect(self._load_cp_preset)
        self.c.view.stardist_groupbox.load_preset_button.clicked.connect(
            self._load_cp_preset
        )

        # Execution and results
        self.c.view.stardist_groupbox.stardist_run_button.pressed.connect(
            self.c.model_stardist.start
        )
        self.c.model_stardist.stardist_done.connect(self._handle_stardist_done)
        self.c.model_stardist.error_signal.connect(self.c.handle_error)
        self.c.model_stardist.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stardist_groupbox.cancel_button.clicked.connect(
            self.c.model_stardist.cancel
        )

    def _open_crop_experiment(self):
        """Open the Crop & Experiment dialog for CellProfiler-like segmentation."""
        stardist = self.c.model_stardist
        with stardist._state_lock:
            params = dict(stardist.params)
            channels = stardist.protein_channels

        wrapper = channels.get(params["channel"]) if channels else None
        if wrapper is None:
            self.c.handle_error("No cell channel loaded")
            return

        cell_image = wrapper.data
        contrast_min = getattr(wrapper, "contrast_min", 0)
        contrast_max = getattr(wrapper, "contrast_max", 255)

        current_settings = {
            "threshold_method": params.get("threshold_method", "MCT"),
            "threshold_scope": params.get("threshold_scope", "Global"),
            "threshold_smoothing_scale": float(
                params.get("threshold_smoothing_scale", 1.3488)
            ),
            "threshold_correction_factor": float(
                params.get("threshold_correction_factor", 1.0)
            ),
            "threshold_range": (
                float(params.get("threshold_lower_bound", 0.0)),
                float(params.get("threshold_upper_bound", 1.0)),
            ),
            "manual_threshold": float(params.get("manual_threshold", 0.5)),
            "two_class_otsu": bool(params.get("two_class_otsu", True)),
            "assign_middle_to_foreground": bool(
                params.get("assign_middle_to_foreground", True)
            ),
            "object_fraction": float(params.get("object_fraction", 0.2)),
            "lower_outlier_fraction": float(params.get("lower_outlier_fraction", 0.05)),
            "upper_outlier_fraction": float(params.get("upper_outlier_fraction", 0.05)),
            "averaging_method": params.get("averaging_method", "Mean"),
            "variance_method": params.get("variance_method", "Standard deviation"),
            "number_of_deviations": float(params.get("number_of_deviations", 2.0)),
            "adaptive_window_size": int(params.get("adaptive_window_size", 50)),
            "fill_holes_after_thresholding": bool(
                params.get("fill_holes_after_thresholding", True)
            ),
            "fill_holes_after_declumping": bool(
                params.get("fill_holes_after_declumping", True)
            ),
            "automatic_smoothing": bool(params.get("automatic_smoothing", True)),
            "smoothing_filter_size": float(params.get("smoothing_filter_size", 10.0)),
            "automatic_maxima_suppression": bool(
                params.get("automatic_maxima_suppression", True)
            ),
            "maxima_suppression_size": float(
                params.get("maxima_suppression_size", 7.0)
            ),
            "low_res_maxima": bool(params.get("low_res_maxima", True)),
            "exclude_border_objects": bool(params.get("exclude_border_objects", True)),
            "invert_image": bool(params.get("invert_image", False)),
        }

        dialog = CropExperimentDialog(
            cell_image,
            contrast_min,
            contrast_max,
            current_settings,
            min_size=int(params.get("min_size", 60)),
            max_size=int(params.get("max_size", 180)),
            enable_dilation=bool(params.get("enable_dilation", True)),
            dilation_radius=int(params.get("radius", 5)),
            use_contrasted_image=bool(params.get("use_contrasted_image", False)),
        )
        dialog.settings_accepted.connect(self._apply_experiment_settings)
        dialog.exec()

    def _apply_experiment_settings(self, settings: dict):
        """Apply settings from the Crop & Experiment dialog back to the UI widgets."""
        cp = self.c.view.stardist_groupbox.cp_advanced
        sg = self.c.view.stardist_groupbox

        cp.threshold_method.setCurrentText(settings.get("threshold_method", "MCT"))
        cp.threshold_scope.setCurrentText(settings.get("threshold_scope", "Global"))
        cp.threshold_correction_factor.setValue(
            float(settings.get("threshold_correction_factor", 1.0))
        )
        cp.threshold_smoothing_scale.setValue(
            float(settings.get("threshold_smoothing_scale", 1.3488))
        )
        cp.threshold_lower_bound.setValue(
            float(settings.get("threshold_lower_bound", 0.0))
        )
        cp.threshold_upper_bound.setValue(
            float(settings.get("threshold_upper_bound", 1.0))
        )
        cp.manual_threshold.setValue(float(settings.get("manual_threshold", 0.5)))

        two_class = settings.get("two_class_otsu", True)
        cp.otsu_class.setCurrentText("Two classes" if two_class else "Three classes")
        fg = settings.get("assign_middle_to_foreground", True)
        cp.otsu_middle.setCurrentText("Foreground" if fg else "Background")

        cp.object_fraction.setValue(float(settings.get("object_fraction", 0.2)))
        cp.lower_outlier_fraction.setValue(
            float(settings.get("lower_outlier_fraction", 0.05))
        )
        cp.upper_outlier_fraction.setValue(
            float(settings.get("upper_outlier_fraction", 0.05))
        )
        cp.averaging_method.setCurrentText(settings.get("averaging_method", "Mean"))
        cp.variance_method.setCurrentText(
            settings.get("variance_method", "Standard deviation")
        )
        cp.number_of_deviations.setValue(
            float(settings.get("number_of_deviations", 2.0))
        )
        cp.adaptive_window_size.setValue(int(settings.get("adaptive_window_size", 50)))
        cp.fill_holes_thresholding.setChecked(
            bool(settings.get("fill_holes_after_thresholding", True))
        )
        cp.fill_holes_declumping.setChecked(
            bool(settings.get("fill_holes_after_declumping", True))
        )
        cp.automatic_smoothing.setChecked(
            bool(settings.get("automatic_smoothing", True))
        )
        cp.smoothing_filter_size.setValue(
            float(settings.get("smoothing_filter_size", 10.0))
        )
        cp.automatic_maxima_suppression.setChecked(
            bool(settings.get("automatic_maxima_suppression", True))
        )
        cp.maxima_suppression_size.setValue(
            float(settings.get("maxima_suppression_size", 7.0))
        )
        cp.low_res_maxima.setChecked(bool(settings.get("low_res_maxima", True)))
        cp.exclude_border_objects.setChecked(
            bool(settings.get("exclude_border_objects", True))
        )

        sg.min_size.setValue(int(settings.get("min_size", 60)))
        sg.max_size.setValue(int(settings.get("max_size", 180)))
        sg.enable_dilation.setChecked(bool(settings.get("enable_dilation", True)))
        sg.radius.setValue(int(settings.get("dilation_radius", 5)))
        sg.use_contrasted_checkbox.setChecked(
            bool(settings.get("use_contrasted_image", False))
        )

    def _gather_cp_settings(self) -> dict:
        """Collect all CellProfiler-like settings from UI widgets into a dict."""
        cp = self.c.view.stardist_groupbox.cp_advanced
        sg = self.c.view.stardist_groupbox
        return {
            "threshold_method": cp.threshold_method.currentText(),
            "threshold_scope": cp.threshold_scope.currentText(),
            "threshold_correction_factor": cp.threshold_correction_factor.value(),
            "threshold_smoothing_scale": cp.threshold_smoothing_scale.value(),
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
            "min_size": sg.min_size.value(),
            "max_size": sg.max_size.value(),
            "enable_dilation": sg.enable_dilation.isChecked(),
            "dilation_radius": sg.radius.value(),
            "use_contrasted_image": sg.use_contrasted_checkbox.isChecked(),
        }

    def _save_cp_preset(self):
        """Save current CellProfiler-like settings to a JSON file."""
        settings = self._gather_cp_settings()
        path, _ = QFileDialog.getSaveFileName(
            self.c.view, "Save Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)

    def _load_cp_preset(self):
        """Load CellProfiler-like settings from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self.c.view, "Load Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                settings = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.c.handle_error(f"Failed to load preset: {e}")
            return
        self._apply_experiment_settings(settings)

    def _setup_registration_connections(self):
        """Registration-related connections"""
        # Parameters
        self.c.view.register_groupbox.overlap.valueChanged.connect(
            self.c.model_register.set_overlap
        )
        self.c.view.register_groupbox.max_size.valueChanged.connect(
            self.c.model_register.set_max_size
        )
        self.c.view.register_groupbox.num_tiles.valueChanged.connect(
            self.c.model_register.set_num_tiles
        )
        self.c.view.register_groupbox.ncc_threshold.valueChanged.connect(
            lambda v: self.c.model_register.set_ncc_threshold(v / 100)
        )

        # Execution
        self.c.view.register_groupbox.find_anchor_button.clicked.connect(
            self._open_crop_anchor_dialog
        )
        self.c.view.register_groupbox.run_button.clicked.connect(
            self.c.model_register.run_registration
        )
        self.c.model_register.cell_image_signal.connect(
            self.c.model_stardist.load_cell_image
        )
        self.c.model_register.protein_signal_arr_signal.connect(
            self.c.model_cell_intensity.load_protein_signal_array
        )
        self.c.model_register.progress.connect(self.c.view.update_progress_bar)
        self.c.view.register_groupbox.cancel_button.clicked.connect(
            self.c.handle_cancel_registration
        )

        # Image selectors
        self.c.view.register_groupbox.image_selector.image_changed.connect(
            self._on_register_image_selected
        )
        self.c.view.register_groupbox.image_selector.channel_changed.connect(
            self.c.model_register.set_moving_channel_key
        )
        self.c.view.register_groupbox.reference_selector.image_changed.connect(
            self._on_register_reference_image_selected
        )
        self.c.view.register_groupbox.reference_selector.channel_changed.connect(
            self.c.model_register.set_reference_channel_key
        )
        tree_model = self.c.view.images_tab.image_tree_model
        tree_model.rowsInserted.connect(lambda *_: self._repopulate_register_selector())
        tree_model.rowsRemoved.connect(lambda *_: self._repopulate_register_selector())
        tree_model.dataChanged.connect(lambda *_: self._repopulate_register_selector())
        self._repopulate_register_selector()

        # Results

    def _repopulate_register_selector(self):
        """Rebuild both Alignment tab image selector lists."""
        tree = self.c.view.images_tab.image_tree_model
        self.c.view.register_groupbox.image_selector.refresh_images(tree)
        self.c.view.register_groupbox.reference_selector.refresh_images(tree)

    def _on_register_image_selected(self, uuid: str) -> None:
        """Called when user picks a moving image in the Alignment tab selector."""
        storage = self.c.view.images_tab.storage
        img_data = storage.get_data(uuid)
        if img_data is None:
            return
        storage.add_data("alignment_moving_uuid", {"value": uuid})
        channels = img_data["data"]
        self.c.model_register.update_moving_image(channels)
        self.c.view.register_groupbox.image_selector.set_channels(channels)
        self.c.model_register.set_moving_channel_key(
            self.c.view.register_groupbox.image_selector.current_channel()
        )

    def _on_register_reference_image_selected(self, uuid: str) -> None:
        """Called when user picks a reference image in the Alignment tab selector."""
        storage = self.c.view.images_tab.storage
        img_data = storage.get_data(uuid)
        if img_data is None:
            return
        storage.add_data("alignment_ref_uuid", {"value": uuid})
        channels = img_data["data"]
        self.c.model_register.update_reference_channels(channels)
        self.c.view.register_groupbox.reference_selector.set_channels(channels)
        self.c.model_register.set_reference_channel_key(
            self.c.view.register_groupbox.reference_selector.current_channel()
        )

    def _open_crop_anchor_dialog(self):
        """Open the auto crop-anchor finder for the selected moving/reference pair.

        Reference = decoding cycle 1 (smaller); its top-left patch is located within
        the moving protein image, which is then rotate-then-cropped at that anchor.
        """
        storage = self.c.view.images_tab.storage
        mov_entry = storage.get_data("alignment_moving_uuid")
        ref_entry = storage.get_data("alignment_ref_uuid")
        if not mov_entry or not ref_entry:
            self.c.handle_error(
                "Select both a Moving (protein) and Reference image in Align Arrays first."
            )
            return
        mov_uuid = mov_entry["value"]
        ref_uuid = ref_entry["value"]
        mov_item = storage.get_data(mov_uuid)
        ref_item = storage.get_data(ref_uuid)
        if mov_item is None or ref_item is None:
            self.c.handle_error("Selected images are no longer available.")
            return

        key = "Channel 1"
        mov_data = mov_item["data"]
        ref_data = ref_item["data"]
        if key not in mov_data or key not in ref_data:
            self.c.handle_error(f"Both images need a '{key}' (brightfield) channel.")
            return
        mov_ch = mov_data[key].data
        ref_ch = ref_data[key].data

        dialog = CropAnchorDialog(
            ref_ch, mov_ch, ref_shape=ref_ch.shape, parent=self.c.view
        )
        dialog.transform_ready.connect(
            lambda payload: self.c.model_canvas.crop_with_transform(
                mov_uuid,
                payload["T"],
                payload["crop_w"],
                payload["crop_h"],
                source_name=mov_item.get("name"),
            )
        )
        dialog.exec()

    def _setup_cell_intensity_conns(self):
        """Cell intensity-related connections"""

        self.c.view.images_tab.image_tree_view.protein_data.connect(
            self.c.model_cell_intensity.load_protein_signal_array_from_storage
        )
        self.c.view.images_tab.image_tree_view.segmentation_label.connect(
            self.c.model_cell_intensity.load_segmentation_labels_from_storage
        )
        self.c.view.images_tab.image_tree_view.generation_source_selected.connect(
            self._handle_generation_source_selected
        )

        self.c.view.cell_intensity_groupbox.emitBeadData.connect(
            self.c.model_cell_intensity.set_bead_data
        )
        self.c.view.cell_intensity_groupbox.emitColorCodes.connect(
            self.c.model_cell_intensity.set_color_codes
        )
        self.c.view.cell_intensity_groupbox.radius_fg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_fg
        )
        self.c.view.cell_intensity_groupbox.radius_bg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_bg
        )
        self.c.view.cell_intensity_groupbox.crosstalk_slider.valueChanged.connect(
            self.c.model_cell_intensity.set_crosstalk_ratio
        )
        self.c.view.cell_intensity_groupbox.testCrosstalkRequested.connect(
            self._handle_test_crosstalk_requested
        )
        self.c.view.cell_intensity_groupbox.generate_cell_data.connect(
            self.c.model_cell_intensity.generate_cell_intensity_table
        )
        self.c.model_cell_intensity.error_signal.connect(self.c.handle_error)
        self.c.view.cell_intensity_groupbox.save_button.clicked.connect(
            self.c.model_cell_intensity.save_cell_data
        )
        self.c.model_cell_intensity.progress.connect(self.c.view.update_progress_bar)
        self.c.view.cell_intensity_groupbox.cancel_requested.connect(
            self.c.model_cell_intensity.cancel
        )
        self.c.view.cell_intensity_groupbox.model = self.c.model_cell_intensity
        self.c.view.cell_intensity_groupbox.requestFilteredStats[int].connect(
            self.c.model_cell_intensity.get_filtered_bead_count
        )
        self.c.model_cell_intensity.protein_distribution_ready.connect(
            self.c.view.cell_intensity_groupbox.show_protein_distribution
        )
        self.c.view.cell_intensity_groupbox.thresholdApplied.connect(
            self.c.model_cell_intensity.set_bead_per_protein_threshold
        )

        # Disable save/filtered-stats buttons during generation
        self.c.view.cell_intensity_groupbox.generate_cell_data.connect(
            lambda: self.c.view.cell_intensity_groupbox.set_processing_buttons_enabled(
                False
            )
        )
        self.c.model_cell_intensity.progress.connect(self._on_cell_intensity_progress)
        self.c.model_cell_intensity.channel_done.connect(
            self.c.view.cell_intensity_groupbox.on_channel_done
        )

        self.c.view.cell_intensity_groupbox.set_generation_enabled(False)

        # Image selector
        self.c.view.cell_intensity_groupbox.image_selector.image_changed.connect(
            self._on_generation_image_selected
        )
        tree_model = self.c.view.images_tab.image_tree_model
        tree_model.rowsInserted.connect(
            lambda *_: self._repopulate_generation_selector()
        )
        tree_model.rowsRemoved.connect(
            lambda *_: self._repopulate_generation_selector()
        )
        tree_model.dataChanged.connect(
            lambda *_: self._repopulate_generation_selector()
        )
        self._repopulate_generation_selector()

    def _on_cell_intensity_progress(self, value: int, msg: str):
        """Re-enable save/filtered-stats buttons when generation completes."""
        if value >= 100:
            self.c.view.cell_intensity_groupbox.set_processing_buttons_enabled(True)
            if msg == "Cell Data is Generated":
                QMessageBox.information(
                    self.c.view,
                    "Quantification Complete",
                    "Cell quantification generation finished successfully."
                )

    def _handle_test_crosstalk_requested(self):
        ui = self.c.view.cell_intensity_groupbox
        model = self.c.model_cell_intensity
        
        # Sync latest data from UI to model
        if ui.bead_data_file is not None:
            model.set_bead_data(ui.bead_data_file)
        if ui.channel_to_color_code:
            model.set_color_codes(ui.channel_to_color_code)

        if getattr(model, "bead_data", None) is None:
            self.c.handle_error("Please load bead data first.")
            return
        if not model.channel_to_color_code:
            self.c.handle_error("Please load at least one color code file first.")
            return

        try:
            ratio = model.params.get("crosstalk_ratio", 10.0)
            
            # Retrieve background-subtracted images
            images_dict = model._get_background_subtracted_images()
            
            dialog = CrosstalkPreviewDialog(
                parent=self.c.view,
                model=model,
                images_dict=images_dict,
                bead_data=model.bead_data,
                channel_names=list(model.channel_to_color_code.keys()),
                ratio=ratio,
            )
            if dialog.exec() == 1:
                final_ratio = dialog.ratio
                model.set_crosstalk_ratio(final_ratio)
                ui.crosstalk_slider.setValue(int(final_ratio))
                ui.crosstalk_val_label.setText(f"{int(final_ratio)}:1")
        except Exception as e:
            logger.error("Crosstalk ratio test failed: %s", e)
            self.c.handle_error(f"Error testing crosstalk: {str(e)}")

    def _repopulate_generation_selector(self):
        """Rebuild the Generation tab image selector list."""
        self.c.view.cell_intensity_groupbox.image_selector.refresh_images(
            self.c.view.images_tab.image_tree_model
        )

    def _on_generation_image_selected(self, uuid: str) -> None:
        """Called when user picks an image in the Generation tab selector."""
        storage = self.c.view.images_tab.storage
        img_data = storage.get_data(uuid)
        if img_data is None:
            self.c.model_cell_intensity.clear_segmentation_labels()
            return
        channels = img_data["data"]
        self.c.view.cell_intensity_groupbox.update_channels(channels)
        self.c.view.cell_intensity_groupbox.image_selector.set_channels(channels)
        self.c.model_cell_intensity.set_source_uuid(uuid)
        self.c.view.cell_intensity_groupbox.set_generation_enabled(True)

        # Load stardist labels if present
        stardist_channel = self.c.view.images_tab.image_tree_view._find_virtual_stardist_channel(uuid)
        if stardist_channel is not None:
            self.c.model_cell_intensity.load_segmentation_labels_from_storage(
                uuid, int(stardist_channel)
            )
        else:
            self.c.model_cell_intensity.clear_segmentation_labels()

    def _handle_generation_source_selected(self, item_uuid, stardist_channel):
        """Enable generation only after source selection and auto-load virtual StarDist labels."""
        has_source = item_uuid is not None
        self.c.model_cell_intensity.set_source_uuid(item_uuid if has_source else None)
        self.c.view.cell_intensity_groupbox.set_generation_enabled(has_source)
        if not has_source:
            self.c.model_cell_intensity.clear_segmentation_labels()
            return

        if stardist_channel is None:
            self.c.model_cell_intensity.clear_segmentation_labels()
            return

        self.c.model_cell_intensity.load_segmentation_labels_from_storage(
            item_uuid, int(stardist_channel)
        )

        # Sync the image selector to reflect the context-menu selection
        selector = self.c.view.cell_intensity_groupbox.image_selector
        selector.set_selected_uuid_silent(item_uuid)
        img_data = self.c.view.images_tab.storage.get_data(item_uuid)
        if img_data:
            selector.set_channels(img_data["data"])
            self.c.view.cell_intensity_groupbox.update_channels(img_data["data"])

    def _setup_img_broadcast_conns(self):
        """Image signal broadcast to multiple targets"""
        # ensure all the relevant UI components are not None

        image_signal = self.c.model_canvas.image_signal
        image_signal.connect(self.c.view.tool_bar.updateChannelSelector)
        image_signal.connect(self.c.view.canvas.load_channels)
        # StarDist, registration, and generation are driven by their image selectors, not canvas signal
        self.c.model_canvas.uuid_changed.connect(self._on_canvas_uuid_changed)

        ref_image = self.c.model_reference_canvas.image_signal
        ref_image.connect(self.c.view.small_view.load_channels)
        ref_image.connect(self._on_reference_canvas_image_changed)

        deleted_uuid_signal = self.c.view.images_tab.image_tree_view.item_deleted
        deleted_uuid_signal.connect(self.c.model_canvas.remove_from_canvas)
        deleted_uuid_signal.connect(self.c.model_reference_canvas.remove_from_canvas)

    def _setup_misc_connections(self):
        """Miscellaneous connections"""
        self.c.view.view_tab.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stacked_widget.currentChanged.connect(
            self.c.update_small_view_visibility
        )

    def _on_canvas_uuid_changed(self, uuid: str) -> None:
        """Forward canvas image switches to all three tab selectors (unless manually overridden)."""
        self.c.view.segmentation_image_selector.follow_canvas_uuid(uuid)
        self.c.view.register_groupbox.image_selector.follow_canvas_uuid(uuid)
        self.c.view.cell_intensity_groupbox.image_selector.follow_canvas_uuid(uuid)

    def _on_reference_canvas_image_changed(self, _channels: dict, _full: bool) -> None:
        """Forward reference canvas image switches to the Alignment reference selector."""
        uuid = self.c.model_reference_canvas.uuid
        if uuid is None:
            return
        self.c.view.register_groupbox.reference_selector.follow_canvas_uuid(str(uuid))

    # ------------------------------------------------------------------
    # Segmentation image/channel selector
    # ------------------------------------------------------------------

    def _setup_segmentation_selector_connections(self):
        """Wire the shared image+channel selector in the Segmentation tab."""
        selector = self.c.view.segmentation_image_selector
        selector.image_changed.connect(self._on_segmentation_image_selected)
        selector.channel_changed.connect(self._on_segmentation_channel_selected)

        tree_model = self.c.view.images_tab.image_tree_model
        tree_model.rowsInserted.connect(
            lambda *_: self._repopulate_segmentation_selector()
        )
        tree_model.rowsRemoved.connect(
            lambda *_: self._repopulate_segmentation_selector()
        )
        tree_model.dataChanged.connect(
            lambda *_: self._repopulate_segmentation_selector()
        )
        # Populate immediately with any images already in the tree (e.g. from project load)
        self._repopulate_segmentation_selector()

    def _repopulate_segmentation_selector(self):
        """Rebuild the image list. Does not auto-select — user must explicitly pick an image."""
        selector = self.c.view.segmentation_image_selector
        selector.refresh_images(self.c.view.images_tab.image_tree_model)

    def _on_segmentation_image_selected(self, uuid: str):
        """Called when the user picks a different image in the segmentation selector."""
        storage = self.c.view.images_tab.storage
        img_data = storage.get_data(uuid)
        if img_data is None:
            return
        channels = img_data["data"]
        name = img_data["name"]

        selector = self.c.view.segmentation_image_selector
        selector.set_channels(channels)
        channel = selector.current_channel()

        # Update StarDist model
        self.c.model_stardist.set_protein_image(channels, channel, name, uuid)

        # Store blur target
        self._seg_source_uuid = uuid
        self._seg_source_channel = channel

    def _on_segmentation_channel_selected(self, channel: str):
        """Called when the user picks a different channel in the segmentation selector."""
        uuid = self.c.view.segmentation_image_selector.current_uuid()
        if not uuid:
            return
        storage = self.c.view.images_tab.storage
        img_data = storage.get_data(uuid)
        if img_data is None:
            return

        # Sync StarDist channel selector UI without re-triggering its signal
        stardist_selector = self.c.view.stardist_groupbox.stardist_channel_selector
        stardist_selector.blockSignals(True)
        stardist_selector.setCurrentText(channel)
        stardist_selector.blockSignals(False)
        self.c.model_stardist.set_channel(channel)

        # Update ImageStorage so the badge delegate reads the correct channel,
        # then emit cell_image_set to repaint badges and refresh the groupbox title.
        name = img_data["name"]
        ImageStorage().add_data("seg_source_uuid", {"value": uuid, "channel": channel})
        self.c.model_stardist.cell_image_set.emit(name, channel)

        # Update blur target
        self._seg_source_channel = channel

    # ------------------------------------------------------------------
    # CellLayerAlignment image/channel selectors
    # ------------------------------------------------------------------

    def _setup_cell_layer_alignment_selector_connections(self):
        """Repopulate CellLayerAlignment image selectors when tree changes."""
        tree_model = self.c.view.images_tab.image_tree_model
        tree_model.rowsInserted.connect(
            lambda *_: self._repopulate_cell_alignment_selectors()
        )
        tree_model.rowsRemoved.connect(
            lambda *_: self._repopulate_cell_alignment_selectors()
        )
        tree_model.dataChanged.connect(
            lambda *_: self._repopulate_cell_alignment_selectors()
        )
        self._repopulate_cell_alignment_selectors()

    def _repopulate_cell_alignment_selectors(self):
        """Rebuild both CellLayerAlignment image selector lists."""
        cell_aln = self.c.view.cell_layer_alignment
        tree = self.c.view.images_tab.image_tree_model
        cell_aln.target_image_selector.refresh_images(tree)
        cell_aln.unaligned_image_selector.refresh_images(tree)
