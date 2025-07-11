"""Class to handle signal connections"""

import os
import typing
import uuid

import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core import (
    CellIntensity,
    ImageGraphicsView,
    ImageWrapper,
    ReferenceGraphicsView,
    Register,
    StarDist,
)
from ui.alignment.alignment_preview_dialog import AlignmentPreviewDialog

if typing.TYPE_CHECKING:
    from ui.app import Ui_MainWindow


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
        if cls._instance is None:
            cls._instance = cls(app)

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("Controller not initialized")
        return cls._instance

    def __init__(self, app: "Ui_MainWindow"):
        self._initialized = True
        self.image_count = 0
        self.prev_tab_index = 0
        self.model_canvas = ImageGraphicsView(self)
        self.model_stardist = StarDist()
        self.model_register = Register()
        self.model_cell_intensity = CellIntensity()
        self.reference_view = ReferenceGraphicsView()
        self.view = app
        self.open_files_dialog = None
        self.view.images_tab.set_model_canvas(self.model_canvas)
        self.view.images_tab.set_model_stardist(self.model_stardist)
        self.signal_manager = SignalConnectionManager(self)
        self.signal_manager.setup_all_connections()
        self.storage = self.view.images_tab.storage

        # Handle initial arguments
        initial_args = vars(app.args)
        if initial_args["image"] is not None:
            self.model_canvas.add_to_canvas(initial_args["image"])
        if initial_args["reference"] is not None:
            self.reference_view.add_to_canvas(initial_args["reference"])

        # self.need_preview_alignment.connect(self._handle_aligned_image)
        self.view.cell_layer_alignment.aligner.aligned_image_signal.connect(
            self._handle_aligned_image
        )
        self.model_register.alignment_complete.connect(self._handle_aligned_image)
        self.model_register.error.connect(self.handle_error)

    def handle_error(self, error_message):
        QMessageBox.critical(self.view, "Error", error_message)

    def pixmap_to_image(self, pixmap: QPixmap):
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

        pm = self.model_canvas.pixmap
        print(pm)
        # qimage = pm.toImage()
        if pm is not None:
            im = self.pixmap_to_image(pm)

            file_name, _ = QFileDialog.getSaveFileName(
                None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
            )
            if file_name:
                print(file_name)
                Image.fromarray(im).save(file_name)

            else:
                return False

        else:
            self.handle_error("No image in canvas, please load image")

    def open_file_dialog(self, viewer):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.tif);;All Files (*)"
        )
        if file_name:
            viewer.add_to_canvas(file_name)

    def on_action_reference_triggered(self):
        self.open_file_dialog(self.reference_view)

    def on_action_open_triggered(self):
        self.open_file_dialog(self.model_canvas)

    # add new image to storage
    def handle_new_image(self, data, file_name):
        storage_item = {}
        storage_item["name"] = os.path.basename(file_name)
        self.image_count += 1

        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.model_canvas.set_uuid(my_uuid)
        self.view.images_tab.add_item(my_uuid)

    def handle_new_reference_image(self, data, file_name):
        storage_item = {}
        storage_item["name"] = os.path.basename(file_name)
        self.image_count += 1

        storage_item["data"] = data
        my_uuid = str(uuid.uuid4())
        self.view.images_tab.add_to_storage(my_uuid, storage_item)
        self.reference_view.set_uuid(my_uuid)
        self.view.images_tab.add_item(my_uuid)

    def _handle_aligned_image(self, aligned_data, target_small, aligned_small):
        """Handle the aligned image result"""
        # Show the preview dialog
        confirmed = self._show_preview_dialog(target_small, aligned_small)
        if confirmed:
            aligned_image = aligned_data["data"]
            item_uuid = aligned_data["uuid"]
            layer = aligned_data["layer"]
            item = self.storage.get_data(item_uuid)
            assert item is not None, "Aligned image data not found in storage"
            data = item["data"]
            filename = item["name"]
            if isinstance(layer, list):
                assert len(aligned_data["data"].keys()) == len(
                    layer
                ), "Aligned data keys do not match the expected layers"
                data = {}
                for l in layer:
                    wrapped_image = ImageWrapper(aligned_image[l], l)
                    data[l] = wrapped_image
                aligned_name = "Registered_" + filename
            else:
                wrapped_image = ImageWrapper(aligned_image, layer)
                aligned_name = f"Aligned_{filename}"
                data[layer] = wrapped_image
            # self.alignmentCompleteSignal.emit(data, aligned_name)
            self.model_canvas.add_to_canvas(data, True, aligned_name)

    def _show_preview_dialog(self, target_small, aligned_small):
        """Show the preview dialog with red/green overlay"""
        # Use the downscaled images for the preview dialog
        preview_dialog = AlignmentPreviewDialog(target_small, aligned_small)
        result = preview_dialog.exec()
        return result == 1 and preview_dialog.result_accepted

    def need_canvas_change(self, new_index):
        if self.prev_tab_index == 2 and new_index != 2:
            if self.view.canvas.pixmap_item.isVisible():
                return
            self.view.canvas.pixmap_item.show()
            self.view.canvas.view_pixmap_item.hide()
            if self.model_canvas.uuid:
                print("swapping channel")
                self.model_canvas.swap_channel(self.model_canvas.current_channel)
            else:
                print("clearing canvas")
                self.model_canvas.clear_canvas()
        elif self.prev_tab_index != 2 and new_index == 2:
            self.view.view_tab.process_images()
            self.view.canvas.pixmap_item.hide()
            self.view.canvas.view_pixmap_item.show()

    def handle_tab_change(self, index):
        self.need_canvas_change(index)
        self.prev_tab_index = index

    def handle_cancel_registration(self):
        self.model_register.cancel()


class SignalConnectionManager:
    """Manages signal connections"""

    def __init__(self, controller: Controller):
        self.c = controller

    def setup_all_connections(self):
        """Set up all signal-slot connections with full IntelliSense support"""
        self._setup_alignment_connections()
        self._setup_menubar_connections()
        self._setup_toolbar_connections()
        self._setup_image_handling_connections()
        self._setup_canvas_connections()
        self._setup_crop_connections()
        self._setup_transform_connections()
        self._setup_stardist_connections()
        self._setup_registration_connections()
        self._setup_cell_intensity_connections()
        self._setup_image_broadcast_connections()
        self._setup_misc_connections()

    def _setup_alignment_connections(self):
        """Alignment section signal connections"""
        # self.c.view.images_tab.tissue_target_selected.connect(
        #     self.c.view.cell_layer_alignment.set_target_image
        # )
        # self.c.view.images_tab.tissue_unaligned_selected.connect(
        #     self.c.view.cell_layer_alignment.set_unaligned_image
        # )
        self.c.view.cell_layer_alignment.alignmentCompleteSignal.connect(
            self.c.handle_new_image
        )
        self.c.view.cell_layer_alignment.loadOnCanvasSignal.connect(
            self.c.model_canvas.add_to_canvas
        )
        self.c.view.cell_layer_alignment.aligner.progress.connect(
            self.c.view.update_progress_bar
        )

    def _setup_menubar_connections(self):
        """Menu bar signal connections"""
        self.c.view.menuBarUI.open_reference.triggered.connect(
            self.c.on_action_reference_triggered
        )
        self.c.view.menuBarUI.open_image.triggered.connect(
            self.c.on_action_open_triggered
        )

    def _setup_toolbar_connections(self):
        """Toolbar signal connections"""
        self.c.view.toolBarUI.actionReset.triggered.connect(
            self.c.model_canvas.reset_image
        )
        self.c.view.toolBarUI.channelChanged.connect(self.c.model_canvas.swap_channel)
        self.c.view.toolBarUI.contrastSlider.valueChanged.connect(
            self.c.model_canvas.update_contrast
        )
        self.c.view.toolBarUI.cmapChanged.connect(self.c.model_canvas.update_image)
        self.c.view.toolBarUI.auto_contrast_button.clicked.connect(
            self.c.model_canvas.auto_contrast
        )
        self.c.view.toolBarUI.tabChanged.connect(
            self.c.view.stackedWidget.setCurrentIndex
        )
        self.c.view.toolBarUI.tabChanged.connect(self.c.handle_tab_change)

    def _setup_image_handling_connections(self):
        """Image loading and display connections"""
        self.c.view.canvas.image_dropped.connect(self.c.model_canvas.add_to_canvas)
        self.c.view.small_view.image_dropped.connect(
            self.c.reference_view.add_to_canvas
        )
        self.c.reference_view.update_reference.connect(self.c.view.small_view.display)
        self.c.model_canvas.new_image_added.connect(self.c.view.canvas.add_new_image)
        self.c.view.view_tab.change_pix.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.canvas_updated.connect(self.c.view.canvas.update_canvas)
        self.c.model_canvas.update_manager.connect(self.c.handle_new_image)
        self.c.reference_view.update_manager.connect(self.c.handle_new_reference_image)

    def _setup_canvas_connections(self):
        """Canvas-related signal connections"""
        self.c.model_canvas.update_progress.connect(self.c.view.update_progress_bar)
        self.c.model_canvas.error_signal.connect(self.c.handle_error)
        self.c.view.canvas.show_crop.connect(self.c.model_canvas.crop)
        # self.c.model_canvas.cropSignal.connect(self.c.view.canvas.set_crop_status)
        self.c.model_canvas.update_cmap.connect(
            self.c.view.toolBarUI.update_cmap_selector
        )
        self.c.model_canvas.change_slider.connect(
            self.c.view.toolBarUI.update_contrast_slider
        )
        self.c.model_canvas.fill_metadata.connect(self.c.view.get_metadata)

        # Crop visibility toggle
        # self.c.model_canvas.crop_signal.connect(
        #     lambda x: self.c.view.small_view.setVisible(not x)
        # )

    def _setup_crop_connections(self):
        """Crop operation connections"""
        self.c.view.crop_groupbox.crop_button.triggered.connect(
            self.c.view.canvas.start_crop_mode
        )
        # self.c.view.crop_groupbox.crop_button.triggered.connect(
        #     lambda: self.c.view.small_view.setVisible(False)
        # )
        self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
            self.c.view.canvas.cancel_crop_mode
        )
        # self.c.view.crop_groupbox.cancel_crop_button.triggered.connect(
        #     lambda: self.c.view.small_view.setVisible(True)
        # )

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
            lambda: self.c.model_canvas.blur_layer(0, confirm=True)
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.view.gaussian_blur.update_slider_label
        )
        self.c.view.gaussian_blur.slider.doubleValueChanged.connect(
            self.c.model_canvas.blur_layer
        )

    def _setup_stardist_connections(self):
        """StarDist-related connections"""
        # Parameter connections
        self.c.view.stardist_groupbox.stardist_channel_selector.currentTextChanged.connect(
            self.c.model_stardist.set_channel
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
        self.c.view.stardist_groupbox.n_tiles.valueChanged.connect(
            self.c.model_stardist.set_num_tiles
        )
        self.c.view.stardist_groupbox.radius.valueChanged.connect(
            self.c.model_stardist.set_dialation_radisu
        )

        # Execution and results
        self.c.view.stardist_groupbox.stardist_run_button.pressed.connect(
            self.c.model_stardist.run_stardist
        )
        self.c.model_stardist.stardist_done.connect(
            self.c.model_canvas.load_stardist_labels
        )
        self.c.model_stardist.stardist_done.connect(
            self.c.model_cell_intensity.load_stardist_labels
        )
        self.c.model_stardist.error_signal.connect(self.c.handle_error)
        self.c.model_stardist.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stardist_groupbox.save_button.clicked.connect(
            self.c.model_stardist.save_image
        )
        self.c.view.stardist_groupbox.cancel_button.clicked.connect(
            self.c.model_stardist.cancel
        )

    def _setup_registration_connections(self):
        """Registration-related connections"""
        # Parameters
        self.c.view.register_groupbox.alignment_layer.currentTextChanged.connect(
            self.c.model_register.set_alignment_layer
        )
        self.c.view.register_groupbox.protein_cell_layer.currentTextChanged.connect(
            self.c.model_register.set_cell_layer
        )
        self.c.view.register_groupbox.intensity_layer.currentTextChanged.connect(
            self.c.model_register.set_protein_detection_layer
        )
        self.c.view.register_groupbox.overlap.valueChanged.connect(
            self.c.model_register.set_overlap
        )
        self.c.view.register_groupbox.max_size.valueChanged.connect(
            self.c.model_register.set_max_size
        )
        self.c.view.register_groupbox.num_tiles.valueChanged.connect(
            self.c.model_register.set_num_tiles
        )

        # Execution
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

        # Results

    def _setup_cell_intensity_connections(self):
        """Cell intensity-related connections"""
        self.c.view.cellIntensity_groupbox.bead_data.clicked.connect(
            self.c.view.cellIntensity_groupbox.loadBeadData
        )
        self.c.view.cellIntensity_groupbox.color_code.clicked.connect(
            self.c.view.cellIntensity_groupbox.loadColorCode
        )
        self.c.view.cellIntensity_groupbox.emitBeadData.connect(
            self.c.model_cell_intensity.get_bead_data
        )
        self.c.view.cellIntensity_groupbox.emitColorCode.connect(
            self.c.model_cell_intensity.get_color_code
        )
        self.c.view.cellIntensity_groupbox.num_cycles.valueChanged.connect(
            self.c.model_cell_intensity.set_num_decoding_cycles
        )
        self.c.view.cellIntensity_groupbox.num_layers_each.valueChanged.connect(
            self.c.model_cell_intensity.set_num_decoding_colors
        )
        self.c.view.cellIntensity_groupbox.radius_fg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_fg
        )
        self.c.view.cellIntensity_groupbox.radius_bg.valueChanged.connect(
            self.c.model_cell_intensity.set_radius_bg
        )
        self.c.view.cellIntensity_groupbox.run_button.clicked.connect(
            self.c.model_cell_intensity.generate_cell_intensity_table
        )
        self.c.model_cell_intensity.error_signal.connect(self.c.handle_error)
        self.c.view.cellIntensity_groupbox.save_button.clicked.connect(
            self.c.model_cell_intensity.save_cell_data
        )
        self.c.model_cell_intensity.progress.connect(self.c.view.update_progress_bar)
        self.c.view.cellIntensity_groupbox.cancel_button.clicked.connect(
            self.c.model_cell_intensity.cancel
        )

    def _setup_image_broadcast_connections(self):
        """Image signal broadcast to multiple targets"""

        image_signal = self.c.model_canvas.image_signal
        image_signal.connect(self.c.view.toolBarUI.updateChannelSelector)
        image_signal.connect(self.c.view.stardist_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.register_groupbox.updateChannelSelector)
        image_signal.connect(self.c.view.canvas.loadChannels)
        image_signal.connect(self.c.model_stardist.update_channels)
        image_signal.connect(self.c.view.gaussian_blur.updateChannelSelector)
        image_signal.connect(self.c.model_register.update_moving_image)

        ref_image = self.c.reference_view.image_signal
        ref_image.connect(self.c.model_register.update_reference_channels)
        ref_image.connect(self.c.view.small_view.load_channels)

    def _setup_misc_connections(self):
        """Miscellaneous connections"""
        self.c.view.saveSignal.connect(self.c.control_save)
        self.c.view.view_tab.progress.connect(self.c.view.update_progress_bar)
        self.c.view.stackedWidget.currentChanged.connect(
            lambda x: self.c.view.small_view.setVisible(x == 1)
        )
