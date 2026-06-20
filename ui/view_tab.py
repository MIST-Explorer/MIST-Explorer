import logging
import os
import time
import traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import pandas as pd
import qtrangeslider
import tifffile as tiff
from PIL import Image
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from controller import Controller
from core.dataframe_utils import METADATA_COLUMNS, get_marker_columns
from core.image_utils import auto_contrast_helper, create_lut, scale_adjust
from core.Worker import Worker
from ui.analysis.graphing.UMAPPlot import UMAPVisualizer
from utils import resource_path

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

color_dict = {
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "purple": [128, 0, 128],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "brown": [165, 42, 42],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "silver": [192, 192, 192],
    "gray": [128, 128, 128],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "dark green": [0, 128, 0],
    "teal": [0, 128, 128],
    "navy": [0, 0, 128],
    "indigo": [75, 0, 130],
    "deep pink": [255, 20, 147],
    "chocolate": [210, 105, 30],
    "moccasin": [255, 228, 181],
    "steel blue": [70, 130, 180],
    "light sky blue": [135, 206, 250],
    "sandy brown": [244, 164, 96],
    "green yellow": [173, 255, 47],
    "khaki": [240, 230, 140],
    "tomato": [255, 99, 71],
    "dark slate blue": [72, 61, 139],
}

DEFAULT_PROTEIN_LOWER_PERCENTILE = 1.0
DEFAULT_PROTEIN_UPPER_PERCENTILE = 99.5
DEFAULT_PROTEIN_GAMMA = 0.8


def create_contrast_lut(min_val, max_val):
    """Creates a Look-Up Table for contrast adjustment."""
    if min_val >= max_val:
        return np.zeros(256, dtype=np.uint8)

    lut = np.zeros(256, dtype=np.uint8)
    lut[min_val : max_val + 1] = np.linspace(
        start=0,
        stop=255,
        num=(max_val - min_val + 1),
        endpoint=True,
        dtype=np.uint8,
    )
    lut[max_val + 1 :] = 255
    return lut


class ControlsBox:
    def __init__(self):
        self.name = ""
        self.image = np.array([[]])
        self.base_image = np.array([[]])  # Stores the unthresholded 0-255 image
        self.q_image = None
        self.cell_image = np.array([[]])
        self.contrast_cache = {}

        self.current_opacity = 100.0
        self.current_contrast = [0, 255]
        self.current_visibility_threshold = [0.0, 100.0]
        self.current_visibility = True
        self.current_tint = QColor(255, 255, 255)

        # actual components that we just want to keep track of
        self.tint_label = None
        self.opacity_slider = None
        self.title_label = (
            None  # QLabel in the compact title bar, updated on layer reorder
        )

        # entire component layout
        self.layout = None

        # self.tint_yn
        self.tint_yn = True




# def write_protein(protein_data, reduced_cell_img):
#     t = time.time()
#     print(protein_data)
#     cnv = write_protein_sub(protein_data, reduced_cell_img)
#     print(time.time() - t)

#     return cnv


# @njit(cache=True)
# def write_protein_sub(protein_data=np.array([]), reduced_cell_img=np.array([[]])):
#     # Copy the image
#     cnv = reduced_cell_img.copy()

#     # Extract protein data, winsorize, and rescale
#     protein_1 = protein_data
#     lower, upper = np.percentile(protein_1, [0, 100])
#     protein_1 = np.clip(protein_1, lower, upper)
#     # protein_1 = 60 + (protein_1 - lower) * (255 - 60) / (upper - lower)

#     for i in range(cnv.shape[0]):
#         for j in range(cnv.shape[1]):
#             id = reduced_cell_img[i, j]
#             if id > 0 and id <= len(protein_1):
#                 cnv[i, j] = protein_1[id - 1]


# return cnv
def robust_scale_protein_values(
    protein_data,
    lower_percentile=DEFAULT_PROTEIN_LOWER_PERCENTILE,
    upper_percentile=DEFAULT_PROTEIN_UPPER_PERCENTILE,
    gamma=DEFAULT_PROTEIN_GAMMA,
):
    """Robustly scale per-cell intensities for display.

    Typical microscopy tools clip outliers with percentile windowing, then map to 8-bit.
    This keeps sparse hot pixels from suppressing the majority of cell signal.
    """
    values = np.asarray(protein_data, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)

    foreground = values[values > 0]
    if foreground.size == 0:
        return np.zeros_like(values, dtype=np.uint8)

    if foreground.size < 8:
        lo = 0.0
        hi = float(np.max(foreground))
    else:
        lo, hi = np.percentile(foreground, [lower_percentile, upper_percentile])

    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(foreground))
        lo = min(lo, hi)

    span = hi - lo
    if span <= 0:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values - lo) / span
        normalized = np.clip(normalized, 0.0, 1.0)

    if gamma > 0 and gamma != 1:
        normalized = np.power(normalized, gamma)

    return (normalized * 255.0).astype(np.uint8)


def write_protein(protein_data, reduced_cell_img):
    """
    Generates a protein intensity image using vectorized numpy indexing.
    Intensity is robustly scaled for display with percentile clipping.

    Args:
        protein_data (np.ndarray): 1D array of intensity values for each cell.
        reduced_cell_img (np.ndarray): 2D array where pixel values are cell IDs (integers).

    Returns:
        np.ndarray: 2D array representing the protein intensity image.
    """
    # Ensure protein_data is a numpy array
    if not isinstance(protein_data, np.ndarray):
        protein_data = np.array(protein_data, dtype=np.float32)
    else:
        # Ensure float for division
        protein_data = protein_data.astype(np.float32)

    # Robustly map each protein channel to display range so high outliers
    # do not dominate layer visibility.
    normalized_data = robust_scale_protein_values(protein_data)

    # Create a lookup table (LUT) for protein intensities.
    # The +1 is for the background (cell ID 0), which will have an intensity of 0.
    num_cells = len(normalized_data)
    intensity_lut = np.zeros(num_cells + 1, dtype=np.uint8)
    cell_data_image = np.zeros(num_cells + 1, dtype=np.uint16)
    # intensity_lut.fill(np.nan)

    # Cell ID 1 maps to index 0 in protein_data, so it goes into index 1 of our LUT.
    intensity_lut[1:] = normalized_data
    cell_data_image[1:] = protein_data

    # Use the cell ID image to index into the LUT to create the final image.
    protein_image = intensity_lut[reduced_cell_img]
    cell_image = cell_data_image[reduced_cell_img]

    # protein_image[protein_image==0] = np.nan

    return protein_image, cell_image


def collapse_duplicate_cell_ids(df):
    """Collapse duplicate CellID rows to a single row per cell."""
    if "CellID" not in df.columns:
        return df

    duplicate_row_count = len(df) - df["CellID"].nunique(dropna=True)
    if duplicate_row_count <= 0:
        return df

    coordinate_cols = [col for col in ("Global X", "Global Y") if col in df.columns]
    numeric_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    other_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS and col not in numeric_cols
    ]

    agg_map = {col: "first" for col in coordinate_cols}
    agg_map.update({col: "median" for col in numeric_cols})
    agg_map.update({col: "first" for col in other_cols})

    collapsed = df.groupby("CellID", as_index=False).agg(agg_map)
    collapsed = collapsed[[col for col in df.columns if col in collapsed.columns]]
    logger.warning(
        "Collapsed %s duplicate CellID row(s) to %s unique cells.",
        duplicate_row_count,
        len(collapsed),
    )
    return collapsed


def scale_image_to_255(image_array):
    try:
        if image_array.dtype == np.uint8:
            return image_array
    except AttributeError:
        pass

    scaled_image = (
        255
        * (image_array - image_array.min())
        / (image_array.max() - image_array.min())
    )
    return scaled_image.astype(np.uint8)


def scale(val, src, dst):
    return int(((val - src[0]) / float(src[1] - src[0])) * (dst[1] - dst[0]) + dst[0])


class LayerDialog(QDialog):
    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Layer to Add")
        self.setGeometry(100, 100, 400, 300)

        self.layers = layers
        my_layout = QHBoxLayout()
        assert my_layout is not None

        self.layer_list = QListWidget()
        for i, layer in enumerate(layers):
            item = QListWidgetItem(layer["name"])
            self.layer_list.addItem(item)
        my_layout.addWidget(self.layer_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        my_layout.addWidget(self.buttons)

        self.setLayout(my_layout)

    def get_selected_layer_index(self):
        logger.debug(
            "selected items: %s",
            [item.text() for item in self.layer_list.selectedItems()],
        )
        logger.debug("layer names: %s", [layer["name"] for layer in self.layers])
        logger.debug(
            "selected layer index: %s",
            [layer["name"] for layer in self.layers].index(
                [item.text() for item in self.layer_list.selectedItems()][0]
            ),
        )

        selected_items = self.layer_list.selectedItems()
        if selected_items:
            selected_index = [layer["name"] for layer in self.layers].index(
                [item.text() for item in self.layer_list.selectedItems()][0]
            )

            logger.debug("selected index %s", selected_index)
            return selected_index
        return None


class ColorDialog(QDialog):
    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Tint Color")
        self.setGeometry(100, 100, 400, 300)

        self.colors = colors
        my_layout = QVBoxLayout()

        self.color_list = QListWidget()
        for color_name in colors.keys():
            item = QListWidgetItem(color_name)
            self.color_list.addItem(item)
        my_layout.addWidget(self.color_list)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        my_layout.addWidget(self.buttons)

        self.setLayout(my_layout)

    def get_selected_color_name(self):
        selected_items = self.color_list.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None


# changes made
def adjust_contrast(image, min=5, max=100):
    t0 = time.perf_counter()

    image = scale_adjust(image)
    t1 = time.perf_counter()
    logger.debug(f"Time for scale_adjust: {t1 - t0:.6f} sec")

    lut = create_lut(min, max)
    t2 = time.perf_counter()
    logger.debug(f"Time for create_lut: {t2 - t1:.6f} sec")

    res = np.clip(cv2.LUT(image, lut), 0, 254, dtype=np.uint8)
    t3 = time.perf_counter()
    logger.debug(f"Time for LUT application: {t3 - t2:.6f} sec")

    logger.debug(f"Total adjust_contrast time: {t3 - t0:.6f} sec")
    return res


class ImageOverlay(QWidget):
    update_contrast_sig = pyqtSignal(int, tuple)
    export_png_sig = pyqtSignal(int)
    export_tif_sig = pyqtSignal(int)
    update_layer_cmap_sig = pyqtSignal(int, np.ndarray)
    update_layer_opacity_sig = pyqtSignal(int, float)
    update_layer_visible_sig = pyqtSignal(int, bool)
    change_pix = pyqtSignal(np.ndarray, int)
    progress = pyqtSignal(int, str)
    reset_view_tab = pyqtSignal()

    def __init__(self, pixmap_label, enc):
        super().__init__()

        self.enc = enc

        self.pixmap_label = pixmap_label

        # self.df_path =  "/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        # self.im_path = "/Users/clark/Downloads/new_sd.png"

        self.df_path = None
        self.im_path = None
        self.overlay_path = None

        self.controls = []

        self.loaded_df = None

        self.controller = None
        self.model_canvas = None

        o_timer = QTimer()
        o_timer.setInterval(300)
        o_timer.setSingleShot(True)
        self.opacity_timer = o_timer
        self.contrast_sliders = []
        self.tint_lut_cache = {}
        # Connect timer to the actual work
        self.grayscale = True
        self.initUI()
        self.pending_contrast_idx = 0
        self.contrast_timer = QTimer()
        self.contrast_timer.setInterval(10)  # 300 ms of no movement triggers update
        self.contrast_timer.setSingleShot(True)

    def load_stardist_image(self):
        if self.im_path is None:
            raise ValueError("im_path is None")

        stardist_labels = Image.open(self.im_path)
        stardist_labels = np.array(stardist_labels)

        # return stardist_labels

        # reduced_cell_img = cv2.resize(stardist_labels.astype("float32"), (3000, 3000), interpolation = cv2.INTER_NEAREST_EXACT)

        self.scale_down_factor = 1

        reduced_cell_img = cv2.resize(
            stardist_labels,
            (0, 0),
            fx=self.scale_down_factor,
            fy=self.scale_down_factor,
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        return reduced_cell_img

    def load_df(self):
        if self.req_df() == "":
            raise ValueError("Need to load protein data first.")
        else:
            assert self.df_path is not None
        df = None
        if self.df_path.endswith("csv"):
            df = pd.read_csv(self.df_path)
        elif self.df_path.endswith("xlsx"):
            df = pd.read_excel(self.df_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx")

        # print("df raw", df)
        df = df[df.columns.drop(list(df.filter(regex="N/A")))]

        self.loaded_df = df
        return df

    def get_df(self):
        if self.loaded_df is None:
            raise ValueError("Need to load DF in view tab.")
        return self.loaded_df

    def generate_image(self, index):
        if self.reduced_cell_img.size == 0:
            raise ValueError("Please load an image first.")
        if self.df is None:
            raise ValueError("Dataframe not processed. Please click 'Apply' first.")

        # Get the name of the protein to generate from the layers list
        protein_name = self.layers[index]["name"]

        # The dataframe is now indexed by CellID.
        # We need to construct a 1D array that can be used as a LUT for write_protein.
        # The LUT must be indexed from 0 to max_cell_id.
        max_id_in_image = self.reduced_cell_img.max()

        # Get the sparse series of intensities for the current protein
        protein_series = self.df[protein_name]

        # Create a dense array for the LUT, with 0 for cells not in the data
        # The +1 is because cell IDs are 1-based.
        lut_data = np.zeros(max_id_in_image + 1)

        # Place the protein intensities at the correct index (cell ID)
        # This handles cases where the df doesn't contain all cell IDs present in the image
        valid_indices = protein_series.index[protein_series.index <= max_id_in_image]
        lut_data[valid_indices] = protein_series.loc[valid_indices].values

        # The vectorized write_protein function expects a simple 0-indexed array
        # corresponding to cell IDs 1, 2, 3...
        # So we pass the LUT data, but skip the 0th element.
        im, cell_data = write_protein(lut_data[1:], self.reduced_cell_img)

        # The call to adjust_contrast is removed to show the absolute scaled intensity initially.
        # User can adjust contrast manually with the layer slider.
        # result = tint_grayscale_image(im, [255, 255, 255])

        return im, cell_data

    def build_all(self):
        if not self.controller:
            self.controller = Controller.get()
            self.model_canvas = self.controller.model_canvas


        if not hasattr(self, "im_path"):
            import ui.app

            QMessageBox.critical(
                ui.app.MainWindow(), "Error", "Please an load image first!"
            )
            return

        if not hasattr(self, "df_path"):
            import ui.app

            QMessageBox.critical(
                ui.app.MainWindow(), "Error", "Please load data first!"
            )
            return

        start = time.time()
        self.progress.emit(10, "Loading images and data...")

        self.reduced_cell_img = self.load_stardist_image()

        df = self.loaded_df
        if df is None:
            # This should have been loaded by the user action before build_all is called
            return

        self.progress.emit(30, "Associating cells with data...")
        # --- NEW: Associate DF with Cell IDs ---
        if "CellID" not in df.columns:  # Perform this expensive operation only once
            scale_factor = 1

            # Vectorized coordinate scaling
            scaled_x = (df["Global X"] * scale_factor).astype(int)
            scaled_y = (df["Global Y"] * scale_factor).astype(int)

            # Clip coordinates to be within image bounds
            h, w = self.reduced_cell_img.shape
            scaled_x = np.clip(scaled_x, 0, w - 1)
            scaled_y = np.clip(scaled_y, 0, h - 1)

            # Look up all cell IDs at once using vectorized indexing
            cell_ids = self.reduced_cell_img[scaled_y, scaled_x]
            df["CellID"] = cell_ids

        # Filter out rows that are in the background (don't map to a cell)
        df["CellID"] = pd.to_numeric(df["CellID"], errors="coerce")
        df = df[df["CellID"] > 0].copy()
        df["CellID"] = df["CellID"].astype(np.int64)
        df = collapse_duplicate_cell_ids(df)

        # Set CellID as the index for fast lookups later
        self.df = df.set_index("CellID")
        # --- END NEW ---

        self.progress.emit(70, "Preparing layers...")

        # Get protein names (assuming they are all columns after the first few)
        protein_columns = get_marker_columns(self.loaded_df)
        protein_names = [col for col in protein_columns if col in self.df.columns]

        # Prepare layer structure, but don't generate images yet (on-demand is fast now)
        self.ims = [None for _ in protein_names]
        self.layers = [{"name": name, "image": None} for name in protein_names]

        end = time.time()
        logger.debug(f"Build time: {end - start:.2f} seconds")
        self.progress.emit(100, "Done")

        # Update UI
        self.scroll_area.setVisible(True)
        self.add_layer_button.setVisible(True)
        self.add_other_image_button.setVisible(True)
        self.open_image.setVisible(False)
        self.open_image_label.setVisible(False)
        self.open_df.setVisible(False)
        self.open_df_label.setVisible(False)
        self.apply_button.setVisible(False)
        self.cancel_reset.setVisible(True)
        self.export_tif_button.setVisible(True)
        self.export_png_button.setVisible(True)
        self.splitter.setStretchFactor(0, 1)  # Give more space to the layer list
        self.splitter.setStretchFactor(1, 0)  # Give more space to the layer list

        return (self.ims, protein_names)

    def open_umap_analysis(self):
        if not hasattr(self, "df") or self.df is None:
            QMessageBox.critical(
                self, "Error", "Data not processed. Please click 'Apply' first."
            )
            return

        try:
            # Prepare data for UMAP
            # self.df has CellID as index. Reset it and rename to "Cell ID" as expected by DataModel
            df_for_umap = self.df.reset_index().rename(columns={"CellID": "Cell ID"})

            # Instantiate and show UMAP window
            # We keep a reference to avoid garbage collection
            self.umap_window = UMAPVisualizer(df_for_umap, self.reduced_cell_img, view_tab=self)
            self.umap_window.show()
        except Exception as e:
            tb = traceback.format_exc()
            logger.critical(f"Failed to open UMAP analysis:\n{tb}")
            QMessageBox.critical(
                self,
                "UMAP Error",
                f"Failed to open UMAP analysis:\n\n{e}\n\nSee log for full traceback.",
            )

    # can probably be optimized by having just one array with all names and values, and then filtering out invisible ones
    # avoids looping through self.controls
    def get_layer_values_at(self, x, y):
        if len(self.controls) == 0:
            return None

        layer_values = []
        for c in self.controls:
            if not c.current_visibility:
                continue

            # Skip if this is an "other image" (RGB) which has no cell data
            if c.cell_image.size == 0:
                continue

            value = c.cell_image[y, x]
            if hasattr(c, "label_mapping") and c.label_mapping is not None:
                value = c.label_mapping.get(value, value)
            layer_values.append((c.name, value))

        return layer_values

    def cancel_reset_first(self):
        reply = QMessageBox()
        reply.setText("Are you sure you want to reset?")
        reply.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        resp = reply.exec()
        if resp != QMessageBox.StandardButton.Yes:
            return

        while len(self.controls) > 0:
            self.delete_layer(self.controls[0].layout)

        self.enc.analysis_tab.reset_rois()

        self.open_image.setVisible(True)
        self.open_image_label.setVisible(False)

        self.open_df.setVisible(True)
        self.open_df_label.setVisible(False)

        self.apply_button.setVisible(True)

        self.add_layer_button.setVisible(False)
        self.cancel_reset.setVisible(False)
        self.add_other_image_button.setVisible(False)
        self.export_tif_button.setVisible(False)
        self.export_png_button.setVisible(False)
        self.reset_view_tab.emit()

    def initUI(self):
        main_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_area.setMinimumHeight(450)  # Set a reasonable minimum height
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )  # Disable horizontal scroll

        self.scroll_content = QWidget()
        self.scroll_content.setAutoFillBackground(True)
        self.scroll_content.setBackgroundRole(QPalette.ColorRole.Base)
        self.scroll_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setContentsMargins(10, 10, 10, 20)

        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        # Create a splitter to separate layers from controls
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.scroll_area)

        # Create a container for the bottom controls
        bottom_controls_widget = QWidget()
        bottom_controls_layout = QVBoxLayout(bottom_controls_widget)

        self.splitter.addWidget(bottom_controls_widget)

        # Add splitter to the main layout
        main_layout.addWidget(self.splitter)

        # Set initial sizes for the splitter (optional, but good for default view)
        # This gives more space to the layer list initially
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        # --- Add all controls to the bottom layout ---
        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.show_layer_dialog)
        bottom_controls_layout.addWidget(self.add_layer_button)
        self.add_layer_button.setVisible(False)

        self.add_other_image_button = QPushButton("Add Other Image")
        self.add_other_image_button.clicked.connect(self.open_other_image)
        bottom_controls_layout.addWidget(self.add_other_image_button)
        self.add_other_image_button.setVisible(False)

        self.cancel_reset = QPushButton("Cancel And Upload New")
        self.cancel_reset.clicked.connect(self.cancel_reset_first)
        bottom_controls_layout.addWidget(self.cancel_reset)
        self.cancel_reset.setVisible(False)

        self.open_image = QPushButton("Open Image")
        self.open_image.clicked.connect(self.load_image)
        bottom_controls_layout.addWidget(self.open_image)

        self.open_image_label = QLabel("Path: ")
        self.open_image_label.setVisible(False)
        bottom_controls_layout.addWidget(self.open_image_label)

        self.open_df = QPushButton("Open Cell Data")
        self.open_df.clicked.connect(self.load_df)
        bottom_controls_layout.addWidget(self.open_df)

        self.open_df_label = QLabel("Path: ")
        self.open_df_label.setVisible(False)
        bottom_controls_layout.addWidget(self.open_df_label)

        ### scale slider

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.start_build_all_worker)
        bottom_controls_layout.addWidget(self.apply_button)

        self.export_tif_button = QPushButton("Export to TIF")
        self.export_tif_button.clicked.connect(self.export_to_tif)
        self.export_tif_button.setVisible(False)
        bottom_controls_layout.addWidget(self.export_tif_button)

        self.export_png_button = QPushButton("Export to PNG")
        self.export_png_button.clicked.connect(self.export_to_png)
        self.export_png_button.setVisible(False)
        bottom_controls_layout.addWidget(self.export_png_button)

        # Add a spacer to ensure content can scroll all the way down
        bottom_controls_layout.addStretch(
            1
        )  # Add stretch at the end to push content up

        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.process_images()

    def open_other_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)",
        )

        # file_name = self.overlay_path

        if file_name:
            stardist_labels = Image.open(file_name)
            stardist_labels = np.array(stardist_labels)

            reduced_cell_img = cv2.resize(
                stardist_labels,
                (0, 0),
                fx=self.scale_down_factor,
                fy=self.scale_down_factor,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )

            c = ControlsBox()

            if len(reduced_cell_img.shape) == 2:
                reduced_cell_img = np.stack((reduced_cell_img,) * 3, axis=-1)
            c.image = reduced_cell_img.copy()
            c.name = os.path.basename(file_name)
            c.tint_yn = False
            self.add_layer(c)

    def start_build_all_worker(self):
        self.build_all_worker = Worker(self.build_all)
        self.build_all_worker.start()
        self.build_all_worker.finished.connect(self.build_all_worker.quit)
        self.build_all_worker.finished.connect(self.build_all_worker.deleteLater)
        # self.threadpool.start(self.build_all_worker)

    def less_than_15_chars(self, string):
        if len(string) > 50:
            return string[:49] + "..."

        return string

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif);;All Files (*)",
        )
        if file_name:
            # combined_image = np.array(Image.open(file_name))

            self.open_image_label.setText(
                f"File: {self.less_than_15_chars(os.path.basename(file_name))}"
            )
            self.open_image_label.setVisible(True)
            self.im_path = file_name

    def req_df(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Spreadsheets (*.csv *.xlsx);;All Files (*)"
        )

        if file_name:
            # print()
            self.open_df_label.setText(
                f"File: {self.less_than_15_chars(os.path.basename(file_name))}"
            )
            self.open_df_label.setVisible(True)
            self.df_path = file_name
            return file_name
        return ""

    def show_layer_dialog(self):
        if not hasattr(self, "layers"):
            import ui.app

            QMessageBox.critical(
                ui.app.MainWindow(),
                "Error",
                "Empty canvas, please an load image first",
            )
            return

        dialog = LayerDialog(self.layers, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                selected_index = dialog.get_selected_layer_index()
            except IndexError:
                selected_index = 0
            logger.debug("selected index is %s", selected_index)
            if selected_index is not None:
                c = ControlsBox()

                logger.warning("potential error %s", selected_index)

                try:
                    if self.layers[selected_index]["image"] is None:
                        (
                            self.layers[selected_index]["image"],
                            self.layers[selected_index]["cell_data"],
                        ) = self.generate_image(selected_index)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Error generating image for layer. Please ensure number of instances in segmentation and match number of cells.\n\n"
                        + str(e),
                    )
                reduced_cell_img = np.array(self.layers[selected_index]["image"])
                # reduced_cellImg = reduced_cell_img/255.0
                c.image = reduced_cell_img.copy()
                c.base_image = reduced_cell_img.copy()
                c.cell_image = np.array(self.layers[selected_index]["cell_data"])
                c.name = self.layers[selected_index]["name"]

                self.add_layer(c)

    def show_color_dialog(self, gb):
        idx = self.scroll_layout.indexOf(gb)
        dialog = ColorDialog(color_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color_name = dialog.get_selected_color_name()
            if selected_color_name:
                rgb = color_dict[selected_color_name]
                selected_color = QColor(*rgb)
                self.controls[idx].tint_label.setStyleSheet(
                    f"QPushButton {{ background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #888; border-radius: 2px; }}"
                    "QPushButton:hover { border-color: #555; }"
                )
                self.update_layer_cmap_sig.emit(
                    idx, np.array(self.get_lut(selected_color))
                )

    def add_layer(self, c):
        self.controls.append(c)
        self.add_layer_controls(c)
        idx = len(self.controls) - 1
        self.change_pix.emit(c.image, idx)
        self.update_layer_display(idx)

    def _default_contrast_for_image(self, image):
        if not isinstance(image, np.ndarray) or image.size == 0:
            return (0, 255)
        if image.ndim != 2:
            return (0, 255)

        vmin, vmax = auto_contrast_helper(
            image,
            lower=DEFAULT_PROTEIN_LOWER_PERCENTILE,
            upper=DEFAULT_PROTEIN_UPPER_PERCENTILE,
        )
        vmin = int(np.clip(round(vmin), 0, 255))
        vmax = int(np.clip(round(vmax), 0, 255))
        if vmax <= vmin:
            return (0, 255)
        return (vmin, vmax)

    def update_current_image(self, image):
        last_index = len(self.controls) - 1
        self.controls[last_index].image = image.copy()
        self.process_images()

    def delete_layer(self, gb):
        index = self.scroll_layout.indexOf(gb)
        c = self.controls.pop(index)
        self.contrast_sliders.pop(index)

        layer = c.layout
        self.scroll_layout.removeWidget(layer)
        layer.deleteLater()
        layer = None

        for i, control in enumerate(self.controls):
            chevron = control.title_label.text().split("  ", 1)[0]
            control.title_label.setText(f"{chevron}  Layer {i + 1}: {control.name}")
        self.process_images()
        self.change_pix.emit(np.ndarray(0), index)

    def restart_contrast_timer(self):
        if self.contrast_timer.isActive():
            return  # Don't restart while still moving
        self.contrast_timer.start()

    @staticmethod
    def _layer_theme() -> dict:
        from ui.theme import ThemeManager

        c = ThemeManager.instance().get_current()
        return {
            "label": c["text_secondary"],
            "muted": c["text_secondary"],
            "accent": c["accent"],
            "subtle": c["text_secondary"],
            "subtle_open": c["text_primary"],
            "tint_none": c["text_secondary"],
        }

    @staticmethod
    def make_svg_icon(svg_path: str, size: int = 22) -> QIcon:
        from ui.theme import make_themed_icon

        return make_themed_icon(svg_path, size)

    def add_layer_controls(self, c):
        idx = len(self.controls) - 1
        theme = self._layer_theme()
        from ui.theme import ThemeManager

        tc = ThemeManager.instance().get_current()

        # --- Icons (loaded once per call; small overhead) ---
        icon_eye_open = self.make_svg_icon(resource_path("assets/icons/eye_open.svg"))
        icon_eye_closed = self.make_svg_icon(
            resource_path("assets/icons/eye_closed.svg")
        )

        # ── Outer card (replaces QGroupBox; this is the "gb" passed to all handlers) ──
        layer_card = QFrame()
        layer_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layer_card.setFrameShape(QFrame.Shape.StyledPanel)
        card_layout = QVBoxLayout(layer_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # ── Title bar ──
        title_bar = QWidget()
        title_bar.setObjectName("layerTitleBar")
        title_bar.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        title_bar.setStyleSheet(
            f"QWidget#layerTitleBar {{ background: transparent; }}"
            f"QWidget#layerTitleBar:hover {{ background: {tc['bg_pressed']}; }}"
        )
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(8, 3, 4, 3)
        title_bar_layout.setSpacing(4)

        name_label = QPushButton(f"▼  Layer {idx + 1}: {c.name}")
        name_label.setFlat(True)
        name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        name_label.setStyleSheet(
            f"QPushButton {{ font-weight: bold; font-size: 11px; color: {tc['text_secondary']};"
            f" background: transparent; border: none; text-align: left; padding: 0; }}"
            f" QPushButton:hover {{ color: {tc['accent']}; }}"
        )
        title_bar_layout.addWidget(name_label, stretch=1)

        tint_square = QPushButton()
        tint_square.setFixedSize(15, 15)
        tint_square.setToolTip("Set layer tint")
        tint_square.setFlat(True)
        tint_square.setStyleSheet(
            f"QPushButton {{ background: white; border: 1px solid {tc['border']}; border-radius: 2px; }} "
            f"QPushButton:hover {{ border-color: {tc['accent_dim']}; }}"
        )
        title_bar_layout.addWidget(tint_square)

        eye_btn = QPushButton()
        eye_btn.setIcon(icon_eye_open)
        eye_btn.setFixedSize(22, 22)
        eye_btn.setCheckable(True)
        eye_btn.setChecked(True)
        eye_btn.setToolTip("Hide layer")
        eye_btn.setFlat(True)
        eye_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; margin-left: 4px; }} QPushButton:hover {{ background: {tc['warning']}; border-radius: 3px; }}"
        )
        title_bar_layout.addWidget(eye_btn)

        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(22, 22)
        delete_btn.setToolTip("Delete layer")
        delete_btn.setFlat(True)
        delete_btn.setStyleSheet(
            f"QPushButton:hover {{ background: {tc['danger']}; color: {tc['text_on_accent']}; border-radius: 3px; }} QPushButton {{ font-size:18px; }}"
        )
        title_bar_layout.addWidget(delete_btn)

        card_layout.addWidget(title_bar)

        # ── Collapsible body ──
        body_widget = QWidget()
        body_layout = QVBoxLayout(body_widget)
        body_layout.setContentsMargins(7, 5, 7, 6)
        body_layout.setSpacing(3)

        # Opacity + Contrast rows share a grid so their sliders align
        slider_grid = QGridLayout()
        slider_grid.setSpacing(4)
        slider_grid.setColumnStretch(2, 1)  # slider column expands

        lbl_style = f"font-size: 10px; color: {theme['label']}; font-weight: 600;"

        # Editable number fields: subtle bg + bottom accent on focus
        _edit_style = (
            f"QLineEdit {{"
            f"  font-size: 10px; font-weight: 600;"
            f"  color: {tc['accent']};"
            f"  background: {tc['bg_hover']};"
            f"  border: none; border-bottom: 1px solid {tc['border']};"
            f"  border-radius: 0px; padding: 1px 2px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-bottom: 2px solid {tc['accent']};"
            f"  background: {tc['bg_pressed']};"
            f"}}"
        )
        _edit_muted_style = (
            f"QLineEdit {{"
            f"  font-size: 10px; font-weight: 600;"
            f"  color: {theme['muted']};"
            f"  background: {tc['bg_hover']};"
            f"  border: none; border-bottom: 1px solid {tc['border']};"
            f"  border-radius: 0px; padding: 1px 2px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-bottom: 2px solid {tc['accent']};"
            f"  color: {tc['accent']};"
            f"  background: {tc['bg_pressed']};"
            f"}}"
        )

        # Opacity row (grid row 0)
        opa_lbl = QLabel("Opacity")
        opa_lbl.setStyleSheet(lbl_style)
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        opa_val_edit = QLineEdit("100")
        opa_val_edit.setFixedWidth(36)
        opa_val_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        opa_val_edit.setToolTip("Click to type a value")
        opa_val_edit.setStyleSheet(_edit_style)
        slider_grid.addWidget(opa_lbl, 0, 0)
        slider_grid.addWidget(opacity_slider, 0, 2)
        slider_grid.addWidget(opa_val_edit, 0, 3)

        # Contrast row (grid row 1)
        con_lbl = QLabel("Contrast")
        con_lbl.setStyleSheet(lbl_style)
        if getattr(c, "is_label_layer", False):
            initial_contrast = (0, 255)
        else:
            initial_contrast = self._default_contrast_for_image(c.image)
        con_lo_edit = QLineEdit(str(int(initial_contrast[0])))
        con_lo_edit.setFixedWidth(36)
        con_lo_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        con_lo_edit.setToolTip("Click to type a value")
        con_lo_edit.setStyleSheet(_edit_style)
        contrast_slider = qtrangeslider.QDoubleRangeSlider(Qt.Orientation.Horizontal)
        contrast_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        contrast_slider.setMaximum(255)
        contrast_slider.blockSignals(True)
        contrast_slider.setValue(initial_contrast)
        contrast_slider.blockSignals(False)
        con_hi_edit = QLineEdit(str(int(initial_contrast[1])))
        con_hi_edit.setFixedWidth(36)
        con_hi_edit.setToolTip("Click to type a value")
        con_hi_edit.setStyleSheet(_edit_style)
        slider_grid.addWidget(con_lbl, 1, 0)
        slider_grid.addWidget(con_lo_edit, 1, 1)
        slider_grid.addWidget(contrast_slider, 1, 2)
        slider_grid.addWidget(con_hi_edit, 1, 3)

        body_layout.addLayout(slider_grid)

        self.contrast_sliders.append(contrast_slider)
        self.controls[idx].current_contrast = [initial_contrast[0], initial_contrast[1]]

        # Visibility threshold — collapsible section
        vis_toggle_btn = QPushButton("▸ Visibility Threshold")
        vis_toggle_btn.setFlat(True)
        vis_toggle_btn.setStyleSheet(
            f"font-size: 10px; color: {theme['subtle']}; text-align: left; padding: 1px 0;"
        )
        body_layout.addWidget(vis_toggle_btn)

        vis_body = QWidget()
        vis_body.setVisible(False)
        vis_body_layout = QVBoxLayout(vis_body)
        vis_body_layout.setContentsMargins(0, 0, 0, 0)
        vis_body_layout.setSpacing(3)
        vis_row = QHBoxLayout()
        vis_row.setContentsMargins(0, 0, 0, 0)
        vis_row.setSpacing(4)
        vis_lbl = QLabel("Vis")
        vis_lbl.setFixedWidth(24)
        vis_lbl.setStyleSheet(
            f"font-size: 10px; color: {theme['muted']}; font-weight: 600;"
        )
        vis_lo_edit = QLineEdit("0.0")
        vis_lo_edit.setFixedWidth(36)
        vis_lo_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        vis_lo_edit.setToolTip("Click to type a value")
        vis_lo_edit.setStyleSheet(_edit_muted_style)
        vis_slider = qtrangeslider.QDoubleRangeSlider(Qt.Orientation.Horizontal)
        vis_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        vis_slider.setMinimum(0.0)
        vis_slider.setMaximum(100.0)
        vis_slider.blockSignals(True)
        vis_slider.setValue((0.0, 100.0))
        vis_slider.blockSignals(False)
        vis_hi_edit = QLineEdit("100.0")
        vis_hi_edit.setFixedWidth(36)
        vis_hi_edit.setToolTip("Click to type a value")
        vis_hi_edit.setStyleSheet(_edit_muted_style)
        vis_row.addWidget(vis_lbl)
        vis_row.addWidget(vis_lo_edit)
        vis_row.addWidget(vis_slider)
        vis_row.addWidget(vis_hi_edit)
        vis_body_layout.addLayout(vis_row)
        auto_contrast_button = QPushButton("Auto Contrast")
        vis_body_layout.addWidget(auto_contrast_button)
        body_layout.addWidget(vis_body)

        card_layout.addWidget(body_widget)

        # ── Signal connections ──
        opacity_slider.valueChanged.connect(
            lambda s, gb=layer_card: self.update_opacity(s, gb)
        )

        def on_opacity_slider_changed(s):
            opa_val_edit.blockSignals(True)
            opa_val_edit.setText(str(s))
            opa_val_edit.blockSignals(False)

        opacity_slider.valueChanged.connect(on_opacity_slider_changed)

        def on_opacity_edited():
            try:
                v = max(0, min(int(opa_val_edit.text()), 100))
            except ValueError:
                v = opacity_slider.value()
            opa_val_edit.setText(str(v))
            opacity_slider.setValue(v)

        opa_val_edit.editingFinished.connect(on_opacity_edited)

        def on_contrast_changed(val, gb=layer_card):
            con_lo_edit.blockSignals(True)
            con_hi_edit.blockSignals(True)
            con_lo_edit.setText(str(int(val[0])))
            con_hi_edit.setText(str(int(val[1])))
            con_lo_edit.blockSignals(False)
            con_hi_edit.blockSignals(False)
            self.set_contrast_from_slider(val, gb)

        contrast_slider.valueChanged.connect(on_contrast_changed)

        def on_lo_edited():
            try:
                v = max(
                    0, min(int(con_lo_edit.text()), int(contrast_slider.value()[1]))
                )
            except ValueError:
                v = int(contrast_slider.value()[0])
            con_lo_edit.setText(str(v))
            contrast_slider.setValue((v, contrast_slider.value()[1]))

        def on_hi_edited():
            try:
                v = max(
                    int(contrast_slider.value()[0]), min(int(con_hi_edit.text()), 255)
                )
            except ValueError:
                v = int(contrast_slider.value()[1])
            con_hi_edit.setText(str(v))
            contrast_slider.setValue((contrast_slider.value()[0], v))

        con_lo_edit.editingFinished.connect(on_lo_edited)
        con_hi_edit.editingFinished.connect(on_hi_edited)

        def on_vis_slider_changed(val, gb=layer_card):
            vis_lo_edit.blockSignals(True)
            vis_hi_edit.blockSignals(True)
            vis_lo_edit.setText(f"{val[0]:.1f}")
            vis_hi_edit.setText(f"{val[1]:.1f}")
            vis_lo_edit.blockSignals(False)
            vis_hi_edit.blockSignals(False)
            self.update_visibility_threshold(val, gb)

        vis_slider.valueChanged.connect(on_vis_slider_changed)

        def on_vis_lo_edited():
            try:
                v = max(0.0, min(float(vis_lo_edit.text()), vis_slider.value()[1]))
            except ValueError:
                v = vis_slider.value()[0]
            vis_lo_edit.setText(f"{v:.1f}")
            vis_slider.setValue((v, vis_slider.value()[1]))

        def on_vis_hi_edited():
            try:
                v = max(vis_slider.value()[0], min(float(vis_hi_edit.text()), 100.0))
            except ValueError:
                v = vis_slider.value()[1]
            vis_hi_edit.setText(f"{v:.1f}")
            vis_slider.setValue((vis_slider.value()[0], v))

        vis_lo_edit.editingFinished.connect(on_vis_lo_edited)
        vis_hi_edit.editingFinished.connect(on_vis_hi_edited)

        def on_vis_toggle():
            open_ = not vis_body.isVisible()
            vis_body.setVisible(open_)
            vis_toggle_btn.setText(("▾ " if open_ else "▸ ") + "Visibility Threshold")
            color = theme["subtle_open"] if open_ else theme["subtle"]
            vis_toggle_btn.setStyleSheet(
                f"font-size: 10px; color: {color}; text-align: left; padding: 1px 0;"
            )

        vis_toggle_btn.clicked.connect(on_vis_toggle)

        def on_expand_toggled():
            expanded = not body_widget.isVisible()
            body_widget.setVisible(expanded)
            chevron = "▼" if expanded else "▶"
            layer_text = name_label.text().split("  ", 1)[1]
            name_label.setText(f"{chevron}  {layer_text}")

        name_label.clicked.connect(on_expand_toggled)

        def on_eye_toggled(checked, gb=layer_card):
            self.update_visibility(checked, gb)
            eye_btn.setIcon(icon_eye_open if checked else icon_eye_closed)
            eye_btn.setToolTip("Hide layer" if checked else "Show layer")

        eye_btn.toggled.connect(on_eye_toggled)

        auto_contrast_button.clicked.connect(
            lambda _, gb=layer_card: self.auto_contrast(gb)
        )
        tint_square.clicked.connect(lambda _, gb=layer_card: self.show_color_dialog(gb))
        delete_btn.clicked.connect(lambda _, gb=layer_card: self.delete_layer(gb))

        # ── Store refs ──
        self.controls[idx].opacity_slider = opacity_slider
        self.controls[idx].tint_label = tint_square
        self.controls[idx].title_label = name_label
        self.controls[idx].layout = layer_card
        self.scroll_layout.addWidget(layer_card)

    def set_contrast_from_slider(self, value, gb):
        min_val, max_val = value
        idx = self.scroll_layout.indexOf(gb)
        self.update_contrast((min_val, max_val), idx)

    def update_opacity(self, value, gb):
        idx = self.scroll_layout.indexOf(gb)
        self.controls[idx].current_opacity = value / 100.0
        self.update_layer_opacity_sig.emit(idx, self.controls[idx].current_opacity)

    def update_layer_display(self, idx):
        # t0 = time.perf_counter()

        c = self.controls[idx]
        contrast_key = tuple(c.current_contrast)
        min, max = contrast_key
        min /= 255.0
        max /= 255.0
        self.update_contrast_sig.emit(idx, contrast_key)

    def update_contrast(self, value, idx):
        if idx == -1:
            return
        value = [int(value[0]), int(value[1])]
        self.controls[idx].current_contrast = value
        self.update_layer_display(idx)
        self.process_images()

    def auto_contrast(
        self,
        gb,
        lower=DEFAULT_PROTEIN_LOWER_PERCENTILE,
        upper=DEFAULT_PROTEIN_UPPER_PERCENTILE,
    ):
        idx = self.scroll_layout.indexOf(gb)
        c = self.controls[idx]
        if getattr(c, "is_label_layer", False):
            new_min, new_max = 0, 255
        else:
            img = c.image
            assert isinstance(img, np.ndarray)
            new_min, new_max = auto_contrast_helper(img, lower, upper)
        self.contrast_sliders[idx].setValue((int(new_min), int(new_max)))
        self.update_contrast([new_min, new_max], idx)

    def update_visibility_threshold(self, value, gb):
        idx = self.scroll_layout.indexOf(gb)
        if idx == -1:
            return

        c = self.controls[idx]
        c.current_visibility_threshold = [value[0], value[1]]

        # Get foreground raw intensity values
        raw_cell_image = c.cell_image
        foreground = raw_cell_image[raw_cell_image > 0]

        if foreground.size > 0:
            # Calculate actual intensity thresholds
            lower_val, upper_val = np.percentile(foreground, [value[0], value[1]])

            # Create a mask for visible cells
            mask = (raw_cell_image >= lower_val) & (raw_cell_image <= upper_val)

            # Update the image
            c.image = np.where(mask, c.base_image, 0)
        else:
            c.image = c.base_image.copy()

        # Trigger redraw
        self.change_pix.emit(c.image, idx)
        self.update_layer_display(idx)

    def update_visibility(self, checked, gb):
        idx = self.scroll_layout.indexOf(gb)
        self.controls[idx].current_visibility = checked
        self.update_layer_visible_sig.emit(idx, checked)

    def get_lut(self, color: QColor):
        color_name = color.name()
        if color_name in self.tint_lut_cache:
            return self.tint_lut_cache[color_name]

        # Ramp from 0..255 normalized to [0,1]
        ramp = np.linspace(0, 1, 256)[:, None]  # shape (256, 1)

        # Extract RGB values
        r, g, b, _ = color.getRgb()  # returns (r,g,b,a)
        rgb = np.array([r, g, b], dtype=np.float32)  # shape (3,)

        # Multiply ramp with RGB to create LUT
        lut = (ramp * rgb).astype(np.uint8)  # shape (256, 3)

        # Cache and return
        self.tint_lut_cache[color_name] = lut
        return lut

    def adjust_contrast(self, img, min=5, max=100):
        # pixvals = np.array(img)
        image = scale_adjust(img)
        lut = create_lut(min, max)
        res = np.clip(cv2.LUT(image, lut), 0, 254, dtype=np.uint8)
        return res

    def contrasted_image(self, img, contrast):
        min_val, max_val = contrast

        return adjust_contrast(img, min_val, max_val)

    def process_images(self, display=True):
        pass

    def export_to_png(self):
        self.export_png_sig.emit(len(self.controls))

    def export_to_tif(self):
        """
        Exports the current layers to a multi-channel TIFF file.
        It applies the current contrast and opacity settings for each layer.
        """
        # 1. Prompt user for a save location
        # Assumes 'self' is a QWidget or has access to one for the dialog parent.
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Multi-Channel TIFF", "", "TIFF Files (*.tif *.tiff)"
        )

        # Exit if the user cancelled the dialog
        if not file_path:
            logger.info("Export cancelled.")
            return

        multi_channel_image = []

        for i in range(len(self.controls)):
            # --- Get layer properties ---
            control = self.controls[i]
            opacity = control.current_opacity  # Expected: 0-100
            contrast = control.current_contrast  # Expected: (min, max) tuple, 0-255
            img = control.image  # Expected: uint16 NumPy array

            # Get image data type info (e.g., for uint16, max is 65535)
            dtype_info = np.iinfo(img.dtype)
            max_dtype_val = float(dtype_info.max)

            # --- A. Adjust Contrast ---
            # Convert image to float for calculations to avoid clipping/rounding errors
            img_float = img.astype(np.float32)

            # Scale the 0-255 contrast range to the image's native range (e.g., 0-65535)
            c_min, c_max = contrast
            min_val = (c_min / 255.0) * max_dtype_val
            max_val = (c_max / 255.0) * max_dtype_val

            # Apply the contrast adjustment (windowing/leveling)
            # Handle the edge case where min and max are the same
            if max_val <= min_val:
                processed_float = np.where(img_float > min_val, max_dtype_val, 0.0)
            else:
                # Clip the data to the new min/max range
                processed_float = np.clip(img_float, min_val, max_val)
                # Scale the clipped data to the full 0.0 to max_dtype_val range
                processed_float = (
                    (processed_float - min_val) / (max_val - min_val) * max_dtype_val
                )

            # --- B. Adjust Opacity ---
            # Apply the opacity as a scaling factor
            opacity_factor = opacity / 100.0
            processed_float *= opacity_factor

            # --- C. Finalize and Append ---
            # Clip again to ensure values are valid, then convert back to original dtype
            final_img = np.clip(processed_float, 0, max_dtype_val).astype(img.dtype)
            multi_channel_image.append(final_img)

        # 2. Save the result as a multi-channel TIFF
        if not multi_channel_image:
            logger.warning("No images to save.")
            return

        # multi_channel_image = np.array(multi_channel_image)
        try:
            # Stack the list of 2D images into a single 3D array (C, H, W)
            output_stack = np.stack(multi_channel_image, axis=0)
            logger.debug(output_stack.shape)

            # Save the stack to the selected file path
            tiff.imwrite(file_path, output_stack, imagej=True)

            logger.info(
                f"Successfully exported {len(multi_channel_image)} channels to: {file_path}"
            )

        except Exception as e:
            logger.error(f"Error saving TIFF file: {e}")
            # Consider showing a QMessageBox to the user here for better UX

    def add_cluster_layer(self, adata, cluster_key, cluster_colors, visible, renames):
        c = ControlsBox()
        c.is_label_layer = True
        c.tint_yn = False

        valid_ids = adata.obs["cell_id"].values.astype(int)
        codes = adata.obs[cluster_key].cat.codes.values

        max_id = max(self.reduced_cell_img.max(), valid_ids.max())
        mapping = np.zeros(max_id + 1, dtype=np.int32)

        visible_set = set(visible)
        for val_id, code in zip(valid_ids, codes):
            if code in visible_set:
                mapping[val_id] = code + 1
            else:
                mapping[val_id] = 0

        safe_seg = self.reduced_cell_img.copy()
        mask_overflow = safe_seg >= len(mapping)
        safe_seg[mask_overflow] = 0

        c.image = mapping[safe_seg].astype(np.uint8)
        c.base_image = c.image.copy()
        c.cell_image = c.image.copy()

        label_mapping = {0: "Background"}
        for code in range(len(cluster_colors)):
            label_mapping[code + 1] = renames.get(code, f"Cluster {code}")
        c.label_mapping = label_mapping

        c.name = f"{cluster_key.capitalize()} Clusters"

        self.add_layer(c)

        idx = len(self.controls) - 1

        lut = np.zeros((256, 3), dtype=np.uint8)
        for code in range(len(cluster_colors)):
            hex_color = cluster_colors[code]
            h = hex_color.lstrip("#")
            rgb_u8 = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
            if code in visible_set:
                lut[code + 1] = rgb_u8

        self.update_layer_cmap_sig.emit(idx, lut)


def numpy_to_qimage(array):
    if len(array.shape) == 2:  # Grayscale image
        height, width = array.shape
        qimage = QImage(
            array.data, width, height, width, QImage.Format.Format_Grayscale8
        )
    elif len(array.shape) == 3 and array.shape[2] == 3:  # RGB image
        height, width, channels = array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
    elif len(array.shape) == 3 and array.shape[2] == 4:  # RGBA image
        height, width, channels = array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
        )
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    return qimage
