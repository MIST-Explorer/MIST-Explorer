import os
import threading

import cv2 as cv
import numpy as np
from matplotlib import colormaps
from PIL import Image
from pyclesperanto import dilate_labels
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from stardist.models import StarDist2D

from core import ImageStorage, ImageWrapper
from core.cellprofiler_segmentation import segment_primary_objects
from core.image_utils import window_image_by_contrast
from core.project_naming import SEGMENTATION_BASE_NAME, prefix_with_project_name
from utils import resource_path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEGMENTATION_METHOD_STARDIST = "StarDist"
SEGMENTATION_METHOD_PRIMARY_OBJECTS = "CellProfiler-like"


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class StarDist(QThread):
    stardist_done = pyqtSignal(ImageWrapper, bool, str)
    cell_image_set = pyqtSignal(str, str)
    progress = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)
    cell_channel = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._state_lock = threading.Lock()
        self.protein_channels = None
        self.np_image = None
        self.project_name = None
        self.is_temp_project = False
        self.source_uuid = None
        self._last_result_source_uuid = None
        self.params = {
            "channel": "Channel 1",
            "segmentation_method": SEGMENTATION_METHOD_STARDIST,
            "model": "2D_versatile_fluo",
            "percentile_low": 3,
            "percentile_high": 99.80,
            "prob_threshold": 0.48,
            "nms_threshold": 0.3,
            "scale": 1.0,
            "n_tiles": 0,
            "radius": 5,
            "enable_dilation": True,
            "use_contrasted_image": False,
            "min_size": 60,
            "max_size": 180,
            # CellProfiler-like advanced settings
            "threshold_method": "MCT",
            "threshold_scope": "Global",
            "threshold_smoothing_scale": 1.3488,
            "threshold_correction_factor": 1.0,
            "threshold_lower_bound": 0.0,
            "threshold_upper_bound": 1.0,
            "manual_threshold": 0.5,
            "two_class_otsu": True,
            "assign_middle_to_foreground": True,
            "object_fraction": 0.2,
            "lower_outlier_fraction": 0.05,
            "upper_outlier_fraction": 0.05,
            "averaging_method": "Mean",
            "variance_method": "Standard deviation",
            "number_of_deviations": 2.0,
            "adaptive_window_size": 50,
            "fill_holes_after_thresholding": True,
            "fill_holes_after_declumping": True,
            "automatic_smoothing": True,
            "smoothing_filter_size": 10.0,
            "automatic_maxima_suppression": True,
            "maxima_suppression_size": 7.0,
            "low_res_maxima": True,
            "exclude_border_objects": True,
        }
        self.aligned = False
        self.current_model = ""

    def load_cell_image(self, arr):
        self.cell_image = arr
        self.aligned = True

    def __get_cell_image(self):
        if self.protein_channels is None and self.np_image is not None:
            return self.np_image
        if self.protein_channels and self.np_image is None:
            wrapper = self.protein_channels.get(self.params["channel"])
            if wrapper is not None:
                return wrapper.data
        return None

    def _window_image_by_contrast(self, image: np.ndarray, contrast_min, contrast_max):
        return window_image_by_contrast(image, contrast_min, contrast_max)

    def _resolve_segmentation_input(
        self, params=None, protein_channels=None, np_image=None
    ):
        if params is None:
            params = self.params
        if protein_channels is None:
            protein_channels = self.protein_channels
        if np_image is None:
            np_image = self.np_image

        wrapper = None
        image = None

        if protein_channels is not None and np_image is None:
            wrapper = protein_channels.get(params["channel"])
            if wrapper is not None:
                image = wrapper.data
        elif protein_channels is None and np_image is not None:
            image = np_image

        if image is None:
            return None

        if not params.get("use_contrasted_image", False):
            return image

        if wrapper is None:
            return image

        contrast_min = getattr(wrapper, "contrast_min", None)
        contrast_max = getattr(wrapper, "contrast_max", None)
        if contrast_min is None or contrast_max is None:
            return image

        return self._window_image_by_contrast(image, contrast_min, contrast_max)

    def run(self):
        self._cancel_requested = False  # reset each run
        with self._state_lock:
            run_params = dict(self.params)
            run_protein_channels = self.protein_channels
            run_np_image = self.np_image
            run_project_name = self.project_name
            run_is_temp_project = self.is_temp_project
            run_source_uuid = self.source_uuid

        cell_image = self._resolve_segmentation_input(
            params=run_params,
            protein_channels=run_protein_channels,
            np_image=run_np_image,
        )
        if cell_image is None:
            self._fatal_error_message("No cell image available for processing")
            return
        assert isinstance(cell_image, np.ndarray), "cell_image must be a numpy array"

        segmentation_method = run_params.get(
            "segmentation_method", SEGMENTATION_METHOD_STARDIST
        )
        if segmentation_method == SEGMENTATION_METHOD_PRIMARY_OBJECTS:
            self._run_primary_objects(
                cell_image,
                run_params,
                run_project_name,
                run_is_temp_project,
                run_source_uuid,
            )
            return

        self._run_stardist(
            cell_image,
            run_params,
            run_project_name,
            run_is_temp_project,
            run_source_uuid,
        )

    def cancel(self):
        self._cancel_requested = True
        self.progress.emit(99, "Cancelling...")

    def save_image(self):
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save File", "image.png", "*.png;;*.jpg;;*.tif;; All Files(*)"
        )
        if self.stardist_labels_grayscale is not None:
            Image.fromarray(self.stardist_labels_grayscale).save(file_name)
        else:
            self.error_signal.emit("Cannot save. No stardist labels available")

    def change_cmap(self):
        pass

    def generate_lut(self, cmap: str):
        label_range = np.linspace(0, 1, 256)
        return np.uint8(colormaps[cmap](label_range)[:, 2::-1] * 256).reshape(256, 1, 3)

    def label2rgb(self, labels, lut):
        return cv.LUT(cv.merge((labels, labels, labels)), lut)

    def update_channels(self, protein_channels, _):
        with self._state_lock:
            self.np_image = None
            self.protein_channels = protein_channels

    def set_protein_image(
        self, protein_channels, channel="Channel 1", name=None, source_uuid=None
    ):
        with self._state_lock:
            self.protein_channels = protein_channels
            self.params["channel"] = channel
            if source_uuid is not None:
                self.source_uuid = str(source_uuid)
                ImageStorage().add_data(
                    "seg_source_uuid", {"value": str(source_uuid), "channel": channel}
                )
        self.cell_image_set.emit(name, channel)
        with self._state_lock:
            self.np_image = None

    def set_image_to_process(self, np_image):
        with self._state_lock:
            self.protein_channels = None
            self.np_image = np_image

    def set_project_context(self, project_name, is_temp_project=False):
        with self._state_lock:
            self.project_name = project_name
            self.is_temp_project = bool(is_temp_project)

    def set_source_uuid(self, source_uuid):
        with self._state_lock:
            self.source_uuid = str(source_uuid) if source_uuid is not None else None

    @property
    def last_result_source_uuid(self):
        with self._state_lock:
            return self._last_result_source_uuid

    def set_channel(self, channel):
        with self._state_lock:
            self.params["channel"] = channel

    def set_segmentation_method(self, method):
        normalized = str(method).strip()
        with self._state_lock:
            if normalized == SEGMENTATION_METHOD_PRIMARY_OBJECTS:
                self.params["segmentation_method"] = SEGMENTATION_METHOD_PRIMARY_OBJECTS
                return
            self.params["segmentation_method"] = SEGMENTATION_METHOD_STARDIST

    def set_model(self, model):
        with self._state_lock:
            self.params["model"] = model

    def set_percentile_low(self, value):
        with self._state_lock:
            self.params["percentile_low"] = value

    def set_percentile_high(self, value):
        with self._state_lock:
            self.params["percentile_high"] = value

    def set_prob_thresh(self, value):
        with self._state_lock:
            self.params["prob_threshold"] = value

    def set_number_tiles(self, value):
        with self._state_lock:
            self.params["n_tiles"] = value

    def set_dilation_radius(self, value):
        with self._state_lock:
            self.params["radius"] = value

    def set_nms_thresh(self, value):
        with self._state_lock:
            self.params["nms_threshold"] = value

    def set_scale(self, value):
        scale = float(value)
        with self._state_lock:
            self.params["scale"] = scale if scale > 0 else 1.0

    def set_num_tiles(self, value):
        with self._state_lock:
            self.params["n_tiles"] = value

    def set_dialation_radisu(self, value):
        with self._state_lock:
            self.params["radius"] = value

    def set_enable_dilation(self, enabled):
        with self._state_lock:
            self.params["enable_dilation"] = bool(enabled)

    def set_use_contrasted_image(self, enabled):
        with self._state_lock:
            self.params["use_contrasted_image"] = bool(enabled)

    def set_min_size(self, value):
        parsed = int(value)
        parsed = max(1, parsed)
        with self._state_lock:
            self.params["min_size"] = parsed

    def set_max_size(self, value):
        parsed = int(value)
        parsed = max(1, parsed)
        with self._state_lock:
            self.params["max_size"] = parsed

    # CellProfiler-like advanced setters

    def set_threshold_method(self, value):
        with self._state_lock:
            self.params["threshold_method"] = str(value)

    def set_threshold_scope(self, value):
        with self._state_lock:
            self.params["threshold_scope"] = str(value)

    def set_threshold_smoothing_scale(self, value):
        with self._state_lock:
            self.params["threshold_smoothing_scale"] = float(value)

    def set_threshold_correction_factor(self, value):
        with self._state_lock:
            self.params["threshold_correction_factor"] = float(value)

    def set_threshold_lower_bound(self, value):
        with self._state_lock:
            self.params["threshold_lower_bound"] = float(value)

    def set_threshold_upper_bound(self, value):
        with self._state_lock:
            self.params["threshold_upper_bound"] = float(value)

    def set_manual_threshold(self, value):
        with self._state_lock:
            self.params["manual_threshold"] = float(value)

    def set_two_class_otsu(self, value):
        with self._state_lock:
            self.params["two_class_otsu"] = str(value) == "Two classes"

    def set_assign_middle_to_foreground(self, value):
        with self._state_lock:
            self.params["assign_middle_to_foreground"] = str(value) == "Foreground"

    def set_object_fraction(self, value):
        with self._state_lock:
            self.params["object_fraction"] = float(value)

    def set_lower_outlier_fraction(self, value):
        with self._state_lock:
            self.params["lower_outlier_fraction"] = float(value)

    def set_upper_outlier_fraction(self, value):
        with self._state_lock:
            self.params["upper_outlier_fraction"] = float(value)

    def set_averaging_method(self, value):
        with self._state_lock:
            self.params["averaging_method"] = str(value)

    def set_variance_method(self, value):
        with self._state_lock:
            self.params["variance_method"] = str(value)

    def set_number_of_deviations(self, value):
        with self._state_lock:
            self.params["number_of_deviations"] = float(value)

    def set_adaptive_window_size(self, value):
        with self._state_lock:
            self.params["adaptive_window_size"] = int(value)

    def set_fill_holes_after_thresholding(self, enabled):
        with self._state_lock:
            self.params["fill_holes_after_thresholding"] = bool(enabled)

    def set_fill_holes_after_declumping(self, enabled):
        with self._state_lock:
            self.params["fill_holes_after_declumping"] = bool(enabled)

    def set_automatic_smoothing(self, enabled):
        with self._state_lock:
            self.params["automatic_smoothing"] = bool(enabled)

    def set_smoothing_filter_size(self, value):
        with self._state_lock:
            self.params["smoothing_filter_size"] = float(value)

    def set_automatic_maxima_suppression(self, enabled):
        with self._state_lock:
            self.params["automatic_maxima_suppression"] = bool(enabled)

    def set_maxima_suppression_size(self, value):
        with self._state_lock:
            self.params["maxima_suppression_size"] = float(value)

    def set_low_res_maxima(self, enabled):
        with self._state_lock:
            self.params["low_res_maxima"] = bool(enabled)

    def set_exclude_border_objects(self, enabled):
        with self._state_lock:
            self.params["exclude_border_objects"] = bool(enabled)

    def set_invert_image(self, enabled):
        with self._state_lock:
            self.params["invert_image"] = bool(enabled)

    def _fatal_error_message(self, msg):
        self.error_signal.emit(msg)
        self.progress.emit(100, "")

    def _prepare_model_input(self, image: np.ndarray, model):
        axes = model.config.axes
        if "C" not in axes:
            return image

        expected_ndim = len(axes)
        channel_axis = axes.index("C")
        expected_channels = int(model.config.n_channel_in)

        img = image
        if img.ndim == expected_ndim - 1:
            if expected_channels == 1:
                return img
            img = np.expand_dims(img, axis=channel_axis)
            return np.repeat(img, expected_channels, axis=channel_axis)

        if img.ndim != expected_ndim:
            raise ValueError(
                f"Input image has {img.ndim} dims, but model expects {expected_ndim - 1} or {expected_ndim} dims "
                f"(axes={axes})."
            )

        in_channels = img.shape[channel_axis]
        if in_channels == expected_channels:
            return img
        if in_channels == 1 and expected_channels > 1:
            return np.repeat(img, expected_channels, axis=channel_axis)
        if expected_channels == 1 and in_channels > 1:
            return np.mean(img, axis=channel_axis)

        raise ValueError(
            f"Input channels ({in_channels}) are incompatible with model expected channels ({expected_channels}) "
            f"for axes {axes}."
        )

    def _sort_labels_by_centroid(self, labels: np.ndarray) -> np.ndarray:
        from scipy.ndimage import center_of_mass

        unique = np.unique(labels)
        unique = unique[unique != 0]
        if len(unique) == 0:
            return labels
        centroids = center_of_mass(
            np.ones_like(labels, dtype=np.float32), labels, unique
        )
        order = np.lexsort(([c[1] for c in centroids], [c[0] for c in centroids]))
        remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for new_id, old_id in enumerate(unique[order], start=1):
            remap[old_id] = new_id
        return remap[labels]

    def _emit_segmentation_result(
        self, project_name, is_temp_project, source_uuid=None
    ):
        self.stardist_labels_grayscale = self._sort_labels_by_centroid(
            self.stardist_labels_grayscale
        )
        result = ImageWrapper(
            self.stardist_labels_grayscale, name="Channel 1", cmap="gray"
        )
        label_name = prefix_with_project_name(
            SEGMENTATION_BASE_NAME,
            project_name,
            is_temp_project=is_temp_project,
        )
        with self._state_lock:
            self._last_result_source_uuid = (
                str(source_uuid) if source_uuid is not None else None
            )
        self.stardist_done.emit(result, True, label_name)

    def _run_primary_objects(
        self,
        cell_image: np.ndarray,
        params: dict,
        project_name,
        is_temp_project,
        source_uuid=None,
    ):
        self.progress.emit(0, "Starting CellProfiler-like segmentation")
        min_size = int(params.get("min_size", 60))
        max_size = int(params.get("max_size", 180))
        if max_size < min_size:
            min_size, max_size = max_size, min_size
            params["min_size"] = min_size
            params["max_size"] = max_size

        cp_settings = {
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
        }

        seg_settings = {
            **cp_settings,
            "min_size": min_size,
            "max_size": max_size,
            "enable_dilation": params.get("enable_dilation", True),
            "radius": params.get("radius", 5),
            "invert_image": bool(params.get("invert_image", False)),
        }

        try:
            self.stardist_labels_grayscale = segment_primary_objects(cell_image, seg_settings)
        except Exception as exc:
            self._fatal_error_message(
                f"CellProfiler-like segmentation failed: {exc}. You may need to install pocl-opencl-icd (WSL2 users)."
            )
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        self.progress.emit(100, "CellProfiler-like segmentation done")
        self._emit_segmentation_result(project_name, is_temp_project, source_uuid)

    def _run_stardist(
        self,
        cell_image: np.ndarray,
        params: dict,
        project_name,
        is_temp_project,
        source_uuid=None,
    ):
        self.progress.emit(0, "Starting StarDist")
        if self.current_model != str(params["model"]):
            try:
                if params["model"] != "2D_versatile_fluo":
                    self._fatal_error_message(f"Unsupported model: {params['model']}")
                    return
                self.model = StarDist2D.from_openvino(
                    model_dir=resource_path(f"assets/{params['model']}")
                )
            except Exception as exc:
                self._fatal_error_message(f"Model load failed: {exc}")
                return
            self.current_model = str(params["model"])
        model = self.model

        self.progress.emit(10, "Model loaded")

        try:
            model_input = self._prepare_model_input(cell_image, model)
        except ValueError as exc:
            self._fatal_error_message(str(exc))
            return

        norm_img = normalize(
            model_input,
            params["percentile_low"],
            params["percentile_high"],
        )

        guess_tiles = params["n_tiles"]
        if guess_tiles == 0:
            guess_tiles = model._guess_n_tiles(model_input)

        total_tiles = int(guess_tiles[0] * guess_tiles[1])

        try:
            labels_gen = model._predict_instances_generator(
                norm_img,
                prob_thresh=params["prob_threshold"],
                nms_thresh=params["nms_threshold"],
                scale=params["scale"],
                n_tiles=guess_tiles,
            )
        except Exception as exc:
            self._fatal_error_message(f"Prediction error: {exc}")
            return

        stardist_labels = None
        for i, (labels, *_) in enumerate(labels_gen):
            if self._cancel_requested:
                self.progress.emit(100, "Cancelled")
                return
            stardist_labels = labels
            pct = 10 + int(70 * max(0, (i - 2) / total_tiles))
            self.progress.emit(pct, f"Processing tile {max(0, i - 2)}/{total_tiles}")

        if stardist_labels is None:
            self._fatal_error_message("No labels produced")
            return

        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return

        self.stardist_labels_grayscale = np.asarray(stardist_labels, dtype=np.int32)

        if params.get("enable_dilation", True):
            self.progress.emit(95, "Dilating")
            try:
                # Preserve large label ids for downstream processing.
                self.stardist_labels_grayscale = np.asarray(
                    dilate_labels(stardist_labels, radius=params["radius"]),
                    dtype=np.int32,
                )
            except Exception as exc:
                self._fatal_error_message(
                    f"Error during dilation: {exc}. You may need to install pocl-opencl-icd (WSL2 users)."
                )
                return

            if self._cancel_requested:
                self.progress.emit(100, "Cancelled")
                return

        self.progress.emit(100, "StarDist Done")
        self._emit_segmentation_result(project_name, is_temp_project, source_uuid)
