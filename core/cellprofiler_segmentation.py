import numpy as np
import scipy.ndimage
import skimage.filters
import skimage.segmentation
import centrosome.cpmorphology
import centrosome.smooth
from centrosome.threshold import (
    get_otsu_threshold,
    get_mog_threshold,
    get_background_threshold,
    get_robust_background_threshold,
    get_ridler_calvard_threshold,
    get_kapur_threshold,
    get_adaptive_threshold,
)


PRIMARY_OBJECT_SETTINGS = {
    "threshold_method": "MCT",
    "threshold_scope": "Global",
    "threshold_smoothing_scale": 1.3488,
    "threshold_correction_factor": 1.0,
    "threshold_range": (0.0, 1.0),
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

# Centrosome adaptive threshold uses string method identifiers.
_CENTROSOME_METHOD_MAP = {
    "MCT": "MCT",
    "Otsu": "Otsu",
    "MoG": "MoG",
    "Background": "Background",
    "RobustBackground": "RobustBackground",
    "RidlerCalvard": "RidlerCalvard",
    "Kapur": "Kapur",
}

_AVERAGING_FN = {
    "Mean": np.mean,
    "Median": np.median,
    "Mode": None,  # handled separately
}

_VARIANCE_FN = {
    "Standard deviation": np.std,
    "MAD": None,  # handled separately
}


def _binned_mode(data):
    """Compute mode via histogram binning (matches centrosome approach)."""
    if data.size == 0:
        return 0.0
    nbins = 256
    hist, edges = np.histogram(data, bins=nbins)
    idx = np.argmax(hist)
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _mad(data):
    """Median absolute deviation."""
    med = np.median(data)
    return np.median(np.abs(data - med)) * 1.4826


def _li_threshold(image, mask):
    """Minimum cross-entropy threshold (Li's method)."""
    pixels = image[mask]
    if pixels.size == 0:
        return 0.0
    if np.all(pixels == pixels.ravel()[0]):
        return float(pixels.ravel()[0])

    unique = np.unique(pixels)
    if unique.size > 1:
        tol = max(float(np.min(np.diff(unique))) / 2.0, 0.5 / 65536.0)
    else:
        tol = 0.5 / 65536.0
    return float(skimage.filters.threshold_li(pixels, tolerance=tol))


def _compute_global_threshold(image, mask, method, settings):
    """Compute a single global threshold value using the specified method."""
    if method == "Manual":
        return float(settings.get("manual_threshold", 0.5))

    if method == "MCT":
        return _li_threshold(image, mask)

    if method == "Otsu":
        return float(
            get_otsu_threshold(
                image,
                mask,
                two_class_otsu=settings.get("two_class_otsu", True),
                assign_middle_to_foreground=settings.get(
                    "assign_middle_to_foreground", True
                ),
            )
        )

    if method == "MoG":
        return float(
            get_mog_threshold(
                image,
                mask,
                object_fraction=settings.get("object_fraction", 0.2),
            )
        )

    if method == "Background":
        return float(get_background_threshold(image, mask))

    if method == "RobustBackground":
        avg_name = settings.get("averaging_method", "Mean")
        avg_fn = _AVERAGING_FN.get(avg_name)
        if avg_fn is None:
            avg_fn = _binned_mode

        var_name = settings.get("variance_method", "Standard deviation")
        var_fn = _VARIANCE_FN.get(var_name)
        if var_fn is None:
            var_fn = _mad

        return float(
            get_robust_background_threshold(
                image,
                mask,
                lower_outlier_fraction=settings.get("lower_outlier_fraction", 0.05),
                upper_outlier_fraction=settings.get("upper_outlier_fraction", 0.05),
                deviations_above_average=settings.get("number_of_deviations", 2.0),
                average_fn=avg_fn,
                variance_fn=var_fn,
            )
        )

    if method == "RidlerCalvard":
        return float(get_ridler_calvard_threshold(image, mask))

    if method == "Kapur":
        return float(get_kapur_threshold(image, mask))

    # Fallback to MCT
    return _li_threshold(image, mask)


def _compute_threshold(image, mask, settings):
    """Compute threshold value or per-pixel threshold array.

    Returns a scalar (Global scope) or 2D array (Adaptive scope).
    """
    method = settings.get("threshold_method", "MCT")
    scope = settings.get("threshold_scope", "Global")
    correction = float(settings.get("threshold_correction_factor", 1.0))
    tmin, tmax = settings.get("threshold_range", (0.0, 1.0))

    if method == "Manual":
        return float(settings.get("manual_threshold", 0.5))

    if scope == "Adaptive":
        window_size = int(settings.get("adaptive_window_size", 50))
        global_thresh = _compute_global_threshold(image, mask, method, settings)
        try:
            adaptive_thresh = get_adaptive_threshold(
                method,
                image,
                global_thresh,
                mask=mask,
                adaptive_window_size=window_size,
                two_class_otsu=settings.get("two_class_otsu", True),
                assign_middle_to_foreground=settings.get(
                    "assign_middle_to_foreground", True
                ),
                object_fraction=settings.get("object_fraction", 0.2),
                lower_outlier_fraction=settings.get("lower_outlier_fraction", 0.05),
                upper_outlier_fraction=settings.get("upper_outlier_fraction", 0.05),
                deviations_above_average=settings.get("number_of_deviations", 2.0),
            )
        except Exception:
            # Fall back to global if adaptive fails
            adaptive_thresh = global_thresh

        adaptive_thresh = np.asarray(adaptive_thresh, dtype=np.float64)
        adaptive_thresh *= correction
        return np.clip(adaptive_thresh, float(tmin), float(tmax))

    # Global scope
    threshold = _compute_global_threshold(image, mask, method, settings)
    threshold *= correction
    return min(max(threshold, float(tmin)), float(tmax))


def _apply_threshold(image, threshold, mask, smoothing_scale):
    """Apply threshold to image, optionally smoothing first.

    threshold can be a scalar or a 2D per-pixel array (adaptive).
    """
    if smoothing_scale == 0:
        return (image >= threshold) & mask, 0.0

    sigma = float(smoothing_scale) / 0.6744 / 2.0
    blurred = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
        mask,
    )
    return (blurred >= threshold) & mask, sigma


def _calc_declump_smoothing_filter_size(min_diameter):
    return 2.35 * float(min_diameter) / 3.5


def _smooth_for_declumping(image, mask, filter_size):
    if filter_size == 0:
        return image

    sigma = filter_size / 2.35
    radius = max(int(float(filter_size) / 2.0), 1)
    filt = (
        1.0
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(-0.5 * np.arange(-radius, radius + 1) ** 2 / sigma**2)
    )

    def fgaussian(arr):
        out = scipy.ndimage.convolve1d(arr, filt, axis=0, mode="constant")
        return scipy.ndimage.convolve1d(out, filt, axis=1, mode="constant")

    edge = fgaussian(mask.astype(float))
    masked = image.copy()
    masked[~mask] = 0
    smoothed = fgaussian(masked)
    masked[mask] = smoothed[mask] / edge[mask]
    return masked


def _get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    if image_resize_factor < 1.0:
        shape = np.array(image.shape) * image_resize_factor
        ij = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float) / image_resize_factor
        resized_image = scipy.ndimage.map_coordinates(image, ij)
        resized_labels = scipy.ndimage.map_coordinates(labeled_image, ij, order=0).astype(
            labeled_image.dtype
        )
    else:
        resized_image = image
        resized_labels = labeled_image

    binary_maxima = centrosome.cpmorphology.is_local_maximum(
        resized_image, resized_labels, maxima_mask
    )
    binary_maxima[resized_image <= 0] = 0

    if image_resize_factor < 1.0:
        inv_factor = float(image.shape[0]) / float(binary_maxima.shape[0])
        ij = np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float) / inv_factor
        binary_maxima = scipy.ndimage.map_coordinates(binary_maxima.astype(float), ij) > 0.5

    return centrosome.cpmorphology.binary_shrink(binary_maxima)


def _filter_on_border(labels):
    if labels.max() <= 0:
        return labels

    border_labels = np.concatenate(
        [
            labels[0, :],
            labels[:, 0],
            labels[-1, :],
            labels[:, -1],
        ]
    )
    border_labels = np.unique(border_labels[border_labels > 0])
    if border_labels.size == 0:
        return labels
    labels[np.isin(labels, border_labels)] = 0
    return labels


def _filter_on_size(labels, min_diameter, max_diameter):
    object_count = int(labels.max())
    if object_count <= 0:
        return labels

    areas = scipy.ndimage.sum(
        np.ones(labels.shape),
        labels,
        np.arange(0, object_count + 1, dtype=np.int32),
    )
    areas = np.asarray(areas, dtype=float)

    min_allowed_area = np.pi * (float(min_diameter) ** 2) / 4.0
    max_allowed_area = np.pi * (float(max_diameter) ** 2) / 4.0

    area_image = areas[labels]
    labels[(area_image < min_allowed_area) | (area_image > max_allowed_area)] = 0
    return labels


def identify_primary_objects(image_2d, min_size, max_size, settings=None):
    """CellProfiler-like primary object identification.

    Parameters
    ----------
    image_2d : 2D numpy array
    min_size : int — minimum object diameter in pixels
    max_size : int — maximum object diameter in pixels
    settings : dict or None — algorithm settings (defaults to PRIMARY_OBJECT_SETTINGS)
    """
    if settings is None:
        settings = PRIMARY_OBJECT_SETTINGS

    image = np.asarray(image_2d, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if np.nanmax(image) > 1.0 or np.nanmin(image) < 0.0:
        lo, hi = np.percentile(image, [0.5, 99.5])
        if hi > lo:
            image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        else:
            image = np.clip(image, 0.0, 1.0)

    mask = np.ones(image.shape, dtype=bool)

    final_threshold = _compute_threshold(image, mask, settings)

    binary, _ = _apply_threshold(
        image=image,
        threshold=final_threshold,
        mask=mask,
        smoothing_scale=settings.get("threshold_smoothing_scale", 1.3488),
    )

    if settings.get("fill_holes_after_thresholding", True):
        def size_fn(size, _):
            return size < (float(max_size) * float(max_size))
        binary = centrosome.cpmorphology.fill_labeled_holes(binary, size_fn=size_fn)

    labeled, _ = scipy.ndimage.label(binary, np.ones((3, 3), dtype=bool))

    if labeled.max() > 0:
        if settings.get("automatic_smoothing", True):
            declump_filter_size = _calc_declump_smoothing_filter_size(min_size)
        else:
            declump_filter_size = float(
                settings.get("smoothing_filter_size", 10.0)
            )

        blurred = _smooth_for_declumping(image, mask, declump_filter_size)

        if min_size > 10 and settings.get("low_res_maxima", True):
            image_resize_factor = 10.0 / float(min_size)
            if settings.get("automatic_maxima_suppression", True):
                maxima_suppression_size = 7.0
            else:
                maxima_suppression_size = float(
                    settings.get("maxima_suppression_size", 7.0)
                )
        else:
            image_resize_factor = 1.0
            if settings.get("automatic_maxima_suppression", True):
                maxima_suppression_size = float(min_size) / 1.5
            else:
                maxima_suppression_size = float(
                    settings.get("maxima_suppression_size", 7.0)
                )

        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )
        maxima = _get_maxima(blurred, labeled, maxima_mask, image_resize_factor)
        labeled_maxima, object_count = scipy.ndimage.label(
            maxima, np.ones((3, 3), dtype=bool)
        )

        if object_count > 0:
            watershed_image = 1.0 - image
            markers_dtype = (
                np.int16 if object_count < np.iinfo(np.int16).max else np.int32
            )
            markers = np.zeros(watershed_image.shape, markers_dtype)
            markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
            watershed_boundaries = skimage.segmentation.watershed(
                connectivity=np.ones((3, 3), dtype=bool),
                image=watershed_image,
                markers=markers,
                mask=labeled != 0,
            )
            labeled = -watershed_boundaries

    if settings.get("exclude_border_objects", True) and labeled.max() > 0:
        labeled = _filter_on_border(labeled)
    if labeled.max() > 0:
        labeled = _filter_on_size(labeled, min_size, max_size)

    if settings.get("fill_holes_after_declumping", True) and labeled.max() > 0:
        labeled = centrosome.cpmorphology.fill_labeled_holes(labeled)

    labeled, _ = centrosome.cpmorphology.relabel(labeled)
    # Keep labels in int32 to avoid the uint16 instance-id ceiling (65,535).
    return np.asarray(labeled, dtype=np.int32)


def segment_primary_objects(image: np.ndarray, settings: dict) -> np.ndarray:
    """identify_primary_objects + optional inversion/dilation. Pure function, no Qt/signals."""
    min_size = int(settings["min_size"])
    max_size = int(settings["max_size"])
    if max_size < min_size:
        min_size, max_size = max_size, min_size

    if image.ndim == 3 and image.shape[-1] == 3:
        import cv2 as _cv2
        image = _cv2.cvtColor(image.astype(np.float32), _cv2.COLOR_RGB2GRAY)
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)

    if settings.get("invert_image", False):
        image = 1.0 - image.astype(np.float32)

    labels = identify_primary_objects(image, min_size=min_size, max_size=max_size, settings=settings)

    if settings.get("enable_dilation", True) and labels.max() > 0:
        from pyclesperanto import dilate_labels
        radius = settings.get("dilation_radius") or settings.get("radius", 5)
        labels = np.asarray(dilate_labels(labels, radius=radius), dtype=np.int32)

    return labels
