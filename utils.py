import logging
import os
import sys

import cv2
import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from PyQt6.QtGui import QImage, QPixmap
from skimage.filters import threshold_otsu
logger = logging.getLogger(__name__)

_CV2_WARP_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def to_cv2_friendly(arr: np.ndarray) -> np.ndarray:
    """Return an array that cv2 warp/resize can consume directly.

    Uses issubdtype rules rather than a hard-coded allowlist so new or unusual
    numpy dtypes (np.int32, np.float16, bool, etc.) still work.
    """
    if arr.dtype in _CV2_WARP_DTYPES:
        return arr
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.uint8, copy=False)
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32)


# Import extracted functions from core.image_utils

def numpy_to_qimage(array: np.ndarray) -> QImage:
    if not array.data.contiguous:
        array = np.ascontiguousarray(array)

    qimage = None
    if len(array.shape) == 2:
        # Grayscale image
        height, width = array.shape
        format = (
            QImage.Format.Format_Grayscale16
            if array.dtype == np.uint16
            else QImage.Format.Format_Grayscale8
        )
        bytes_per_pixel = 2 if array.dtype == np.uint16 else 1
        bytes_per_line = width * bytes_per_pixel  # uint8
        qimage = QImage(array.data, width, height, bytes_per_line, format)
    elif len(array.shape) == 3:
        height, width, channels = array.shape
        if channels == 3:
            # RGB image
            qimage = QImage(
                array.data, width, height, width * channels, QImage.Format.Format_RGB888
            )
        elif channels == 4:
            # RGBA image
            qimage = QImage(
                array.data,
                width,
                height,
                width * channels,
                QImage.Format.Format_RGBA8888,
            )
    else:
        raise ValueError("Unsupported array shape: {}".format(array.shape))
    if qimage is None:
        raise ValueError("Failed to create QImage from numpy array")
    qimage.ndarray = array  # keep numpy buffer alive as long as QImage lives
    return qimage


def qimage_to_numpy(qimage: QImage):
    # Ensure the QImage format is suitable for conversion
    ptr = qimage.bits()
    width = qimage.width()
    height = qimage.height()

    if qimage.format() == QImage.Format.Format_Grayscale8:
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(width * height)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width)
        return arr
    elif qimage.format() == QImage.Format.Format_Grayscale16:
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(width * height * 2)
        arr = np.frombuffer(ptr, dtype=np.uint16).reshape(height, width)
        return arr
    elif qimage.format() in [
        QImage.Format.Format_RGB32,
        QImage.Format.Format_ARGB32,
        QImage.Format.Format_RGBX8888,
        QImage.Format.Format_RGBA8888,
        QImage.Format.Format_ARGB32_Premultiplied,
    ]:
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(width * height * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 4)
        return arr
    elif qimage.format() == QImage.Format.Format_RGB888:
        assert ptr is not None, "QImage bits() returned None"
        ptr.setsize(width * height * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 3)
        return arr
    else:
        raise ValueError(f"Unsupported QImage format: {qimage.format()}")



def to_pixmap(data: QPixmap | np.ndarray | QImage):
    """Sends a pixmap to the canvas for display"""
    # convert pixmap to pixmapItem
    pixmap = None
    if isinstance(data, QPixmap):
        pixmap = data
    elif isinstance(data, QImage):
        pixmap = QPixmap(data)
    elif isinstance(data, np.ndarray):
        pixmap = QPixmap(numpy_to_qimage(data))
    assert pixmap is not None
    return pixmap


def is_grayscale(image: np.ndarray) -> bool:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return False
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return True
    else:
        raise ValueError("Image format not recognized")


def to_uint8_rgb(image):
    """Convert RGB image to uint8 with per-channel scaling"""
    if image.dtype == np.uint8:
        return image
    img_float = image.astype(np.float32)
    img_uint8 = np.zeros_like(image, dtype=np.uint8)
    for c in range(image.shape[2]):
        ch = img_float[..., c]
        if ch.max() > ch.min():
            img_uint8[..., c] = (
                (ch - ch.min()) * 255.0 / (ch.max() - ch.min())
            ).astype(np.uint8)
    return img_uint8


def to_uint8(image):
    """Convert image to uint8 with proper scaling"""
    # Check if image is already uint8
    if image.dtype == np.uint8:
        return image
    if not is_grayscale(image):
        return to_uint8_rgb(image)
    # Convert to float and scale to 0-255
    img_float = image.astype(np.float32)
    if img_float.max() > img_float.min():  # Check to avoid division by zero
        img_norm = (img_float - img_float.min()) * (
            255.0 / (img_float.max() - img_float.min())
        )
        return img_norm.astype(np.uint8)
    else:
        logger.warning("Image has no variation, returning zeros")
        return np.zeros_like(image, dtype=np.uint8)


def calculate_ncc(img1, img2):
    """
    Calculate NCC (Normalized Cross-Correlation) between two images.

    Args:
        img1: First image (reference/target)
        img2: Second image (aligned)

    Returns:
        NCC value between -1 and 1 (1 = perfect correlation)
    """
    try:
        # Ensure images have the same shape
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]

        # Convert to float to avoid overflow
        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)

        # Flatten images
        img1_flat = img1_float.flatten()
        img2_flat = img2_float.flatten()

        # Calculate means
        mean1 = np.mean(img1_flat)
        mean2 = np.mean(img2_flat)

        # Center the data
        img1_centered = img1_flat - mean1
        img2_centered = img2_flat - mean2

        # Calculate NCC
        numerator = np.sum(img1_centered * img2_centered)
        denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))

        if denominator == 0:
            return 0.0  # No correlation if one image is constant

        ncc = numerator / denominator
        return ncc

    except Exception as e:
        logger.error(f"Error calculating NCC: {str(e)}")
        return None


def to_uint16_from_uint8(image_uint8, original_min, original_max):
    """Rescale uint8 image back to original float32 or uint16 range."""
    image_float = image_uint8.astype(np.float32) / 255.0
    rescaled = image_float * (original_max - original_min) + original_min
    return rescaled.astype(np.uint16)


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    normalized_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    normalized_data = normalized_data.astype(np.uint8)
    return normalized_data


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def convert_image_to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    if is_grayscale(img):
        return img
    else:
        return rgb2gray(img)


def adjustContrast(img, alpha=5, beta=15):
    alpha = 5  # Contrast control
    beta = 15  # Brightness control
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# uint16 to uint8


def grayscale_to_agrb(img_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D grayscale image (0-255) to an ARGB image.

    - Alpha channel is img_2d normalized to [0, 1].
    - RGB channels are always [255, 255, 255].

    Returns: float32 array of shape (H, W, 4).
    """
    # Normalize alpha (0-1)
    alpha = img_2d.astype(np.float32) / 255.0
    alpha[alpha == 0] = np.nan
    alpha[alpha != np.nan] = 0.5

    # Create RGB channels (all 255)
    h, w = img_2d.shape
    # rgb = np.ones((h, w, 3), dtype=np.float32) * alpha*255.0
    rgb = np.stack((img_2d,) * 3, axis=-1, dtype=np.float32)

    # Stack into ARGB: (H, W, 4)
    argb = np.dstack([rgb, alpha])
    return argb


# def to_float64(arr: np.ndarray):
# if arr.dtype == np.uint16:




def gaussian_kernel_1d(sigma, radius=None):
    """Generate 1D Gaussian kernel."""
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def downsample(image: NDArray[np.float64], scale=0.5):
    """Downsample image by given scale factor."""
    step = int(round(1 / scale))
    return image[::step, ::step]


def adjust_contrast(
    img: NDArray[np.float32] | NDArray[np.float64], min_percentile=2, max_percentile=98
):
    """Adjust image contrast using percentile-based clipping for float images.

    Returns float32 in [0.0, 1.0] — float64 is unnecessary precision for
    downstream display / normalise-to-uint8 and doubles memory.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    minval = np.percentile(img, min_percentile)
    maxval = np.percentile(img, max_percentile)

    if maxval - minval < 1e-12:
        return np.zeros_like(img)

    img_adjusted = np.clip(img, minval, maxval)
    img_adjusted = (img_adjusted - minval) / (maxval - minval)
    return img_adjusted



def background_subtraction_with_histogram(image: np.ndarray) -> np.ndarray:
    """
    Implements the described method:
    1. Otsu threshold to extract background.
    2. Compute histogram of background pixels.
    3. Use the peak of background histogram as refined threshold.
    4. Subtract refined threshold from original image.

    https://www.nature.com/articles/s44303-025-00088-w#MOESM1
    """

    original_dtype = image.dtype
    img = image.astype(np.float32, copy=False)

    otsu_thresh = threshold_otsu(img)
    background_mask = img <= otsu_thresh

    background_pixels = img[background_mask]
    hist, bin_edges = np.histogram(background_pixels, bins=256, range=(0, img.max()))

    peak_idx = np.argmax(hist)
    refined_threshold = bin_edges[peak_idx]

    result = img - refined_threshold
    np.clip(result, 0, None, out=result)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        result = np.clip(result, info.min, info.max).astype(original_dtype, copy=False)
    elif np.issubdtype(original_dtype, np.floating):
        result = result.astype(original_dtype, copy=False)

    return result, otsu_thresh, refined_threshold, hist, bin_edges


def pad_to_shape(image, target_shape):
    """Pad image to target_shape with zeros (symmetric padding)"""
    pad_height = target_shape[0] - image.shape[0]
    pad_width = target_shape[1] - image.shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )


def make_same_shape(img1, img2):
    if img1.shape == img2.shape:
        return img1, img2
    max_height = max(img1.shape[0], img2.shape[0])
    max_width = max(img1.shape[1], img2.shape[1])

    return pad_to_shape(img1, (max_height, max_width)), pad_to_shape(
        img2, (max_height, max_width)
    )


def remove_padding(padded_image, original_shape):
    """Remove symmetric padding to restore original shape"""
    current_shape = padded_image.shape

    # Calculate padding that was added
    pad_height = current_shape[0] - original_shape[0]
    pad_width = current_shape[1] - original_shape[1]

    # Calculate crop coordinates (reverse of padding)
    pad_top = pad_height // 2
    pad_left = pad_width // 2

    # Extract the original region
    end_row = pad_top + original_shape[0]
    end_col = pad_left + original_shape[1]

    return padded_image[pad_top:end_row, pad_left:end_col]


def warp_image(img, transform_matrix) -> np.ndarray:
    """Warp an image using the given backward affine matrix (DST→SRC).

    Preserves the input dtype when cv2 supports it (uint8, uint16, int16,
    float32, float64). Promotes other integer/float dtypes to float32, and
    bool to uint8. No silent normalisation or uint8 conversion — callers
    that need a specific output dtype should do the cast themselves.
    """
    if transform_matrix is None:
        return img

    if transform_matrix.shape == (2, 3):
        M = transform_matrix
    elif transform_matrix.shape == (3, 3):
        M = transform_matrix[:2, :]
    else:
        raise ValueError("Transform matrix must be 2x3 or 3x3")

    original_dtype = img.dtype
    src = to_cv2_friendly(img)
    h, w = src.shape[:2]
    warped = cv2.warpAffine(
        src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0
    )
    if warped.dtype != original_dtype and original_dtype in _CV2_WARP_DTYPES:
        warped = warped.astype(original_dtype, copy=False)
    return warped


def _safe_warp_affine(
    img: np.ndarray,
    M: np.ndarray,
    out_shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    original_dtype = img.dtype
    _cv2_ok = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
    if original_dtype not in _cv2_ok:
        img = img.astype(np.float32)
    h, w = out_shape_hw if out_shape_hw else img.shape[:2]
    # By default, cv2.warpAffine expects a backward map (dst->src).
    # If M is a forward map (src->dst), we should either invert it or use the flag.
    # To be consistent with warp_image, we use the flag and expect M to be a backward map,
    # OR we follow the controller's expectation of forward maps and NO flag.
    # Given apply_ransac_affine_warp gets a forward map from estimateAffine2D,
    # we should NOT use the flag so warpAffine inverts it.
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return warped.astype(original_dtype) if warped.dtype != original_dtype else warped


def apply_ransac_affine_warp(
    img: np.ndarray,
    fixed_pts_colrow: np.ndarray,
    moving_pts_colrow: np.ndarray,
    ransac_threshold: float = 10.0,
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit affine with RANSAC and warp img. Returns (warped, {"M", "inliers", "reprojection_px"})."""
    src = moving_pts_colrow.astype(np.float32)
    dst = fixed_pts_colrow.astype(np.float32)
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
    )
    if M is None:
        raise RuntimeError("Affine RANSAC failed — too few inliers.")
    # estimateAffine2D returns a forward map (SRC→DST), which is what
    # cv2.warpAffine expects by default.
    warped = _safe_warp_affine(img, M, out_shape)
    proj = (M[:, :2] @ moving_pts_colrow.T + M[:, 2:]).T
    err = float(np.mean(np.linalg.norm(proj - fixed_pts_colrow, axis=1)))
    n_inliers = int(inliers.sum()) if inliers is not None else len(src)
    return warped, {"M": M, "inliers": n_inliers, "reprojection_px": err}


# Memory monitoring utility
# Dead code: This function is not used anywhere in the codebase.
def monitor_memory_usage():
    """Simple memory monitoring function."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        "percent": process.memory_percent(),
    }


def match_histograms(src_image, ref_histogram, bins=256):
    """
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/


    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """

    def calculate_cdf(histogram):
        """
        This method calculates the cumulative distribution function
        :param array histogram: The values of the histogram
        :return: normalized_cdf: The normalized cumulative distribution function
        :rtype: array
        """
        # Get the cumulative sum of the elements
        cdf = histogram.cumsum()

        # Normalize the cdf
        normalized_cdf = cdf / float(cdf.max())

        return normalized_cdf

    def calculate_lookup(src_cdf, ref_cdf):
        """
        This method creates the lookup table
        :param array src_cdf: The cdf for the source image
        :param array ref_cdf: The cdf for the reference image
        :return: lookup_table: The lookup table
        :rtype: array
        """
        lookup_table = np.zeros(256)
        lookup_val = 0
        for src_pixel_val in range(len(src_cdf)):
            for ref_pixel_val in range(len(ref_cdf)):
                if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                    lookup_val = ref_pixel_val
                    break
            lookup_table[src_pixel_val] = lookup_val
        return lookup_table

    # Split the images into the different color channels
    src_hist, _ = np.histogram(src_image.flatten(), bins)

    # Compute the normalized cdf for the source and reference image
    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_histogram)

    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    # Use the lookup function to transform the colors of the original
    # source image
    src_after_transform = cv2.LUT(src_image, lookup_table)
    image_after_matching = cv2.convertScaleAbs(src_after_transform)

    return image_after_matching


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def get_read_function(path: str):
    if path.endswith(".tif") or path.endswith(".tiff"):
        return tiff.imread
    elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return cv2.imread
    else:
        raise ValueError("Unsupported file type")


def get_save_function(path: str):
    if path.endswith(".tif") or path.endswith(".tiff"):
        return tiff.imwrite
    elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return cv2.imwrite
    else:
        raise ValueError("Unsupported file type")


def get_home_dir():
    return os.path.expanduser("~")


def get_desktop_dir():
    return os.path.join(get_home_dir(), "Desktop")


def get_most_recent_file(directory: str, ext: str = ".tif"):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(ext) and os.path.isfile(os.path.join(directory, f))
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_file_name(path: str):
    return os.path.basename(path)


def get_file_ext(path: str):
    return os.path.splitext(path)[1]


def get_file_dir(path: str):
    return os.path.dirname(path)


def get_file_name_without_ext(path: str):
    return os.path.splitext(get_file_name(path))[0]


def get_files_in_dir(directory: str, ext: str = ".tif"):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(ext) and os.path.isfile(os.path.join(directory, f))
    ]


def get_sub_dirs(directory: str):
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]


def create_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_shared_contrast_limits(images: list[np.ndarray]):
    min_val = min([np.min(img) for img in images])
    max_val = max([np.max(img) for img in images])
    return min_val, max_val


def get_shared_dtype(images: list[np.ndarray]):
    return np.find_common_type([img.dtype for img in images], [])


def get_shared_shape(images: list[np.ndarray]):
    shapes = [img.shape for img in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images have different shapes")
    return shapes[0]


def get_shared_ndim(images: list[np.ndarray]):
    ndims = [img.ndim for img in images]
    if len(set(ndims)) > 1:
        raise ValueError("Images have different number of dimensions")
    return ndims[0]


def get_shared_itemsize(images: list[np.ndarray]):
    itemsizes = [img.itemsize for img in images]
    if len(set(itemsizes)) > 1:
        raise ValueError("Images have different item sizes")
    return itemsizes[0]


def get_shared_strides(images: list[np.ndarray]):
    strides = [img.strides for img in images]
    if len(set(strides)) > 1:
        raise ValueError("Images have different strides")
    return strides[0]


def get_shared_flags(images: list[np.ndarray]):
    flags = [img.flags.c_contiguous for img in images]
    if len(set(flags)) > 1:
        raise ValueError("Images have different memory layout")
    return flags[0]


def check_images_are_compatible(images: list[np.ndarray]):
    get_shared_dtype(images)
    get_shared_shape(images)
    get_shared_ndim(images)
    get_shared_itemsize(images)
    get_shared_strides(images)
    get_shared_flags(images)
    return True


def check_image_is_grayscale(image: np.ndarray):
    if image.ndim == 2:
        return True
    elif image.ndim == 3 and image.shape[2] == 1:
        return True
    else:
        return False


def check_image_is_rgb(image: np.ndarray):
    if image.ndim == 3 and image.shape[2] == 3:
        return True
    else:
        return False


def check_image_is_rgba(image: np.ndarray):
    if image.ndim == 3 and image.shape[2] == 4:
        return True
    else:
        return False


def check_image_is_uint8(image: np.ndarray):
    if image.dtype == np.uint8:
        return True
    else:
        return False


def check_image_is_uint16(image: np.ndarray):
    if image.dtype == np.uint16:
        return True
    else:
        return False


def check_image_is_float(image: np.ndarray):
    if image.dtype == np.float32 or image.dtype == np.float64:
        return True
    else:
        return False


def check_image_is_bool(image: np.ndarray):
    if image.dtype == np.bool_:
        return True
    else:
        return False


def check_image_is_binary(image: np.ndarray):
    if image.dtype == np.bool_ and len(np.unique(image)) <= 2:
        return True
    else:
        return False


def check_image_is_label(image: np.ndarray):
    if image.dtype == np.uint16 and len(np.unique(image)) > 2:
        return True
    else:
        return False


def check_image_is_distance_transform(image: np.ndarray):
    if image.dtype == np.float32 and np.min(image) >= 0 and np.max(image) <= 1:
        return True
    else:
        return False


def check_image_is_valid(image: np.ndarray):
    if image.ndim not in [2, 3]:
        return False
    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    if image.dtype not in [
        np.uint8,
        np.uint16,
        np.float32,
        np.float64,
        np.bool_,
    ]:
        return False
    return True


def get_image_diagnostics(image: np.ndarray):
    diagnostics = {
        "shape": image.shape,
        "ndim": image.ndim,
        "dtype": image.dtype,
        "min": np.min(image),
        "max": np.max(image),
        "mean": np.mean(image),
        "std": np.std(image),
        "is_grayscale": check_image_is_grayscale(image),
        "is_rgb": check_image_is_rgb(image),
        "is_rgba": check_image_is_rgba(image),
        "is_uint8": check_image_is_uint8(image),
        "is_uint16": check_image_is_uint16(image),
        "is_float": check_image_is_float(image),
        "is_bool": check_image_is_bool(image),
        "is_binary": check_image_is_binary(image),
        "is_label": check_image_is_label(image),
        "is_distance_transform": check_image_is_distance_transform(image),
        "is_valid": check_image_is_valid(image),
    }
    return diagnostics


def print_image_diagnostics(image: np.ndarray):
    diagnostics = get_image_diagnostics(image)
    for key, value in diagnostics.items():
        logger.debug(f"{key}: {value}")


def get_dummy_image(shape=(100, 100), dtype=np.uint8):
    if dtype == np.uint8:
        return np.random.randint(0, 256, shape, dtype=np.uint8)
    elif dtype == np.uint16:
        return np.random.randint(0, 65536, shape, dtype=np.uint16)
    elif dtype == np.float32:
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == np.float64:
        return np.random.rand(*shape)
    elif dtype == np.bool_:
        return np.random.randint(0, 2, shape, dtype=np.bool_)
    else:
        raise ValueError("Unsupported dtype")


def get_dummy_label_image(shape=(100, 100), num_labels=10):
    return np.random.randint(0, num_labels, shape, dtype=np.uint16)


def get_dummy_distance_transform_image(shape=(100, 100)):
    return np.random.rand(*shape).astype(np.float32)


def get_dummy_binary_image(shape=(100, 100)):
    return np.random.randint(0, 2, shape, dtype=np.bool_)


def get_dummy_rgb_image(shape=(100, 100, 3)):
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def get_dummy_rgba_image(shape=(100, 100, 4)):
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def get_dummy_grayscale_image(shape=(100, 100), dtype=np.uint8):
    return get_dummy_image(shape, dtype)


def get_dummy_image_with_nan(shape=(100, 100)):
    image = np.random.rand(*shape).astype(np.float32)
    image[10, 10] = np.nan
    return image


def get_dummy_image_with_inf(shape=(100, 100)):
    image = np.random.rand(*shape).astype(np.float32)
    image[10, 10] = np.inf
    return image


def get_dummy_image_with_negative(shape=(100, 100)):
    image = np.random.rand(*shape).astype(np.float32)
    image[10, 10] = -1
    return image


def get_dummy_image_with_zero_std(shape=(100, 100)):
    return np.ones(shape, dtype=np.uint8)


def get_dummy_image_with_two_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[50:, :] = 1
    return image


def get_dummy_image_with_three_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[33:, :] = 1
    image[66:, :] = 2
    return image


def get_dummy_image_with_four_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[25:, :] = 1
    image[50:, :] = 2
    image[75:, :] = 3
    return image


def get_dummy_image_with_five_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[20:, :] = 1
    image[40:, :] = 2
    image[60:, :] = 3
    image[80:, :] = 4
    return image


def get_dummy_image_with_six_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[16:, :] = 1
    image[32:, :] = 2
    image[48:, :] = 3
    image[64:, :] = 4
    image[80:, :] = 5
    return image


def get_dummy_image_with_seven_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[14:, :] = 1
    image[28:, :] = 2
    image[42:, :] = 3
    image[56:, :] = 4
    image[70:, :] = 5
    image[84:, :] = 6
    return image


def get_dummy_image_with_eight_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[12:, :] = 1
    image[24:, :] = 2
    image[36:, :] = 3
    image[48:, :] = 4
    image[60:, :] = 5
    image[72:, :] = 6
    image[84:, :] = 7
    return image


def get_dummy_image_with_nine_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[11:, :] = 1
    image[22:, :] = 2
    image[33:, :] = 3
    image[44:, :] = 4
    image[55:, :] = 5
    image[66:, :] = 6
    image[77:, :] = 7
    image[88:, :] = 8
    return image


def get_dummy_image_with_ten_values(shape=(100, 100)):
    image = np.zeros(shape, dtype=np.uint8)
    image[10:, :] = 1
    image[20:, :] = 2
    image[30:, :] = 3
    image[40:, :] = 4
    image[50:, :] = 5
    image[60:, :] = 6
    image[70:, :] = 7
    image[80:, :] = 8
    image[90:, :] = 9
    return image