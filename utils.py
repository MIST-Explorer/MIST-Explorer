import os
import sys
from multiprocessing import Value
from typing import List

import cv2
import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from skimage import transform


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
    return qimage if qimage is not None else QImage()


def qimage_to_numpy(qimage: QImage):
    # Ensure the QImage format is suitable for conversion
    valid_formats = [QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16]
    ptr = qimage.bits()
    width = qimage.width()
    height = qimage.height()

    # convert to grayscale
    if qimage.format() not in valid_formats:
        if qimage.format() == QImage.Format.Format_RGB32:
            assert ptr is not None, "QImage bits() returned None"
            ptr.setsize(width * height * 4)
            arr = np.array(ptr, dtype=np.uint8).reshape(height, width, 4)

            return arr
        else:
            raise ValueError("Unsupported dtype")

    elif qimage.format() == QImage.Format.Format_Grayscale16:
        qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)

    elif qimage.format() == QImage.Format.Format_Grayscale8:
        qimage = qimage
    else:
        raise ValueError("Unsupported dtype")

    # Set buffer size based on dtype
    assert ptr is not None, "QImage bits() returned None"
    ptr.setsize(width * height)

    print(qimage.format())
    arr = np.array(ptr).reshape(height, width)

    return arr


# this and qimage to numpy seems repetitive, delete one of them
def pixmap_to_image(pixmap: QPixmap):

    if pixmap == None:
        return None
    # Convert QPixmap to QImage
    qimage = pixmap.toImage()

    # Convert QImage to numpy array
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    assert ptr is not None, "QImage bits() returned None"
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA

    # Save numpy array as an image file using OpenCV
    return arr


def is_grayscale(image: np.ndarray) -> bool:

    if len(image.shape) == 3 and image.shape[2] == 3:
        return False
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return True
    else:
        raise ValueError("Image format not recognized")


def to_uint8(image):
    """Convert image to uint8 with proper scaling"""
    # Check if image is already uint8
    if image.dtype == np.uint8:
        return image

    # Convert to float and scale to 0-255
    img_float = image.astype(np.float32)
    if img_float.max() > img_float.min():  # Check to avoid division by zero
        img_norm = (img_float - img_float.min()) * (
            255.0 / (img_float.max() - img_float.min())
        )
        return img_norm.astype(np.uint8)
    else:
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
        print(f"Error calculating NCC: {str(e)}")
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
def scale_adjust(arr: np.ndarray) -> NDArray[np.uint8]:
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0)).astype(np.uint8)
    elif arr.dtype == np.uint8:
        return arr
    elif arr.dtype == np.uint32:
        array_uint8 = ((arr / arr.max()) * 255).astype(np.uint8)
        return array_uint8
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() > 1.0:
            arr = arr / arr.max()
        array_uint8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return array_uint8
    else:
        raise ValueError("unsupported array type: ", arr.dtype)


def create_lut(new_min, new_max):
    lut = np.zeros(256, dtype=np.uint8)  # uint8 for display
    lut[new_min : new_max + 1] = np.linspace(
        start=0,
        stop=255,
        num=(new_max - new_min + 1),
        endpoint=True,
        dtype=np.uint8,
    )
    lut[:new_min] = 0  # clip between 0 and 255
    lut[new_max + 1 :] = 255

    return lut


# def to_float64(arr: np.ndarray):
# if arr.dtype == np.uint16:


def auto_contrast(img):
    return adjustContrast(scale_adjust(img))


def gaussian_kernel_1d(sigma, radius=None):
    """Generate 1D Gaussian kernel."""
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_separable(image: NDArray[np.float64], sigma=1.0):
    """Apply separable Gaussian blur manually using numpy."""
    kernel = gaussian_kernel_1d(sigma)

    # Manual separable convolution - more efficient than apply_along_axis
    # Horizontal pass
    pad_width = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (pad_width, pad_width)), mode="reflect")
    blurred_h = np.zeros_like(image)

    for i in range(image.shape[0]):
        blurred_h[i] = np.convolve(padded[i], kernel, mode="valid")

    # Vertical pass
    padded = np.pad(blurred_h, ((pad_width, pad_width), (0, 0)), mode="reflect")
    blurred = np.zeros_like(image)

    for j in range(image.shape[1]):
        blurred[:, j] = np.convolve(padded[:, j], kernel, mode="valid")

    return blurred


def downsample(image: NDArray[np.float64], scale=0.5):
    """Downsample image by given scale factor."""
    step = int(round(1 / scale))
    return image[::step, ::step]


def build_optical_flow_pyramid_pure_numpy(
    image: NDArray[np.uint16], max_level=3, scale=0.5, sigma=1.0, min_size=16
) -> List[NDArray[np.float64]]:
    """Version using only numpy with manual separable convolution."""
    assert image.ndim == 2, "Only grayscale images supported"
    float_image = image.astype(np.float64)

    pyramid = [float_image]
    current = float_image

    for level in range(max_level):
        if min(current.shape) < min_size:
            break

        blurred = gaussian_blur_separable(current, sigma=sigma)
        current = downsample(blurred, scale=scale)
        pyramid.append(current)

    return pyramid


def adjust_contrast(img: NDArray[np.float64], min_percentile=2, max_percentile=98):
    """Adjust image contrast using percentile-based clipping for float images"""
    # Calculate percentiles
    minval = np.percentile(img, min_percentile)
    maxval = np.percentile(img, max_percentile)

    # Avoid division by zero
    if maxval - minval < 1e-12:
        return np.zeros_like(img)

    # Clip and rescale to [0.0, 1.0]
    img_adjusted = np.clip(img, minval, maxval)
    img_adjusted = (img_adjusted - minval) / (maxval - minval)

    return img_adjusted  # stays float64, values in [0.0, 1.0]


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


def warp_image(img, transform_matrix):
    """Enhanced image warping with better precision and boundary handling"""
    if transform_matrix is None:
        return img

    # Ensure input is float64 for precision
    img_float = img.astype(np.float64) if img.dtype != np.float64 else img

    try:
        if transform_matrix.shape == (2, 3):  # Affine format
            # Convert 2x3 to 3x3 homogeneous matrix
            full_matrix = np.vstack([transform_matrix, [0, 0, 1]])
            tform = transform.AffineTransform(matrix=full_matrix)
        elif transform_matrix.shape == (3, 3):  # Already homogeneous
            # Check if it's affine (last row should be [0, 0, 1])
            if np.allclose(transform_matrix[2, :], [0, 0, 1]):
                tform = transform.AffineTransform(matrix=transform_matrix)
            else:
                tform = transform.ProjectiveTransform(matrix=transform_matrix)
        else:
            raise ValueError(
                f"Unsupported transform matrix shape: {transform_matrix.shape}"
            )

        # Perform warping with high precision
        warped = transform.warp(
            img_float,
            tform,
            output_shape=img_float.shape,
            preserve_range=True,
            mode="constant",
            cval=0,
            order=3,  # Bicubic interpolation for better quality
        )

        return warped

    except Exception as e:
        print(f"Warping error: {e}")
        return img_float


# Memory monitoring utility
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
