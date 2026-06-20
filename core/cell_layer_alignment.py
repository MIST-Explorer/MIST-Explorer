import concurrent.futures
import logging
import traceback
from typing import Tuple

import cv2
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal
from pystackreg import StackReg
from pystackreg.util import to_uint16
from scipy.ndimage import binary_fill_holes, rotate, zoom

from utils import (
    adjust_contrast,
    background_subtraction_with_histogram,
    make_same_shape,
    match_histograms,
    remove_padding,
    to_cv2_friendly,
    to_uint8,
    warp_image,
)

logger = logging.getLogger(__name__)


class CellLayerAligner(QThread):
    """Two Staged Cell Layer Alignment Using ITK Intensity Based Alignment & pystackreg's Algorithm"""

    DEFAULT_ROTATE_BY = 90
    DEFAULT_COARSE_SCALE = 1 / 16.0
    DEFAULT_FINE_SCALE = 0.5

    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    aligned_image_signal = pyqtSignal(dict, np.ndarray, np.ndarray)
    manual_ready = pyqtSignal(dict)

    def __init__(
        self,
        rotate_by=DEFAULT_ROTATE_BY,
        coarse_scale=DEFAULT_COARSE_SCALE,
        fine_scale=DEFAULT_FINE_SCALE,
        scale=(1.0, 1.0),
    ):
        super().__init__()
        self.target_image = np.zeros(0).astype(np.uint16)
        self.unaligned_image = np.zeros(0).astype(np.uint16)
        self.original_target_shape = None
        self.replace = False
        self.target_channel = "Channel 1"
        self.target_uuid = ""
        self.unaligned_channel = "Channel 1"
        self.unaligned_uuid = ""
        self.scale = scale
        self.need_gradient_descent = True  # Use ITK for coarse alignment
        self.stackreg_type = "affine"
        # ITK metadata to be captured
        self.itk_angle = 0
        self.itk_flip = False
        self.rotate_by = rotate_by
        # Coarse and fine alignment parameters
        self.coarse_scale = coarse_scale
        self.rotation_angles = [0]  # Default to no rotation
        self.fine_scale = fine_scale
        self.target_reg_image = None
        self.unaligned_reg_image = None

    def set_target_image(self, target_image, channel_name, uuid):
        self.target_image = target_image
        self.original_target_shape = target_image.shape
        self.target_channel = channel_name
        self.target_uuid = uuid

    def set_unaligned_image(self, unaligned_image, channel_name, uuid):
        self.unaligned_image = unaligned_image
        self.unaligned_channel = channel_name
        self.unaligned_uuid = uuid

    def set_registration_images(self, target_reg_image, unaligned_reg_image):
        """Set pre-contrasted images used only for computing the transform.

        Pass None for either argument to fall back to the original image.
        The warp output always uses the originals.
        """
        self.target_reg_image = target_reg_image
        self.unaligned_reg_image = unaligned_reg_image

    def set_scale(self, scale):
        self.scale = scale

    def set_fine_scale(self, fine_scale: float):
        self.fine_scale = fine_scale

    @classmethod
    def estimate_peak_memory_bytes(
        cls,
        target_shape,
        moving_shape,
        target_dtype,
        moving_dtype,
        scale=(1.0, 1.0),
        fine_scale=0.5,
        coarse_scale=None,
        use_gradient_descent=True,
        skip_coarse_alignment=False,
    ) -> int:
        """Estimate extra working memory for the alignment pipeline.

        This is an intentionally conservative estimate based on the dominant
        transient allocations in the fine and full-resolution stages, with a
        small coarse-stage term and a safety margin for library internals.
        """
        coarse_scale = cls.DEFAULT_COARSE_SCALE if coarse_scale is None else coarse_scale

        def _pixel_count(shape) -> int:
            return int(np.prod(shape[:2]))

        def _scaled_shape(shape, x_scale: float, y_scale: float) -> tuple[int, int]:
            height, width = shape[:2]
            return (
                max(1, int(height * y_scale)),
                max(1, int(width * x_scale)),
            )

        target_pixels = _pixel_count(target_shape)
        moving_pixels = _pixel_count(moving_shape)
        target_itemsize = np.dtype(target_dtype).itemsize
        moving_itemsize = np.dtype(moving_dtype).itemsize

        x_scale, y_scale = scale
        scaled_moving_shape = _scaled_shape(moving_shape, x_scale, y_scale)
        scaled_moving_pixels = _pixel_count(scaled_moving_shape)
        target_fine_pixels = max(1, int(target_pixels * fine_scale * fine_scale))
        moving_fine_pixels = max(
            1, int(scaled_moving_pixels * fine_scale * fine_scale)
        )

        coarse_pixels = max(
            1,
            int(target_pixels * coarse_scale * coarse_scale),
            int(scaled_moving_pixels * coarse_scale * coarse_scale),
        )

        target_full_bytes = target_pixels * target_itemsize
        moving_full_bytes = moving_pixels * moving_itemsize
        scaled_moving_bytes = scaled_moving_pixels * moving_itemsize
        coarse_stage_bytes = coarse_pixels * (8 + (4 * cls._itk_worker_cap(coarse_pixels, upper=6)))

        if skip_coarse_alignment:
            coarse_stage_bytes = min(coarse_stage_bytes, coarse_pixels * 8)

        # Full-resolution warps and retained arrays dominate the non-fine part
        # of the working set.
        full_stage_bytes = (5 * target_full_bytes) + moving_full_bytes + scaled_moving_bytes

        # The fine stage is where StackReg and optional gradient descent create
        # most transient allocations. Use coefficients derived from empirical
        # profiling so the UI estimate is conservative rather than optimistic.
        fine_bytes_per_pixel = 32 + (32 * fine_scale)
        if use_gradient_descent:
            fine_bytes_per_pixel += 32
        fine_stage_bytes = int(
            (target_fine_pixels + moving_fine_pixels) * fine_bytes_per_pixel
        )
        return int((coarse_stage_bytes + full_stage_bytes + fine_stage_bytes) * 1.1)

    def skip_coarse_alignment(self, skip: bool):
        """
        Skip the coarse alignment step, only 1 angle will be used, only use itk to try to center
        """
        if skip:
            self.rotation_angles = [0]
        else:
            self.rotation_angles = list(range(0, 360, self.rotate_by))

    def skip_gradient_descent(self, skip: bool):
        """
        Skip the gradient descent step, only use ITK for coarse alignment
        """
        self.need_gradient_descent = not skip

    def manually_align(self, matrix: np.ndarray, action: str = "add_layer") -> None:
        """Emit a manual alignment with a single composed 2x3 affine matrix."""
        self.progress.emit(100, "Manual alignment set")
        self.manual_ready.emit(
            {
                "uuid": self.unaligned_uuid,
                "layer": self.unaligned_channel,
                "matrix": matrix,
                "target_shape": self.original_target_shape,
                "action": action,
                "target_uuid": self.target_uuid,
                "target_channel": self.target_channel,
            }
        )

    def run(self):
        """Main processing function that runs in the thread"""
        if self.target_image is None or self.unaligned_image is None:
            self._fatal_error_message(
                "Both target and unaligned images must be provided"
            )
            return
        try:
            x_scale, y_scale = self.scale
            if x_scale != 1.0 or y_scale != 1.0:
                logger.info("Scaling unaligned image by (x=%.3f, y=%.3f)", x_scale, y_scale)
                h, w = self.unaligned_image.shape[:2]
                scaled_w = max(1, int(w * x_scale))
                scaled_h = max(1, int(h * y_scale))
                self.unaligned_image = cv2.resize(
                    self.unaligned_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
                )

            # Resolve registration images (contrasted or original) after scale is applied
            target_pre_contrasted = self.target_reg_image is not None
            moving_pre_contrasted = self.unaligned_reg_image is not None
            reg_target = self.target_reg_image if target_pre_contrasted else self.target_image
            if moving_pre_contrasted:
                if x_scale != 1.0 or y_scale != 1.0:
                    reg_moving = cv2.resize(
                        self.unaligned_reg_image, (scaled_w, scaled_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    reg_moving = self.unaligned_reg_image
            else:
                reg_moving = self.unaligned_image

            # Prepare images for coarse alignment
            self.progress.emit(5, "Preparing images for coarse alignment")
            self.coarse_target, self.coarse_moving = self._prepare_images(
                reg_target, reg_moving, self.coarse_scale,
                target_pre_contrasted, moving_pre_contrasted,
            )
            coarse_transform = np.eye(3)

            # Perform coarse alignment
            self.progress.emit(30, "Intensity-based coarse alignment (ITK)")
            coarse_transform = self._coarse_alignment()
            if coarse_transform is None:
                self._fatal_error_message("Coarse alignment (ITK) failed.")
                return

            logger.debug("Coarse transform:\n%s", coarse_transform)
            # Coarse-scale preprocessed images no longer needed once the
            # transform is extracted. Release before fine stage allocates.
            self.coarse_target = None
            self.coarse_moving = None
            # Prepare images for fine alignment
            self.progress.emit(50, "Preparing images for fine alignment")
            fine_transform = self._scale_transform_matrix(
                coarse_transform[:2, :3].copy(), self.coarse_scale, self.fine_scale
            )
            fine_target, fine_unaligned = self._prepare_images(
                reg_target, reg_moving, self.fine_scale,
                target_pre_contrasted, moving_pre_contrasted,
            )
            fine_target_shape = (
                np.array(self.target_image.shape) * self.fine_scale
            ).astype(int)
            logger.debug("Fine target shape: %s, fine_target: %s, fine_unaligned: %s",
                         fine_target_shape, fine_target.shape, fine_unaligned.shape)
            fine_moving = warp_image(fine_unaligned, fine_transform)
            del fine_unaligned
            fine_moving = to_uint8(fine_moving)
            fine_moving = remove_padding(fine_moving, fine_target_shape)
            fine_target_image = remove_padding(fine_target, fine_target_shape)
            del fine_target
            logger.debug("Fine target shape: %s, Fine moving shape: %s",
                         fine_target_image.shape, fine_moving.shape)

            # Perform fine alignment
            self.progress.emit(60, "Fine alignment (pystackreg)")
            refinement_transform, aligned_preview = self._alignment_stackreg(
                fine_target_image, fine_moving
            )

            if refinement_transform is None:
                logger.warning("StackReg refinement failed, using only coarse alignment.")
                aligned_preview = fine_moving
                refinement_transform = np.eye(3)  # Identity transform
                refinement_transform = self._scale_transform_matrix(
                    refinement_transform[:2, :3], 1, self.fine_scale
                )
            assert (
                refinement_transform is not None
            ), "Refinement transform should not be None"

            assert aligned_preview is not None, "Aligned preview should not be None"

            gradient_transform = None
            if self.need_gradient_descent:
                # refine alignment using gradient descent
                gradient_transform, gradient_aligned = gradient_descent_alignment(
                    aligned_preview, fine_target_image, 200)
                del gradient_aligned
            # Fine-stage buffers no longer needed — release before full-res warps.
            del fine_moving, fine_target_image, aligned_preview

            # Apply final transformation
            self.progress.emit(80, "Applying final transformation")
            full_coarse_matrix = self._scale_transform_matrix(
                coarse_transform[:2, :3], self.coarse_scale, 1
            )
            full_refinement_transform = self._scale_transform_matrix(
                refinement_transform[:2, :3], self.fine_scale, 1
            )
            # Skip the pad/unpad round-trip when shapes already match — it's a
            # full-resolution copy we don't need to make.
            if self.target_image.shape == self.unaligned_image.shape:
                padded_moving = self.unaligned_image
            else:
                _, padded_moving = make_same_shape(
                    self.target_image, self.unaligned_image)
            intermediate_aligned = warp_image(
                padded_moving, full_coarse_matrix)
            del padded_moving
            if intermediate_aligned.shape != self.target_image.shape:
                intermediate_aligned = remove_padding(
                    intermediate_aligned, self.target_image.shape
                )
            final_aligned_image = warp_image(
                intermediate_aligned, full_refinement_transform
            )
            del intermediate_aligned
            full_gradient_transform = None
            if gradient_transform is not None:
                # Apply gradient descent refinement if available
                full_gradient_transform = self._scale_transform_matrix(
                    gradient_transform[:2, :3], self.fine_scale, 1
                )
                next_aligned = warp_image(
                    final_aligned_image, full_gradient_transform
                )
                del final_aligned_image
                final_aligned_image = next_aligned

            # Sequential warps using warp_image() compose as: M_coarse @ M_refine (@ M_grad).
            # Controller receives the inverted (forward Moving→Target) map for Qt
            # compatibility, and handles inverting it back to backward map for warping.
            coarse_3x3 = np.vstack([full_coarse_matrix, [0, 0, 1]])
            refine_3x3 = np.vstack([full_refinement_transform, [0, 0, 1]])
            pre_3x3_bwd = coarse_3x3 @ refine_3x3
            if full_gradient_transform is not None:
                pre_3x3_bwd = pre_3x3_bwd @ np.vstack([full_gradient_transform, [0, 0, 1]])
            pre_matrix = np.linalg.inv(pre_3x3_bwd)[:2, :].astype(np.float64)

            # Convert result back to original dtype for storage. When dtypes
            # already match, _convert_to_original_dtype returns the same array —
            # reuse it as the signal payload so we don't hold two full-res
            # arrays at emission time.
            result_data = self._convert_to_original_dtype(
                final_aligned_image, self.unaligned_image.dtype
            )
            if result_data is not final_aligned_image:
                final_aligned_image = result_data

            result_payload = {
                "uuid": self.unaligned_uuid,
                "layer": self.unaligned_channel,
                "matrix": pre_matrix,
                "target_uuid": self.target_uuid,
                "target_channel": self.target_channel,
                "target_shape": self.original_target_shape,
            }

            self.progress.emit(100, "Two-stage alignment complete")
            self.aligned_image_signal.emit(
                result_payload, self.target_image, final_aligned_image
            )

        except Exception:
            self._fatal_error_message(
                f"Error during alignment: {traceback.format_exc()}"
            )

    def _coarse_alignment(self):
        angle, flip, params = self._itk_align(
            self.rotation_angles,
            self.coarse_target,
            self.coarse_moving,
        )
        self.itk_angle = angle
        self.itk_flip = flip

        params = np.array(params)
        transform_info = extract_complete_transformation(
            params, angle, flip, self.coarse_moving.shape
        )
        t = transform_info["combined_matrix"]
        logger.debug("Transform shape: %s", t.shape)
        if t.shape == (2, 3):
            t = np.vstack([t, [0, 0, 1]])
        inverted = np.linalg.inv(t)

        return inverted

    def _prepare_images(
        self,
        target_image,
        unaligned_image,
        coarse_scale,
        target_pre_contrasted: bool = False,
        moving_pre_contrasted: bool = False,
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Preprocess two images for coarse alignment.

        When ``*_pre_contrasted`` is True the side has already been windowed
        by the user (toolbar contrast sliders) — skip background subtraction,
        histogram matching, and adjust_contrast so we don't override their
        chosen window.
        """
        target_image = to_cv2_friendly(target_image)
        unaligned_image = to_cv2_friendly(unaligned_image)
        coarse_target = cv2.resize(
            target_image,
            (0, 0),
            fx=coarse_scale,
            fy=coarse_scale,
            interpolation=cv2.INTER_AREA,
        )
        coarse_moving = cv2.resize(
            unaligned_image,
            (0, 0),
            fx=coarse_scale,
            fy=coarse_scale,
            interpolation=cv2.INTER_AREA,
        )
        if not target_pre_contrasted:
            coarse_target = background_subtraction_with_histogram(coarse_target)[0]
        if not moving_pre_contrasted:
            coarse_moving = background_subtraction_with_histogram(coarse_moving)[0]
        coarse_target = to_uint8(coarse_target)
        coarse_moving = to_uint8(coarse_moving)
        if not moving_pre_contrasted:
            target_histogram = np.histogram(coarse_target.flatten(), bins=256)[0]
            coarse_moving = match_histograms(coarse_moving, target_histogram)
            coarse_moving = np.clip(coarse_moving, 32, 255) - 32
        # adjust_contrast internally promotes to float32 — skip the explicit
        # float64 cast so we don't pay that memory twice.
        if not target_pre_contrasted:
            coarse_target = adjust_contrast(coarse_target, 2, 99)
        if not moving_pre_contrasted:
            coarse_moving = adjust_contrast(coarse_moving, 2, 99)
        coarse_moving = cv2.normalize(
            coarse_moving, coarse_moving, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        coarse_target = cv2.normalize(
            coarse_target, coarse_target, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        coarse_moving = morph_open(coarse_moving)
        coarse_target, coarse_moving = make_same_shape(
            coarse_target, coarse_moving)
        return coarse_target, coarse_moving

    @staticmethod
    def _itk_worker_cap(per_job_bytes: int, upper: int) -> int:
        """Cap concurrent ITK workers so the combined working set stays under
        ~500 MB. Each worker holds a rotated copy plus two SITK float32 images.
        """
        budget = 512 * 1024 * 1024
        estimate = per_job_bytes * 4 if per_job_bytes > 0 else 1
        return int(np.clip(budget // estimate, 1, upper))

    def _itk_align(self, rotation_angles, fixed, moving):
        logger.info("Stage 1: Finding best angles...")
        angle_results = []
        workers = self._itk_worker_cap(fixed.nbytes, upper=6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(register_combination, fixed, moving, angle, False)
                for angle in rotation_angles
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    angle_results.append(future.result())
                except Exception as e:
                    logger.error("Error during registration: %s", e, exc_info=True)
        if len(angle_results) == 1:
            # If only one angle was tested, return it directly
            best_angle, best_flip, params = angle_results[0][1:]
            logger.info("Only one angle tested: %s, flip: %s", best_angle, best_flip)
            return best_angle, best_flip, params
        valid_angle_results = [r for r in angle_results if r[3] is not None]
        sorted_angle_results = sorted(
            valid_angle_results, key=lambda x: x[0], reverse=True
        )
        top3_angles = sorted_angle_results[:3]

        logger.info("Top 3 angles:")
        for i, (score, angle, _, params) in enumerate(top3_angles):
            logger.info("  %d. Angle %s: score = %s", i+1, angle, score)

        logger.info("Stage 2: Testing flip options for top 3 angles...")
        flip_results = []
        flip_workers = self._itk_worker_cap(fixed.nbytes, upper=3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=flip_workers) as executor:
            futures = [
                executor.submit(register_combination, fixed, moving, angle, True)
                for score, angle, _, params in top3_angles
            ]
            for future in concurrent.futures.as_completed(futures):
                flip_results.append(future.result())

        all_results = sorted_angle_results + flip_results
        valid_all_results = [r for r in all_results if r[3] is not None]
        if not valid_all_results:
            raise RuntimeError(
                "No valid alignment found after all ITK attempts.")

        final_sorted_results = sorted(
            valid_all_results, key=lambda x: x[0], reverse=True
        )
        best_score, best_angle, best_flip, params = final_sorted_results[0]
        logger.info("Final best: angle=%s, flip=%s, score=%s",
                    best_angle, best_flip, best_score)
        return best_angle, best_flip, params

    def _alignment_stackreg(self, target_fine, unaligned_fine):
        """Enhanced StackReg alignment with better preprocessing"""
        try:
            # _image_enhancement allocates fresh arrays (to_uint8, binary_fill_holes),
            # so we don't need to pre-copy the inputs.
            target_enhanced, unaligned_enhanced = self._image_enhancement(
                target_fine, unaligned_fine
            )
            target_uint16 = to_uint16(target_enhanced)
            unaligned_uint16 = to_uint16(unaligned_enhanced)
            sr = StackReg(StackReg.AFFINE)
            aligned_result = sr.register_transform(
                target_uint16, unaligned_uint16)
            refinement_matrix = sr.get_matrix()
            aligned_result = warp_image(unaligned_fine, refinement_matrix)
            return refinement_matrix[:2, :3], aligned_result
        except Exception as e:
            logger.error("StackReg refinement error: %s", e)
            return None, None

    def _image_enhancement(self, target, moving):
        target = to_uint8(target)
        moving = to_uint8(moving)
        cleaned_fixed = binary_fill_holes(target > 0.1)
        assert cleaned_fixed is not None, "Fixed image should not be None"
        cleaned_fixed = cleaned_fixed.astype(np.uint8) * 255
        cleaned_moving = binary_fill_holes(moving > 0.1)
        assert cleaned_moving is not None, "Moving image should not be None"
        cleaned_moving = cleaned_moving.astype(np.uint8) * 255
        return cleaned_fixed, cleaned_moving

    def _scale_transform_matrix(
        self, matrix, from_scale, to_scale
    ) -> np.ndarray | None:
        if matrix is None:
            return None
        scale_factor = to_scale / from_scale
        scaled_matrix = matrix.copy()
        if matrix.shape == (2, 3):
            scaled_matrix[0, 2] *= scale_factor
            scaled_matrix[1, 2] *= scale_factor
        elif matrix.shape == (3, 3):
            # For 3x3 matrices (DST = M @ SRC), scaling coordinates by S
            # (where S is the diag matrix of scale_factor) results in M_new = S @ M @ S_inv.
            # This preserves the rotation/scale part and scales translation by scale_factor.
            s = np.diag([scale_factor, scale_factor, 1.0])
            s_inv = np.diag([1 / scale_factor, 1 / scale_factor, 1.0])
            scaled_matrix = s @ matrix @ s_inv
        return scaled_matrix

    def _convert_to_original_dtype(self, img, original_dtype):
        """Cast img back to original_dtype using np.issubdtype rules.

        Returns img itself when dtype already matches — callers can rely on
        ``is``-identity to detect that no new allocation happened.
        """
        if img.dtype == original_dtype:
            return img
        if np.issubdtype(original_dtype, np.floating):
            return img.astype(original_dtype, copy=False)
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            return np.clip(img, info.min, info.max).astype(original_dtype, copy=False)
        if np.issubdtype(original_dtype, np.bool_):
            return img.astype(bool, copy=False)
        return img.astype(original_dtype, copy=False)

    def _fatal_error_message(self, msg):
        self.error.emit(msg)
        self.progress.emit(100, "Failed")


def calculate_alignment_metrics(fixed_array, aligned_array):
    # Ensure same shape
    if fixed_array.shape != aligned_array.shape:
        # Resize if needed

        zoom_factors = [
            f / a for f,
            a in zip(
                fixed_array.shape,
                aligned_array.shape)]
        aligned_array = zoom(aligned_array, zoom_factors, order=1)

    # Flatten arrays for easier computation
    fixed_flat = fixed_array.flatten()
    aligned_flat = aligned_array.flatten()

    correlation = abs(np.corrcoef(fixed_flat, aligned_flat)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    return correlation


def composite_to_matrix(composite_transform, reference_image):
    """
    Convert a CompositeTransform to a single 2x3 affine transformation matrix.

    Returns matrix in format: [[M00, M01, Tx], [M10, M11, Ty]]
    where the transformation is: x' = M00*x + M01*y + Tx
                                y' = M10*x + M11*y + Ty
    """
    try:
        # Method 1: Try to get matrix directly from individual transforms
        if composite_transform.GetNumberOfTransforms() == 1:
            single_transform = composite_transform.GetNthTransform(0)
            if hasattr(single_transform, "GetMatrix"):
                # Get 2x2 matrix and translation
                matrix_2x2 = single_transform.GetMatrix()
                translation = single_transform.GetTranslation()

                # Convert to 2x3 affine matrix
                return [
                    [matrix_2x2[0], matrix_2x2[1], translation[0]],  # [M00, M01, Tx]
                    [matrix_2x2[2], matrix_2x2[3], translation[1]],  # [M10, M11, Ty]
                ]

        # Method 2: For multiple transforms, compute equivalent matrix by sampling
        # Test points: origin, and points offset by 1 unit in each direction
        test_points = [
            [0.0, 0.0],  # Origin
            [1.0, 0.0],  # Unit X
            [0.0, 1.0],  # Unit Y
        ]

        # Transform the test points
        transformed_points = []
        for point in test_points:
            transformed = composite_transform.TransformPoint(point)
            transformed_points.append(transformed)

        # Extract affine matrix from transformed points
        p_origin, p_unit_x, p_unit_y = transformed_points

        # The transformation of (1,0) gives us the first column + translation
        # The transformation of (0,1) gives us the second column + translation
        # The transformation of (0,0) gives us the translation

        # Extract matrix elements
        m00 = p_unit_x[0] - p_origin[0]  # How (1,0) transforms in X
        m10 = p_unit_x[1] - p_origin[1]  # How (1,0) transforms in Y
        m01 = p_unit_y[0] - p_origin[0]  # How (0,1) transforms in X
        m11 = p_unit_y[1] - p_origin[1]  # How (0,1) transforms in Y
        tx = p_origin[0]  # Translation in X
        ty = p_origin[1]  # Translation in Y

        # Return as 2x3 affine matrix
        return np.array([[m00, m01, tx], [m10, m11, ty]])

    except Exception as e:
        logger.warning("Could not convert CompositeTransform to matrix: %s", e)
        # Fallback: return identity 2x3 matrix
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def register_combination(fixed, moving, angle, flip):
    logger.info("Registering with angle=%s, flip=%s", angle, flip)

    # Apply flip if specified
    t = np.flip(moving, axis=-2) if flip else moving

    # Apply rotation
    rotated = rotate(t, angle, reshape=False, order=1)
    logger.debug("Rotated shape: %s, Fixed shape: %s", rotated.shape, fixed.shape)

    # Convert to ITK images
    source_itk = sitk.GetImageFromArray(rotated)
    target_itk = sitk.GetImageFromArray(fixed)

    # Cast to float32 for registration
    source_itk = sitk.Cast(source_itk, sitk.sitkFloat32)
    target_itk = sitk.Cast(target_itk, sitk.sitkFloat32)

    # Create registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()

    # Create initial transform (Similarity2D for 2D registration)
    initial_transform = sitk.CenteredTransformInitializer(
        source_itk,
        target_itk,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    try:
        # Set up registration parameters
        registration_method.SetInitialTransform(
            initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Set optimizer
        registration_method.SetOptimizerAsPowell()

        # Set optimizer scales (important for different parameter types)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Execute registration
        final_transform = registration_method.Execute(source_itk, target_itk)

        # Get registration score (note: correlation metric should be maximized)
        score = registration_method.GetMetricValue()

        # Get transformation matrix
        if hasattr(final_transform, "GetMatrix"):
            transform_matrix = np.array(final_transform.GetMatrix())
        else:
            # For CompositeTransform, get the matrix from the last transform
            transform_matrix = composite_to_matrix(final_transform, target_itk)
        score = -1 * score  # Invert score for consistency (higher is better)
        logger.debug("Score: %s for angle=%s, flip=%s", score, angle, flip)
        return score, angle, flip, transform_matrix

    except Exception as e:
        logger.error("Failed at angle=%s, flip=%s: %s", angle, flip, e)
        return -1, angle, flip, None


def morph_open(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    # blur = img
    _, t = cv2.threshold(blur, 32, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
    output = closed.copy()
    # Then extract contours
    # contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return output


def create_preprocessing_matrix(angle, flip, image_shape):
    """
    Create transformation matrix for preprocessing steps (flip + rotation).

    Args:
        angle: Rotation angle in degrees
        flip: Whether Y-axis was flipped
        image_shape: Shape of the image (height, width) or (depth, height, width)

    Returns:
        numpy.ndarray: Preprocessing transformation matrix
    """
    is_2d = len(image_shape) == 2
    assert is_2d
    height, width = image_shape
    center_x, center_y = width / 2.0, height / 2.0

    # Step 1: Translate to origin
    translate_to_origin = np.array(
        [[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

    # Step 2: Y-flip (if applied)
    if flip:
        flip_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        flip_matrix = np.eye(3)

    # Step 3: Rotation (if applied)
    if angle != 0:
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    else:
        rotation_matrix = np.eye(3)

    # Step 4: Translate back to center
    translate_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

    # Combine: T2 * R * F * T1
    preprocessing_matrix = (
        translate_back @ rotation_matrix @ flip_matrix @ translate_to_origin
    )

    return preprocessing_matrix


def combine_transforms(itk_params, angle, flip, image_shape):
    """
    Combine ITK registration transform with preprocessing transforms.

    Args:
        itk_params: ITK elastix parameter object
        angle: Rotation angle used in preprocessing
        flip: Whether flip was applied in preprocessing
        image_shape: Shape of the original image

    Returns:
        numpy.ndarray: Combined transformation matrix
    """
    # Extract ITK transformation matrix
    itk_matrix = itk_params

    # Create preprocessing transformation matrix
    preprocessing_matrix = create_preprocessing_matrix(
        angle, flip, image_shape)

    # Make matrices compatible (both 3x3 or both 4x4)
    if itk_matrix.shape == (3, 3) and preprocessing_matrix.shape == (4, 4):
        logger.debug("Convert ITK 3x3 to 4x4")
        itk_4d = np.eye(4)
        itk_4d[:2, :2] = itk_matrix[:2, :2]
        itk_4d[:2, 3] = itk_matrix[:2, 2]
        itk_matrix = itk_4d
    elif itk_matrix.shape == (2, 3) and preprocessing_matrix.shape == (3, 3):
        logger.debug("Convert ITK 2x3 to 3x3")
        itk_3d = np.eye(3)
        itk_3d[:2, :2] = itk_matrix[:2, :2]
        itk_3d[:2, 2] = itk_matrix[:2, 2]
        itk_matrix = itk_3d
    elif itk_matrix.shape == (2, 3) and preprocessing_matrix.shape == (4, 4):
        logger.debug("Convert ITK 2x3 to 4x4")
        itk_4d = np.eye(4)
        itk_4d[:2, :2] = itk_matrix[:2, :2]
        itk_4d[:2, 3] = itk_matrix[:2, 2]
        itk_matrix = itk_4d

    # The complete transformation is: ITK_transform * preprocessing_transform
    # This applies preprocessing first, then ITK registration
    combined_matrix = itk_matrix @ preprocessing_matrix

    return combined_matrix


def extract_complete_transformation(itk_params, angle, flip, image_shape):
    """
    Extract the complete transformation matrix that reproduces ITK registration results.

    Args:
        itk_params: ITK elastix parameter object from registration
        angle: Rotation angle used in preprocessing
        flip: Whether flip was applied in preprocessing
        image_shape: Shape of the original moving image

    Returns:
        dict: Contains combined matrix, ITK matrix, preprocessing matrix, and metadata
    """
    try:
        itk_matrix = itk_params
        preprocessing_matrix = create_preprocessing_matrix(
            angle, flip, image_shape)
        combined_matrix = combine_transforms(
            itk_matrix, angle, flip, image_shape)

        return {
            "combined_matrix": combined_matrix,
            "itk_matrix": itk_matrix,
            "preprocessing_matrix": preprocessing_matrix,
            "angle": angle,
            "flip": flip,
            "image_shape": image_shape,
            "is_2d": len(image_shape) == 2,
        }

    except Exception as e:
        logger.error("Error extracting complete transformation: %s", e)
        return {}


def gradient_descent_alignment(
        moving_image,
        fixed_image,
        num_histogram_bins=300):
    """
    Perform gradient descent alignment between two images.

    Args:
        moving_image: The image to be aligned (moving image).
        fixed_image: The reference image (fixed image).

    Returns:
        The optimized transformation parameters.
    """
    if isinstance(moving_image, np.ndarray):
        moving_image = sitk.GetImageFromArray(moving_image)
    if isinstance(fixed_image, np.ndarray):
        fixed_image = sitk.GetImageFromArray(fixed_image)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=num_histogram_bins
    )
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.05,
        numberOfIterations=50,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    moving_resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        fixed_image.GetPixelID(),
    )
    final_transform_matrix = composite_to_matrix(final_transform, fixed_image)
    final_transform = np.array(final_transform_matrix)
    if final_transform.shape == (3, 3):
        # Convert to 2x3 for 2D images
        final_transform = final_transform[:2, :3]
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    return final_transform, moving_resampled
