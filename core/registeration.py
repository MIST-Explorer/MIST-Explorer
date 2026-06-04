import heapq
import logging
import math
import re
import time

import astroalign as aa
import cv2
import diplib as dip
import numpy as np
import pystackreg.util
import SimpleITK as sitk
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from core import ImageStorage
from core.alignment_utils import (
    adjust_contrast as shared_adjust_contrast,
    find_points as shared_find_points,
    find_points_robust as shared_find_points_robust,
    try_optical_flow_alignment as shared_try_optical_flow_alignment,
)
from utils import calculate_ncc, to_uint8

logger = logging.getLogger(__name__)


class Register(QThread):
    cell_image_signal = pyqtSignal(np.ndarray)
    protein_signal_arr_signal = pyqtSignal(np.ndarray)
    image_ready = pyqtSignal(bool)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    alignment_complete = pyqtSignal(dict, np.ndarray, np.ndarray)

    def __init__(self, template_size=None, max_points=5000):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 99999999999

        # initialize variables
        self._is_running = False
        self.protein_channels = None
        self.reference_channels = None
        self.protein_signal_array = None
        self.has_blue = False
        self.storage = ImageStorage()
        self.reference_channel_key = "Channel 1"
        self.moving_channel_key = "Channel 1"
        self.params = {
            "cell_layer": 1,  # 0 index
            "protein_detection_layer": 2,  # 0 index
            "max_size": 10000,
            "num_tiles": 10,
            "overlap": 250,
            "ncc_threshold": 0.85,
            "itk_mesh_size": 8,
            "itk_iterations": 30,
            "itk_timeout": 120,
        }
        self.tifs = (
            {
                "image_dict": self.reference_channels,
            },
            {
                "image_dict": self.protein_channels,
            },
        )
        self.min_circularity = 0.5
        self.template_size = template_size
        self.max_points = max_points

    def _fatal_error_message(self, msg):
        logger.error("Registration error: %s", msg)
        self.error.emit(msg)
        self.progress.emit(100, "Retry Maybe")

    def _handle_cancel(self):
        if not self._is_running:
            self._fatal_error_message("Cancelled registration")
            return True
        else:
            return False

    def run_registration(self):
        self._is_running = True
        logger.info("Running registration on device: CPU")
        if self.isRunning() and self._is_running:
            self.error.emit("Registration is already running")
            return
        elif self.isRunning():
            self.error.emit("Registration is cancelling")
            return
        self.start()
        self.finished.connect(lambda: setattr(self, "_is_running", False))

    def run(self):
        self.tifs = (
            {
                "image_dict": self.storage.get_alignment_ref_image(),
            },
            {
                "image_dict": self.storage.get_alignment_moving_image(),
            },
        )

        m = self.params[
            "max_size"
        ]  # this is this maximum size allowed for the registration
        self.overlap = self.params["overlap"]  # overlap between each tile
        self.num_tiles = self.params["num_tiles"]  # how many tiles we want
        self.ncc_threshold = self.params["ncc_threshold"]
        reference_image_file = self.tifs[0]
        self.progress.emit(0, "preparing alignment")  # update progress bar
        channel_wrappers = reference_image_file["image_dict"]
        assert channel_wrappers is not None, "channel wrappers are empty"
        alignment_layer = channel_wrappers[self.reference_channel_key].data

        alignment_layer = self.adjust_contrast(alignment_layer, 50, 99)
        # resize to maximum allowed size
        alignment_layer = alignment_layer[0:m, 0:m]

        fixed_map = TileMap(
            "fixed",
            alignment_layer,
            self.overlap,
            self.num_tiles)
        moving_map = None
        # generate tiles
        for tif_n, tif in enumerate(self.tifs):
            # skip reference (tif 0 is always the fixed/reference image)
            if self._handle_cancel():
                return
            if tif_n == 0:
                self.tifs[tif_n]["outputs"] = None
                continue
            alignable_brightfield = self.tifs[tif_n]["image_dict"][
                self.moving_channel_key
            ].data

            alignable_brightfield = alignable_brightfield[0:m, 0:m]
            alignable_brightfield = self.adjust_contrast(
                alignable_brightfield, 50, 99)

            moving_map = TileMap(
                "moving", alignable_brightfield, self.overlap, self.num_tiles
            )

            inputs = []
            radius = int(fixed_map.tile_size)
            for mov_data, fix_data in list(zip(moving_map, fixed_map)):
                (moving_img, moving_bounds) = mov_data
                (fixed_img, _) = fix_data

                x, y = moving_bounds["center"]
                ymin = moving_bounds["ymin"]
                xmin = moving_bounds["xmin"]

                radius = int(fixed_map.tile_size)
                inputs.append(
                    (fixed_img, moving_img, ymin, xmin, radius, x, y))

            # Select the inputs number
            outputs = []

            for tile_n, tile_set in enumerate(inputs):
                # update progress bar
                if self._handle_cancel():
                    return
                progress_update = int(((tile_n + 1) / len(inputs)) * 100)

                if tif_n == 0:
                    outputs.append(self.on_skip(tile_set))
                    continue

                result = self.align_two_img_robust(*tile_set)  # align
                transforms, ymin, xmin, radius, x, y, best_ncc = result
                self.progress.emit(
                    progress_update,
                    str(
                        f"aligning tile {tile_n + 1}/{len(inputs)} - NCC: {best_ncc:.2f}"
                    ),
                )
                if result is None:
                    continue
                outputs.append(result)

            logger.info("Done aligning")

            self.tifs[tif_n]["outputs"] = outputs

        #########################################################
        # move the other layers
        aligned_protein_signal = None
        total_sr_none = 0
        total_aa_none = 0
        total = 0
        assert isinstance(
            moving_map, TileMap), "moving_map is not a TileMap instance"
        for i, tif in enumerate(self.tifs):
            if self._handle_cancel():
                return
            if i == 0:
                continue

            file = tif["image_dict"]
            logger.debug("File keys: %s", file.keys())
            n_frames = len(file)  # 4
            logger.debug("n frames: %d", n_frames)
            new_registered_tif = []

            for layer_number in range(n_frames):
                logger.debug("Layer Number: %d for tif %d", layer_number, i)
                progress_update = int(((layer_number + 1) / n_frames) * 100)
                self.progress.emit(
                    progress_update,
                    f"Layer Number: {layer_number + 1} for tif {i + 1}")

                # channels are index 1
                bf = file[f"Channel {layer_number + 1}"].data

                bf = bf[:m, :m]

                dest = Image.fromarray(
                    np.zeros((m, m), dtype="float")
                )  # need to determine the final bit size
                prev_transf = None
                prev_itk_transf = None
                for result in tif["outputs"]:
                    transforms, ymin, xmin, radius, x, y, best_ncc = result
                    if best_ncc < 0.3:  # use previous
                        transforms[0] = prev_transf
                        transforms[1] = prev_itk_transf
                    else:
                        prev_transf = transforms[0]
                        prev_itk_transf = transforms[1]
                    source = moving_map.get_tile_by_center(bf, x, y)
                    target = moving_map.get_tile_by_center(
                        alignment_layer, x, y
                    ).astype(float)
                    total += 1
                    transf = transforms[0]
                    itk_transf = transforms[1]

                    _cv2_supported = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
                    if source.dtype not in _cv2_supported:
                        source = source.astype(np.float32)
                    if transf is not None:
                        if isinstance(transf, np.ndarray):
                            registered = cv2.warpAffine(
                                source, transf, (target.shape[1], target.shape[0]))
                        else:
                            registered, _ = aa.apply_transform(
                                transf, source, target)
                    else:
                        registered = source
                    # Apply additional ITK transformation if available
                    if itk_transf is not None:
                        registered = sitk.GetArrayFromImage(
                            sitk.Resample(
                                sitk.GetImageFromArray(registered),
                                sitk.GetImageFromArray(target),
                                itk_transf,
                                sitk.sitkLinear,
                                0.0,
                                sitk.sitkUInt16,
                            )
                        )

                    corresponding_tile = registered[
                        ymin: ymin + radius * 2, xmin: xmin + radius * 2
                    ]

                    dest.paste(Image.fromarray(pystackreg.util.to_uint16(
                        corresponding_tile)), (int(x - radius), int(y - radius)), )

                dest_arr = np.array(dest)
                new_registered_tif.append(dest_arr)

            new_registered_tif = [x.astype("uint16")
                                  for x in new_registered_tif]
            new_registered_tif = np.stack(new_registered_tif)

            aligned_protein_signal = new_registered_tif

            # alignment done
        assert aligned_protein_signal is not None, "aligned_protein_signal is None"
        self.protein_signal_array = aligned_protein_signal[
            self.params["protein_detection_layer"], :m, :m
        ]  # -> use to generate cell intensity table
        cell_image = aligned_protein_signal[
            self.params["cell_layer"], :m, :m
        ]  # -> stardist

        data = {}
        for i in range(len(aligned_protein_signal)):
            layer = aligned_protein_signal[i, :m, :m].copy()
            data[f"Channel {i + 1}"] = layer
        result = {}
        result["data"] = data
        layers = list(data.keys())
        layers = sorted(layers)
        result["layer"] = layers
        moving_uuid = self.storage.get_data("alignment_moving_uuid")
        assert moving_uuid is not None, "No alignment moving UUID found"
        moving_uuid = moving_uuid["value"]
        result["uuid"] = moving_uuid
        if self._handle_cancel():
            return
        moving_channel_idx = int(self.moving_channel_key.replace("Channel ", "")) - 1
        self.alignment_complete.emit(
            result,
            aligned_protein_signal[moving_channel_idx][:m, :m],
            alignment_layer[:m, :m],
        )
        self.protein_signal_arr_signal.emit(
            self.protein_signal_array
        )  # emit protein signal for cell intensity
        self.cell_image_signal.emit(cell_image)  # emit cell image for stardist
        self.progress.emit(100, "Alignment Done")

    def find_points(self, image, min_circularity=0.5, top_k=500):
        return shared_find_points(image, min_circularity=min_circularity, top_k=top_k)

    def find_points_robust(self, img, top_k=500):
        return shared_find_points_robust(img, top_k=top_k, min_circularity=self.min_circularity)

    def align_two_img_robust(
        self,
        fixed_img,
        moving_img,
        ymin,
        xmin,
        radius,
        x,
        y,
        tile_id=None,
        try_astro_align=False,
    ):
        source = moving_img.copy()
        target = fixed_img.copy()
        source = self.adjust_contrast(source, 50, 99)
        target = self.adjust_contrast(target, 50, 99)
        # Strategy 1: Optical flow with multiple scales
        moving_points = self.find_points_robust(source, top_k=self.max_points)
        fixed_points = self.find_points_robust(target, top_k=self.max_points)
        # ncc before
        ncc_before = calculate_ncc(source, target)
        # Try optical flow first
        best_ncc = 0
        transf = None
        name = ""
        ncc_after = -1
        best_registered = source
        itk_transf = None
        for key in moving_points:
            if key not in fixed_points:
                continue
            if len(moving_points[key]) < 4 or len(fixed_points[key]) < 4:
                continue
            moving_points[key] = moving_points[key][: self.max_points]
            fixed_points[key] = fixed_points[key][: self.max_points]
            M, success = self.try_optical_flow_alignment(
                source, target, moving_points[key], fixed_points[key]
            )
            if success:
                assert M is not None, "Optical flow transformation matrix is None"
                registered = cv2.warpAffine(
                    source, M, (target.shape[1], target.shape[0])
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after alignment is None"
                if ncc_after > best_ncc:
                    name = key
                    best_registered = registered
                    best_ncc = ncc_after
                    transf = M
                logger.debug(
                    "Optical flow alignment successful with %s points, NCC before: %.4f, after: %.4f",
                    key, ncc_before, ncc_after,
                )
        if try_astro_align:
            try:
                t, _ = aa.find_transform(
                    moving_points["CUSTOM"],
                    fixed_points["CUSTOM"],
                    detection_sigma=3,
                    min_area=5,
                    max_control_points=50,
                )
                registered, _ = aa.apply_transform(t, source, target)
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after astro alignment is None"

                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                    transf = t
                    best_registered = registered
                    name = "AstroAlign"
            except Exception as e:
                logger.warning("AstroAlign failed: %s", e)

        if (
            best_ncc < 0.5
        ):  # Only try feature matching if optical flow didn't yield good results
            M, success = self.try_feature_matching_alignment(source, target)
            if M is not None and success:
                registered = cv2.warpAffine(
                    source, M, (target.shape[1], target.shape[0])
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after feature matching is None"
                logger.debug(
                    "Feature matching alignment successful, NCC before: %.4f, after: %.4f",
                    ncc_before, ncc_after,
                )
                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                    transf = M
                    name = "SIFT"
                    best_registered = registered
        if best_ncc < self.ncc_threshold:  # If still not good, improve with ITK
            itk_transf, s, t = self.try_itk_alignment(best_registered, target)
            if itk_transf is not None:
                registered = sitk.GetArrayFromImage(
                    sitk.Resample(
                        s, t, itk_transf, sitk.sitkLinear, 0.0, s.GetPixelID()
                    )
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after ITK alignment is None"
                logger.debug(
                    "ITK alignment successful, NCC before: %.4f, after: %.4f",
                    best_ncc, ncc_after,
                )
                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                else:
                    itk_transf = None
        return [transf, itk_transf], ymin, xmin, radius, x, y, best_ncc

    def try_optical_flow_alignment(
            self,
            source,
            target,
            moving_points,
            fixed_points):
        return shared_try_optical_flow_alignment(source, target, moving_points, fixed_points)

    def try_feature_matching_alignment(self, source, target):
        """Use SIFT/ORB feature matching instead of optical flow"""
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(source, None)
        kp2, des2 = sift.detectAndCompute(target, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, False

        # Ensure we do not exceed OpenCV's hard limit of 262,144 descriptors for FlannBasedMatcher/BFMatcher train dataset
        MAX_DESCRIPTORS = 260000
        if des1.shape[0] > MAX_DESCRIPTORS:
            des1 = des1[:MAX_DESCRIPTORS]
            kp1 = kp1[:MAX_DESCRIPTORS]
        if des2.shape[0] > MAX_DESCRIPTORS:
            des2 = des2[:MAX_DESCRIPTORS]
            kp2 = kp2[:MAX_DESCRIPTORS]

        # Match features
        matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 50})
        matches = matcher.knnMatch(des1, des2, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return None, False

        # Extract matched points
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate transformation with RANSAC
        M, mask = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=100000,
            confidence=0.99,
        )

        return M, M is not None and np.sum(mask) >= 4

    def try_itk_alignment(self, source, target):
        """Use ITK for image registration"""
        try:
            start = time.time()
            timeout = self.params.get("itk_timeout", 120)

            # Convert images to SimpleITK format
            source_itk = sitk.GetImageFromArray(source)
            target_itk = sitk.GetImageFromArray(target)

            # Set pixel type to float
            source_itk = sitk.Cast(source_itk, sitk.sitkFloat32)
            target_itk = sitk.Cast(target_itk, sitk.sitkFloat32)

            # Initialize registration method
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsCorrelation()

            mesh_size = self.params.get("itk_mesh_size", 8)
            transform_domain_mesh_size = [mesh_size, mesh_size]
            bspline_transform = sitk.BSplineTransformInitializer(
                target_itk, transform_domain_mesh_size
            )

            registration_method.SetInitialTransform(
                bspline_transform, inPlace=False)

            iterations = self.params.get("itk_iterations", 30)
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-4,
                numberOfIterations=iterations,
                maximumNumberOfCorrections=5,
                costFunctionConvergenceFactor=1e6,
                maximumNumberOfFunctionEvaluations=200,
            )
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Abort callback for timeout protection
            def check_timeout(reg_method):
                if time.time() - start > timeout:
                    logger.warning("ITK alignment timed out after %ds", timeout)
                    reg_method.StopRegistration()

            registration_method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: check_timeout(registration_method),
            )

            final_transform = registration_method.Execute(target_itk, source_itk)

            return final_transform, source_itk, target_itk
        except Exception as e:
            logger.warning("ITK alignment failed: %s", e)
            return None, None, None

    def set_reference_channel_key(self, channel_key: str) -> None:
        self.reference_channel_key = channel_key

    def set_moving_channel_key(self, channel_key: str) -> None:
        self.moving_channel_key = channel_key

    def set_cell_layer(self, channel):
        match = re.search(r"\d+", channel)
        if match:
            number = int(match.group())
            result = number - 1  # 0 index
            self.params["cell_layer"] = result

    def set_protein_detection_layer(self, channel):
        match = re.search(r"\d+", channel)
        if match:
            number = int(match.group())
            result = number - 1  # 0 index
            self.params["protein_detection_layer"] = result

    def set_max_size(self, value):
        self.params["max_size"] = value

    def set_num_tiles(self, value):
        self.params["num_tiles"] = value

    def set_overlap(self, value):
        self.params["overlap"] = value

    def set_ncc_threshold(self, value):
        self.params["ncc_threshold"] = value

    def on_skip(self, param):
        _, _, ymin, xmin, radius, x, y = param
        return [None, None], ymin, xmin, radius, x, y, 0

    def adjust_contrast(self, img, contrast_min=2, contrast_max=98):
        return shared_adjust_contrast(img, contrast_min=contrast_min, contrast_max=contrast_max)

    def equalize_shape(self, cy1_rescale, cy2_rescale):
        [cy1x, cy1y] = cy1_rescale.shape
        [cy2x, cy2y] = cy2_rescale.shape

        def relu(x):
            return x if x > 0 else 0

        pos = relu

        cy2_rescale = np.pad(cy2_rescale,
                             ((int(math.floor(pos(cy1x - cy2x) / 2)),
                               int(math.ceil(pos(cy1x - cy2x) / 2)),
                               ),
                                 (math.floor((pos(cy1y - cy2y) / 2)),
                                  math.ceil((pos(cy1y - cy2y) / 2))),
                              ),
                             "empty",
                             )
        # Sometimes "edge" might work better

        cy2_rescale = cy2_rescale[0:cy1x, 0:cy1y]

        return cy1_rescale, cy2_rescale

    def update_moving_image(self, channels) -> None:
        self.protein_channels = channels
        self.tifs[1]["image_dict"] = channels
        if self.reference_channels is not None:
            self.image_ready.emit(True)
            logger.info("Moving/protein signal image updated")

    def update_reference_channels(self, reference_channels) -> None:
        self.reference_channels = reference_channels
        self.tifs[0]["image_dict"] = reference_channels
        if self.protein_channels is not None:
            self.image_ready.emit(True)
            logger.info("Reference image updated")

    def cancel(self):
        # self.exit?
        # self.quit?
        self._is_running = False


############################
class TileMap:
    def __init__(
            self,
            name: str,
            image: np.ndarray,
            overlap: int,
            height_width: int):
        """
        :param name:
        :param image:
        :param overlap: pixel amount of overlap
        :param height_width:
        """

        self.name = name
        self.image = image

        self.height_width = height_width

        self.tile_center_points = self.blockify(
            height_width) * self.image.shape[0]

        self.tile_size = self.tile_center_points[0][0][0]

        self.overlap = overlap

    @staticmethod
    def find_mask(moving_array):
        def blur(img):
            img = img.copy()
            kernel = np.ones((5, 5), np.float64) / 225
            dst = cv2.filter2D(img, -1, kernel)
            return dst

        def threshold(im, percentile):
            p = np.percentile(im, percentile)
            im = im.copy()
            im[im < p] = 0
            im[im >= p] = 255
            return im

        small = cv2.resize(
            moving_array,
            tuple((np.array(moving_array.shape) / 10).astype(int)),
            interpolation=cv2.INTER_LINEAR,
        )

        im = np.invert(threshold(blur(small), 20))

        out = dip.AreaOpening(
            im,
            filterSize=150,
            connectivity=2)  # type: ignore
        out = np.array(out)

        big = cv2.resize(
            out,
            tuple((np.array(moving_array.shape)).astype(int)),
            interpolation=cv2.INTER_LINEAR,
        )
        big[moving_array == 0] = 255

        return np.invert((big / 255).astype(bool))

    def get_tile_by_center(self, image, x, y):
        y = round(y)
        x = round(x)
        tile_size = round(self.tile_size) + self.overlap

        return image[self.keep_in_bounds(y -
                                         tile_size): self.keep_in_bounds(y +
                                                                         tile_size), self.keep_in_bounds(x -
                                                                                                         tile_size): self.keep_in_bounds(x +
                                                                                                                                         tile_size), ]

    def get_bounds_of_tile(self, x, y):
        tile_size = round(self.tile_size) + self.overlap
        ymin = (
            self.overlap
            if self.keep_in_bounds(int(y - tile_size)) == int(y - tile_size)
            else 0
        )
        ymax = (
            self.overlap
            if self.keep_in_bounds(int(y + tile_size)) == int(y + tile_size)
            else 0
        )
        xmin = (
            self.overlap
            if self.keep_in_bounds(int(x - tile_size)) == int(x - tile_size)
            else 0
        )
        xmax = (
            self.overlap
            if self.keep_in_bounds(int(x + tile_size)) == int(x + tile_size)
            else 0
        )

        return {
            "center": (x, y),
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
        }

    def __iter__(self):
        for i in self.tile_center_points:
            for j in i:
                tile = self.get_tile_by_center(self.image, j[0], j[1])
                bounds = self.get_bounds_of_tile(j[0], j[1])

                yield (tile, bounds)

    def keep_in_bounds(self, num):
        if num < 0:
            return 0
        if num > self.image.shape[0]:
            return self.image.shape[0]

        return int(num)

    @staticmethod
    def blockify(cuts):
        centerpoints = []
        for i in range(cuts):
            row = []
            for j in range(cuts):
                row.append(
                    np.array([(2 * i + 1) / (cuts * 2), (2 * j + 1) / (cuts * 2)])
                )

            centerpoints.append(np.array(row))

        return np.array(centerpoints)
