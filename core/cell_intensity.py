import itertools
import logging

import cv2 as cv
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from scipy.spatial import KDTree
from skimage.measure import regionprops

from core import ImageWrapper
from core.canvas import ImageStorage
from core.project_naming import (
    default_project_prefixed_filename,
    is_segmentation_channel,
)

logger = logging.getLogger(__name__)


def build_channel_cell_dataframe(
    median_values_for_cell_data_dict, cell_centroids, protein_headers
):
    """Build one per-channel cell table keyed by CellID."""
    rows = []
    ordered_cell_ids = sorted(median_values_for_cell_data_dict.keys())
    for cell_id in ordered_cell_ids:
        centroid = cell_centroids.get(cell_id)
        if centroid is None:
            continue
        cx, cy = centroid
        rows.append(
            [cell_id, int(cx), int(cy), *median_values_for_cell_data_dict[cell_id]]
        )

    columns = ["CellID", "Global X", "Global Y", *protein_headers]
    df = pd.DataFrame(rows, columns=columns)
    if "CellID" in df.columns:
        df["CellID"] = df["CellID"].astype(np.int64, copy=False)
    return df


def merge_channel_cell_data(
    existing_df, curr_cell_data, first_channel_num, channel_num
):
    """Merge per-channel cell tables by CellID while preserving first-channel coords."""
    curr_payload = curr_cell_data.drop(
        columns=["Global X", "Global Y"], errors="ignore"
    )
    return existing_df.merge(
        curr_payload,
        on=["CellID"],
        suffixes=(f"_cy{first_channel_num}", f"_cy{channel_num}"),
        validate="one_to_one",
    )


class CellIntensity(QThread):
    error_signal = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    filtered_stats_ready = pyqtSignal(float)
    protein_distribution_ready = pyqtSignal(object)
    channel_done = pyqtSignal(int, int)  # (channels_done, channels_total)

    def __init__(self):
        super().__init__()
        self.params = {
            "max_size": 23000,
            "num_decoding_cycles": 3,
            "num_decoding_colors": 3,
            "radius_fg": 2,
            "radius_bg": 6,
            "bead_per_protein_threshold": 0.0,
            "crosstalk_ratio": 10.0,
        }

        self.channel_to_color_code = {}
        self.segmentation_labels = np.array([], dtype=np.int32)
        self.df_cell_data = None
        self.storage = ImageStorage()
        self.protein_signal_array = None
        self.source_uuid = None
        self.project_name = None
        self.is_temp_project = False
        self._cancel_requested = False
        self._filter_num_proteins = 0
        self.bead_data = None

    def load_protein_signal_array_from_storage(self, uuid, channel):
        if uuid is None:
            uuid = self.source_uuid
            if uuid is None:
                uuid = self.storage.get_data("canvas_uuid")
                if uuid is None:
                    raise ValueError("Protein Image not found in storage.")
                uuid = uuid["value"]
        c = "Channel " + str(channel + 1)
        item = self.storage.get_data(uuid)
        assert item is not None, "item not found in storage"
        data = item.get("data", None)
        assert data is not None, "data not found in storage item"
        wrapper = data[c]
        if is_segmentation_channel(wrapper):
            raise ValueError(
                f"{c} is a virtual StarDist channel and cannot be used for generation."
            )
        self.load_protein_signal_array(wrapper.data)
        logger.info("loaded protein signal array from storage")

    def load_segmentation_labels_from_storage(self, uuid, channel):
        item = self.storage.get_data(uuid)
        assert item is not None, "item not found in storage"
        data = item.get("data", None)
        assert data is not None, "data not found in storage item"
        c = "Channel " + str(channel + 1)
        segmentation_labels = data[c]
        self.load_segmentation_labels(segmentation_labels)
        logger.info("loaded stardist labels from storage")

    def load_protein_signal_array(self, arr):
        logger.info("loaded protein signal array")
        self.protein_signal_array = arr
        self.blur_and_set_protein_layer()

    def generate_cell_intensity_table(self):
        self.progress.emit(0, "Starting Cell Intensity...")
        self._cancel_requested = False
        if self.isRunning():
            self.critical_error("Cell Intensity Calculation is already running")
            return
        self.start()

    def critical_error(self, msg):
        self.error_signal.emit(msg)
        self.progress.emit(100, "Error encountered, see message")
        return

    def compute_all_centroids(self):
        """
        Compute centroids for all unique labels in the mask (excluding 0).
        Returns a dict: {label: (cx, cy)}
        """
        if self._is_cancel_requested():
            return None

        centroids = {}
        regions = regionprops(self.segmentation_labels)
        total_regions = len(regions)
        if total_regions == 0:
            return centroids

        p = 0
        for idx, region in enumerate(regions, start=1):
            if self._is_cancel_requested():
                return None
            progress = int((idx / total_regions) * 100)
            if progress > p:
                p = progress
                self.progress.emit(
                    progress,
                    f"Finding centroid for cell {idx}/{total_regions}",
                )
            cy, cx = region.centroid
            centroids[int(region.label)] = (int(cx), int(cy))
        return centroids

    def _is_cancel_requested(self):
        if self._cancel_requested:
            self.progress.emit(100, "Cancelled")
            return True
        return False

    def infer_params(self):
        # Infer the number of decoding colors and cycles from the bead data and color code if not explicitly set
        # based on any of the color code files
        self.params["num_decoding_cycles"] = (
            self.color_code.columns.size - 1
        )  # minus 1 for protein name column

        color_code_np = self.color_code.iloc[:, 1:].to_numpy()
        max_color_value = np.max(color_code_np)
        self.params["num_decoding_colors"] = (
            max_color_value + 1
        )  # assuming colors are 0-indexed

    def run(self):
        # check all required properties
        if (
            self.segmentation_labels is None
            or self.segmentation_labels.size == 0
            or self.bead_data is None
            or not self.channel_to_color_code
        ):
            err_msg = "Missing: "
            if self.segmentation_labels is None or self.segmentation_labels.size == 0:
                err_msg += "stardist labels, "
            if self.bead_data is None:
                err_msg += "bead data, "
            if not self.channel_to_color_code:
                err_msg += "color code, "
            err_msg = err_msg.rstrip(", ")
            logger.error(err_msg)
            self.critical_error(err_msg)
            return

        self.progress.emit(5, "Preprocessing channels...")
        try:
            # 1. Load and background subtract all channel images
            images_dict = self._get_background_subtracted_images()
        except Exception as exc:
            logger.error("Failed to load and preprocess channel images: %s", exc)
            self.critical_error(str(exc))
            return

        # 2. Pre-calculate cell centroids (runs exactly ONCE)
        self.progress.emit(10, "Computing cell centroids...")
        cell_centroids = self.compute_all_centroids()
        if cell_centroids is None:
            # Cancelled or error
            return

        # 3. Calculate crosstalk suppression
        self.progress.emit(15, "Evaluating crosstalk suppression...")
        suppressed_count, suppressed_mask = self.calculate_crosstalk_suppression(
            self.params["crosstalk_ratio"]
        )
        logger.info("Crosstalk suppression complete. Suppressed %d bead-channel signals", suppressed_count)

        first_channel_num = None
        channels_total = len(self.channel_to_color_code)
        channels_done = 0
        self.df_cell_data = None

        # Build list of unique cell IDs
        unique_cell_ids = np.unique(self.segmentation_labels)
        unique_cell_ids = unique_cell_ids[unique_cell_ids > 0]
        if unique_cell_ids.size == 0:
            self.critical_error("No segmented cells found in StarDist labels.")
            return

        # Setup coordinate boundaries and mapping (same for all channels)
        radius_bg = self.params["radius_bg"]
        max_size = self.params["max_size"]
        bead_xs_all = self.bead_data[:, 0].astype(int)
        bead_ys_all = self.bead_data[:, 1].astype(int)

        # Map every bead to a cell ID
        cell_ids_for_beads = self.segmentation_labels[bead_ys_all, bead_xs_all]

        # Shared masks for cell containment & boundaries
        in_cell_mask = cell_ids_for_beads > 0
        in_bounds_mask = (
            (bead_xs_all > radius_bg)
            & (bead_ys_all > radius_bg)
            & (bead_xs_all < (max_size - radius_bg))
            & (bead_ys_all < (max_size - radius_bg))
        )
        valid_bead_mask = in_cell_mask & in_bounds_mask

        # Loop through each channel
        for c_idx, (channel, color_code) in enumerate(self.channel_to_color_code.items()):
            if self._is_cancel_requested():
                return

            self.color_code = color_code
            channel_num = int(channel.split(" ")[-1])
            logger.info(f"Generating channel {channel_num} (index {c_idx})")

            # Setup self.protein_signal_array for the intensity correction method
            self.protein_signal_array = images_dict[channel]

            self.infer_params()
            logger.debug("Inferred params: %s", self.params)

            # Generate color permutations to map code combinations to indices
            possible_values = list(range(self.params["num_decoding_colors"]))
            all_perms = [
                "".join(map(str, p))
                for p in itertools.product(
                    possible_values, repeat=self.params["num_decoding_cycles"]
                )
            ]
            color_code_to_index = {int(k): i for i, k in enumerate(all_perms)}
            index_to_color_code = {v: k for k, v in color_code_to_index.items()}

            num_proteins = len(color_code_to_index)
            cell_data_dict = {
                cell_id: [[] for _ in range(num_proteins)]
                for cell_id in unique_cell_ids.tolist()
            }

            # Map the bead colors to code integers
            cycle_cols = self.bead_data[:, 2 : 2 + self.params["num_decoding_cycles"]]
            data_modified = np.zeros((len(self.bead_data), 3))
            data_modified[:, 0:2] = self.bead_data[:, 0:2].astype("uint16")
            data_modified[:, 2] = np.array(
                [int("".join(map(str, map(int, bead)))) for bead in cycle_cols]
            )

            # Filter beads by channel-specific non-suppressed mask
            non_suppressed_mask = ~suppressed_mask[:, c_idx]
            channel_valid_mask = valid_bead_mask & non_suppressed_mask

            valid_beads = data_modified[channel_valid_mask]
            valid_cell_ids = cell_ids_for_beads[channel_valid_mask]

            self.progress.emit(
                20 + int((c_idx / channels_total) * 60),
                f"Processing {len(valid_beads)} beads for {channel}...",
            )

            # Process intensity for all non-suppressed beads inside cells
            for i, bead in enumerate(valid_beads):
                if i % 1000 == 0:
                    self.progress.emit(
                        20 + int((c_idx / channels_total) * 60) + int((i / max(len(valid_beads), 1)) * 5),
                        f"Intensity for {channel}: {i + 1}/{len(valid_beads)} beads",
                    )
                bead_x, bead_y, color_code = int(bead[0]), int(bead[1]), bead[2]
                cell_associated_id = int(valid_cell_ids[i])

                adjusted_median_intensity = self.get_adjusted_median_intensity(
                    bead_x, bead_y
                )
                protein_idx = color_code_to_index.get(color_code)
                if (
                    protein_idx is not None
                    and adjusted_median_intensity is not None
                ):
                    cell_data_dict[cell_associated_id][protein_idx].append(
                        adjusted_median_intensity
                    )

            # Build KD-tree map using only non-suppressed beads
            protein_kdtree_map = {}
            for i in range(num_proteins):
                protein_code = index_to_color_code.get(i)
                if protein_code is not None:
                    # Filter protein beads: must match code and be non-suppressed
                    protein_beads = data_modified[
                        (data_modified[:, 2] == protein_code) & non_suppressed_mask
                    ][:, 0:2].astype(int)
                    if len(protein_beads) > 0:
                        protein_kdtree_map[i] = KDTree(protein_beads)

            # Impute values for cells with incomplete protein profiles
            logger.info(f"Imputing incomplete profiles for {channel}")
            for i, cell_id in enumerate(cell_data_dict.keys()):
                cell_center = cell_centroids[cell_id]
                for protein_idx, intensities in enumerate(cell_data_dict[cell_id]):
                    if not intensities:
                        kdtree = protein_kdtree_map.get(protein_idx)
                        if kdtree:
                            _, index = kdtree.query(cell_center)
                            nn_x, nn_y = kdtree.data[index]
                            if (
                                nn_x > radius_bg
                                and nn_y > radius_bg
                                and nn_x < (max_size - radius_bg)
                                and nn_y < (max_size - radius_bg)
                            ):
                                adjusted_intensity = (
                                    self.get_adjusted_median_intensity(
                                        int(nn_x), int(nn_y)
                                    )
                                )
                                if adjusted_intensity is not None:
                                    cell_data_dict[cell_id][protein_idx].append(
                                        adjusted_intensity
                                    )

            # Compute medians for each cell
            median_values_for_cell_data_dict = {}
            for cell_id in cell_data_dict:
                array_of_subarrays = cell_data_dict[cell_id]
                array_of_subarrays_medians = [
                    np.median(subarr) if subarr else np.nan for subarr in array_of_subarrays
                ]
                median_values_for_cell_data_dict[cell_id] = array_of_subarrays_medians

            # Drop empty columns from color_code to translate names
            try:
                self.color_code = self.color_code.dropna(how="all", axis=1).dropna(
                    how="all", axis=0
                )
            except Exception:
                self.color_code = pd.DataFrame(self.color_code)
                self.color_code = self.color_code.dropna(how="all", axis=1).dropna(
                    how="all", axis=0
                )
            color_code_arr = self.color_code.to_numpy()

            color_code_translation_dict = {}
            for row in color_code_arr:
                try:
                    protein_name = row[0]
                    code = int("".join([str(int(x)) for x in row[1:]]))
                    color_code_translation_dict[code] = protein_name
                except ValueError:
                    pass

            protein_headers = []
            for subarray_index in index_to_color_code:
                corresponding_protein_code = index_to_color_code[subarray_index]
                if corresponding_protein_code in color_code_translation_dict:
                    readable_protein_name = color_code_translation_dict[corresponding_protein_code]
                    protein_headers.append(readable_protein_name)
                else:
                    protein_headers.append("N/A")

            curr_cell_data = build_channel_cell_dataframe(
                median_values_for_cell_data_dict=median_values_for_cell_data_dict,
                cell_centroids=cell_centroids,
                protein_headers=protein_headers,
            )

            # Drop columns for unmapped combinations
            na_cols = [c for c in curr_cell_data.columns if c == "N/A" or str(c).startswith("N/A_")]
            if na_cols:
                curr_cell_data = curr_cell_data.drop(columns=na_cols)

            if self.df_cell_data is None:
                first_channel_num = channel_num
                self.df_cell_data = curr_cell_data
            else:
                self.df_cell_data = merge_channel_cell_data(
                    existing_df=self.df_cell_data,
                    curr_cell_data=curr_cell_data,
                    first_channel_num=first_channel_num,
                    channel_num=channel_num,
                )

            channels_done += 1
            self.channel_done.emit(channels_done, channels_total)
        self.progress.emit(100, "Cell Data is Generated")

    def cancel(self):
        self._cancel_requested = True
        self.progress.emit(99, "Cancelling...")

    def set_color_codes(self, channel_to_code):
        assert isinstance(channel_to_code, dict)
        self.channel_to_color_code = channel_to_code.copy()

    def set_source_uuid(self, source_uuid):
        if source_uuid is None:
            self.source_uuid = None
            return
        self.source_uuid = str(source_uuid)

    def set_project_context(self, project_name, is_temp_project=False):
        self.project_name = project_name
        self.is_temp_project = bool(is_temp_project)

    def save_cell_data(self):
        logger.info("saving cell data")
        if self.df_cell_data is None:
            self.critical_error("Cannot save. No cell data available")
            return

        suggested_name = default_project_prefixed_filename(
            "cell_data.csv",
            self.project_name,
            is_temp_project=self.is_temp_project,
        )
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save Cell Data File", suggested_name, "*.csv;;*.xlsx;; All Files(*)"
        )
        if not file_name:
            return

        passing_ids = self._get_passing_cell_ids()
        if passing_ids is not None:
            df_to_save = self.df_cell_data[
                self.df_cell_data["CellID"].isin(passing_ids)
            ]
            logger.info(
                "Threshold filter applied: %d / %d cells pass (threshold=%.2f)",
                len(df_to_save), len(self.df_cell_data),
                self.params["bead_per_protein_threshold"],
            )
        else:
            df_to_save = self.df_cell_data

        df_to_save.to_csv(file_name, index=False)

    def get_adjusted_median_intensity(self, bead_x, bead_y, bead_median_threshold=5000):
        """
        Calculate the adjusted median intensity given the bead coordinates

        :param bead_x: The x-coordinate of the bead
        :param bead_y: The y-coordinate of the bead
        :param bead_median_threshold: the threshold needed to apply median intensity correction
        :type bead_x: int
        :type bead_y: int
        :type bead_median_threshold: int

        :returns: The adjusted median intensity value of the bead
        :rtype: float
        """

        if self.protein_signal_array is None:
            return

        radius_bg = self.params["radius_bg"]
        radius_fg = self.params["radius_fg"]

        # Extract the 5x5 region around the bead
        bead_region = self.protein_signal_array[
            bead_y - radius_fg : bead_y + radius_fg + 1,
            bead_x - radius_fg : bead_x + radius_fg + 1,
        ]

        # Calculate the mean and median intensity of the 5x5 bead region
        mean_5x5 = np.mean(bead_region)
        bead_median_org = np.median(bead_region)
        bead_median = bead_median_org.copy()

        # Extract the 15x15 surrounding region
        surrounding_region = self.protein_signal_array[
            bead_y - radius_bg : bead_y + radius_bg + 1,
            bead_x - radius_bg : bead_x + radius_bg + 1,
        ]  # Convert to float to handle NaN values

        # Ensure the 15x15 region is valid
        if surrounding_region.shape != (15, 15):
            return bead_median_org  # Return unadjusted median if the 15x15 region is invalid

        # Mask out the 5x5 region from the 15x15 region
        surrounding_region[
            bead_y - radius_fg : bead_y + radius_fg + 1,
            bead_x - radius_fg : bead_x + radius_fg + 1,
        ] = 0

        # Calculate the mean intensity of the surrounding 15x15 area, excluding
        # the 5x5 region
        surrounding_mean_15x15 = np.nanmean(surrounding_region)

        # Apply correction only if 15x15 mean is 1.5x greater than 5x5 mean,
        # and bead median > threshold
        if (
            surrounding_mean_15x15 > 1.5 * mean_5x5
            and bead_median > bead_median_threshold
        ):
            # Calculate the correction factor and apply linear correction
            correction_factor = mean_5x5 * (mean_5x5 / surrounding_mean_15x15)
            y = self.linear_correction(correction_factor)

            # Apply the correction to the bead median
            bead_median = bead_median - y + 2000

        # Ensure no negative values
        if bead_median < 1:
            bead_median = 1

        # Return the final adjusted bead median
        return bead_median

    def linear_correction(self, x):
        """Define the linear function for the correction equation"""
        return 0.8266 * x + 3970.1

    def clear_segmentation_labels(self):
        """Reset loaded StarDist labels."""
        self.segmentation_labels = np.array([], dtype=np.int32)

    def load_segmentation_labels(self, stardist: ImageWrapper) -> None:
        if stardist.data.size == 0:
            logger.warning("Received empty stardist label image")
            self.clear_segmentation_labels()
            return
        logger.debug("stardist label dtype: %s", stardist.data.dtype)
        logger.debug(
            "stardist label max and min %s %s",
            np.max(stardist.data),
            np.min(stardist.data),
        )
        self.segmentation_labels = np.asarray(stardist.data, dtype=np.int32)
        self._remap_labels_spatially()

    def _remap_labels_spatially(self):
        """Remap cell labels so IDs increase from top-left to bottom-right (row-major)."""
        regions = regionprops(self.segmentation_labels)
        if not regions:
            return
        regions_sorted = sorted(regions, key=lambda r: (r.centroid[0], r.centroid[1]))
        max_label = self.segmentation_labels.max()
        remap = np.zeros(max_label + 1, dtype=np.int32)
        for new_id, region in enumerate(regions_sorted, start=1):
            remap[region.label] = new_id
        self.segmentation_labels = remap[self.segmentation_labels]

    def get_filtered_bead_count(self, num_proteins: int):
        if self.bead_data is None or self.segmentation_labels.size == 0:
            self.error_signal.emit("Bead data or stardist labels not loaded.")
            return
        coords = self.bead_data[:, 0:2].astype(int)
        x_limit, y_limit = self.segmentation_labels.shape[1], self.segmentation_labels.shape[0]
        in_bounds = (
            (coords[:, 0] >= 0) & (coords[:, 0] < x_limit)
            & (coords[:, 1] >= 0) & (coords[:, 1] < y_limit)
        )
        coords = coords[in_bounds]
        cell_ids = self.segmentation_labels[coords[:, 1], coords[:, 0]]

        # Count beads per cell
        in_cell_mask = cell_ids > 0
        cell_ids_in_cells = cell_ids[in_cell_mask]
        unique_cells_with_beads, bead_counts = np.unique(cell_ids_in_cells, return_counts=True)

        # Total cells in segmentation (all non-zero unique cell IDs)
        total_cells = len(np.unique(self.segmentation_labels[self.segmentation_labels > 0]))
        if total_cells == 0:
            self.protein_distribution_ready.emit(
                {"distribution": [], "per_cell_counts": [], "num_proteins": num_proteins, "total_cells": 0}
            )
            return

        # Per-cell bead counts: cells with beads + zeros for cells with no beads
        cells_with_0 = total_cells - len(unique_cells_with_beads)
        per_cell_counts = np.concatenate(
            [bead_counts, np.zeros(cells_with_0, dtype=np.int64)]
        )

        # Build distribution table rows 0 … (num_proteins - 1), then a final "X+" row
        distribution = []
        distribution.append((0, cells_with_0 / total_cells * 100))
        for n in range(1, num_proteins):
            count = int(np.sum(bead_counts == n))
            distribution.append((n, count / total_cells * 100))
        # Last row: num_proteins or more
        count_last = int(np.sum(bead_counts >= num_proteins))
        distribution.append((f"{num_proteins}+", count_last / total_cells * 100))

        self.protein_distribution_ready.emit({
            "distribution": distribution,
            "per_cell_counts": per_cell_counts.tolist(),
            "num_proteins": num_proteins,
            "total_cells": total_cells,
        })

    def set_bead_data(self, bead_data):
        if isinstance(bead_data, np.ndarray):
            self.bead_data = bead_data

    def set_radius_fg(self, value):
        self.params["radius_fg"] = value

    def set_radius_bg(self, value):
        self.params["radius_bg"] = value

    def set_crosstalk_ratio(self, value):
        self.params["crosstalk_ratio"] = float(value)

    def set_bead_per_protein_threshold(self, threshold: float, num_proteins: int):
        """Store the threshold and num_proteins for save-time filtering."""
        self.params["bead_per_protein_threshold"] = threshold
        self._filter_num_proteins = num_proteins

    def _get_passing_cell_ids(self):
        """Return the set of cell IDs that pass the bead/protein threshold.

        Returns None if threshold is 0.0 or data is unavailable (no filtering).
        """
        threshold = self.params["bead_per_protein_threshold"]
        num_proteins = self._filter_num_proteins
        if threshold <= 0.0 or num_proteins <= 0:
            return None
        if self.bead_data is None or self.segmentation_labels.size == 0:
            return None

        coords = self.bead_data[:, 0:2].astype(int)
        x_limit = self.segmentation_labels.shape[1]
        y_limit = self.segmentation_labels.shape[0]
        in_bounds = (
            (coords[:, 0] >= 0) & (coords[:, 0] < x_limit)
            & (coords[:, 1] >= 0) & (coords[:, 1] < y_limit)
        )
        coords = coords[in_bounds]
        cell_ids = self.segmentation_labels[coords[:, 1], coords[:, 0]]

        in_cell_mask = cell_ids > 0
        cell_ids_in_cells = cell_ids[in_cell_mask]
        unique_cells, bead_counts = np.unique(cell_ids_in_cells, return_counts=True)

        ratios = bead_counts / num_proteins
        passing_mask = ratios >= threshold
        return set(unique_cells[passing_mask].tolist())

    def _subtract_background(self, array, blur_percentage=1):
        """Gaussian blur subtraction helper."""
        blurred_mask = cv.GaussianBlur(array, (101, 101), 0)
        blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
        corrected_array = cv.subtract(array, blurred_mask_adjusted)
        corrected_array = np.clip(corrected_array, 0, 65535).astype(np.uint16)
        return corrected_array

    def blur_and_set_protein_layer(self, blur_percentage=1):
        """
        Applies Gaussian blur to the protein signal layer and subtracts
        the specified percentage of the blurred image from the original.
        """
        if self.protein_signal_array is not None:
            self.protein_signal_array = self._subtract_background(self.protein_signal_array, blur_percentage)
        return True

    def _get_background_subtracted_images(self):
        uuid = self.source_uuid
        if uuid is None:
            uuid = self.storage.get_data("canvas_uuid")
            if uuid is None:
                raise ValueError("Protein Image not found in storage.")
            uuid = uuid["value"]
        
        item = self.storage.get_data(uuid)
        assert item is not None, "item not found in storage"
        data = item.get("data", None)
        assert data is not None, "data not found in storage item"

        images = {}
        for channel_name in self.channel_to_color_code.keys():
            channel_num = int(channel_name.split(" ")[-1])
            c = "Channel " + str(channel_num)
            wrapper = data[c]
            if is_segmentation_channel(wrapper):
                raise ValueError(
                    f"{c} is a virtual StarDist channel and cannot be used for generation."
                )
            images[channel_name] = self._subtract_background(wrapper.data)
        return images

    def calculate_crosstalk_suppression(self, ratio_threshold):
        """
        Calculate crosstalk suppression mask and count suppressed bead-channel assignments.
        Returns (suppressed_count, suppressed_mask) where suppressed_mask has shape (num_beads, num_channels)
        """
        if self.bead_data is None or not self.channel_to_color_code:
            return 0, np.array([])

        # Check cache validity
        cache_valid = False
        if (
            hasattr(self, "_crosstalk_cache")
            and self._crosstalk_cache is not None
            and self._crosstalk_cache.get("bead_data") is self.bead_data
            and self._crosstalk_cache.get("channels") == list(self.channel_to_color_code.keys())
            and self._crosstalk_cache.get("source_uuid") == self.source_uuid
            and self._crosstalk_cache.get("radius_fg") == self.params.get("radius_fg")
            and self._crosstalk_cache.get("radius_bg") == self.params.get("radius_bg")
        ):
            cache_valid = True

        channel_names = list(self.channel_to_color_code.keys())
        num_channels = len(channel_names)

        if cache_valid:
            raw_intensities = self._crosstalk_cache["raw_intensities"]
            max_vals = self._crosstalk_cache["max_vals"]
            max_ch_indices = self._crosstalk_cache["max_ch_indices"]
            ratios = self._crosstalk_cache["ratios"]
        else:
            # 1. Get background-subtracted images for all channels
            images_dict = self._get_background_subtracted_images()
            num_beads = len(self.bead_data)

            # 2. Extract X, Y bead coordinates
            bead_xs = self.bead_data[:, 0].astype(int)
            bead_ys = self.bead_data[:, 1].astype(int)

            # 3. Calculate raw median intensities for all beads across all channels in a vectorized way
            raw_intensities = np.zeros((num_beads, num_channels), dtype=float)

            for c_idx, ch_name in enumerate(channel_names):
                img = images_dict[ch_name]
                height, width = img.shape
                
                # Pad the image by 2 pixels (since foreground radius is 2) to safely handle border beads
                padded_img = np.pad(img, pad_width=2, mode='constant', constant_values=0)
                
                # Map coordinates to padded space
                padded_xs = bead_xs + 2
                padded_ys = bead_ys + 2
                
                # Check image bounds in padded space (X and Y coordinates should be >= 0 and < width / height)
                valid_coords_mask = (bead_xs >= 0) & (bead_xs < width) & (bead_ys >= 0) & (bead_ys < height)
                
                # Meshgrid offsets for 5x5 neighborhood
                dx, dy = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
                dx = dx.flatten()
                dy = dy.flatten()

                # For all beads: shape (num_beads, 25)
                neighbor_xs = padded_xs[:, np.newaxis] + dx
                neighbor_ys = padded_ys[:, np.newaxis] + dy

                # Retrieve neighborhoods using advanced indexing
                neighbor_xs = np.clip(neighbor_xs, 0, width + 3)
                neighbor_ys = np.clip(neighbor_ys, 0, height + 3)
                
                neighborhoods = padded_img[neighbor_ys, neighbor_xs] # shape (num_beads, 25)
                raw_medians = np.median(neighborhoods, axis=1)
                
                # Only set raw medians for beads that are within image bounds
                raw_intensities[:, c_idx] = np.where(valid_coords_mask, raw_medians, 0.0)

            # Compute max_vals, max_ch_indices, ratios
            max_vals = np.max(raw_intensities, axis=1) # shape (num_beads,)
            max_ch_indices = np.argmax(raw_intensities, axis=1) # shape (num_beads,)
            ratios = max_vals[:, np.newaxis] / np.maximum(raw_intensities, 1.0) # shape (num_beads, num_channels)

            # Save to cache
            self._crosstalk_cache = {
                "bead_data": self.bead_data,
                "channels": list(self.channel_to_color_code.keys()),
                "source_uuid": self.source_uuid,
                "radius_fg": self.params.get("radius_fg"),
                "radius_bg": self.params.get("radius_bg"),
                "raw_intensities": raw_intensities,
                "max_vals": max_vals,
                "max_ch_indices": max_ch_indices,
                "ratios": ratios,
            }

        # A bead-channel assignment is suppressed if:
        # - The ratio is >= ratio_threshold
        # - The raw intensity in this channel is > 0 (to avoid suppressing zero/background pixels)
        suppressed_mask = (ratios >= ratio_threshold) & (raw_intensities > 0)
        
        # Make sure we don't suppress the max channel itself
        for c_idx in range(num_channels):
            suppressed_mask[:, c_idx] &= (max_ch_indices != c_idx)

        suppressed_count = int(np.sum(suppressed_mask))
        return suppressed_count, suppressed_mask
