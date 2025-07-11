import copy
import gc

# Standard library imports
import os
import threading
import uuid
import xml.etree.ElementTree as ET
from typing import Dict, Optional, OrderedDict, Union

import cv2

# Third-party imports
import numpy as np
import tifffile as tiff
from cv2 import LUT
from matplotlib import colormaps
from PIL import Image

# PyQt6 imports
from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pystackreg.util import to_uint16

# Local/project imports
from core.Worker import Worker
from utils import (
    adjustContrast,
    create_lut,
    numpy_to_qimage,
    qimage_to_numpy,
    scale_adjust,
    to_uint8,
)


class MemoryEfficientImageCache:
    """Memory-efficient cache with size limits and automatic cleanup."""

    def __init__(self, max_cache_size_mb=500, max_entries_per_channel=5):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_entries_per_channel = max_entries_per_channel
        self.cache = {}  # {uuid: {channel: OrderedDict of {cache_key: image_data}}}
        self.current_size_bytes = 0

    def get(self, uuid, channel, cache_key):
        """Get cached image if available."""
        if uuid not in self.cache or channel not in self.cache[uuid]:
            return None

        channel_cache = self.cache[uuid][channel]
        if cache_key in channel_cache:
            # Move to end (most recently used)
            image_data = channel_cache[cache_key]
            del channel_cache[cache_key]
            channel_cache[cache_key] = image_data
            return image_data
        return None

    def put(self, uuid, channel, cache_key, image_data):
        """Cache image data with memory management."""
        if uuid not in self.cache:
            self.cache[uuid] = {}
        if channel not in self.cache[uuid]:
            self.cache[uuid][channel] = OrderedDict()

        channel_cache = self.cache[uuid][channel]
        image_size_bytes = image_data.nbytes

        # Remove if already exists to update size tracking
        if cache_key in channel_cache:
            old_data = channel_cache[cache_key]
            self.current_size_bytes -= old_data.nbytes
            del channel_cache[cache_key]

        # Check if we need to free memory
        while (
            self.current_size_bytes + image_size_bytes > self.max_cache_size_bytes
            or len(channel_cache) >= self.max_entries_per_channel
        ):
            if not channel_cache:
                # Try to find another channel in this uuid or other uuids to evict from
                evicted = False
                for other_channel_cache in self.cache[uuid].values():
                    if other_channel_cache:
                        _, old_data = other_channel_cache.popitem(last=False)
                        self.current_size_bytes -= old_data.nbytes
                        del old_data
                        evicted = True
                        break

                if not evicted:
                    # Try other uuids
                    for other_uuid_cache in self.cache.values():
                        for other_channel_cache in other_uuid_cache.values():
                            if other_channel_cache:
                                _, old_data = other_channel_cache.popitem(last=False)
                                self.current_size_bytes -= old_data.nbytes
                                del old_data
                                evicted = True
                                break
                        if evicted:
                            break

                if not evicted:
                    break

            else:
                # Remove least recently used item from current channel
                _, old_data = channel_cache.popitem(last=False)
                self.current_size_bytes -= old_data.nbytes
                del old_data

        # Add new data
        channel_cache[cache_key] = image_data
        self.current_size_bytes += image_size_bytes

        # Force garbage collection if cache is getting large
        if self.current_size_bytes > self.max_cache_size_bytes * 0.8:
            gc.collect()

    def clear_channel(self, uuid, channel):
        """Clear cache for specific channel of specific uuid."""
        if uuid in self.cache and channel in self.cache[uuid]:
            channel_cache = self.cache[uuid][channel]
            for image_data in channel_cache.values():
                self.current_size_bytes -= image_data.nbytes
            channel_cache.clear()
            gc.collect()

    def clear_uuid(self, uuid):
        """Clear all cache for specific uuid."""
        if uuid in self.cache:
            for channel_cache in self.cache[uuid].values():
                for image_data in channel_cache.values():
                    self.current_size_bytes -= image_data.nbytes
                channel_cache.clear()
            del self.cache[uuid]
            gc.collect()

    def clear_all(self):
        """Clear entire cache."""
        for uuid_cache in self.cache.values():
            for channel_cache in uuid_cache.values():
                channel_cache.clear()
        self.cache.clear()
        self.current_size_bytes = 0
        gc.collect()

    def get_memory_info(self):
        """Get current memory usage info."""
        total_entries = 0
        total_channels = 0
        for uuid_cache in self.cache.values():
            total_channels += len(uuid_cache)
            for channel_cache in uuid_cache.values():
                total_entries += len(channel_cache)

        return {
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_cache_size_bytes / (1024 * 1024),
            "uuids": len(self.cache),
            "channels": total_channels,
            "total_entries": total_entries,
        }


class ImageStorage:
    """
    Thread-safe singleton class for managing image data storage.
    This class implements a singleton pattern with thread-safe operations for storing,
    retrieving, and managing image data. It uses locks to ensure thread safety during
    both singleton initialization and data operations.
    Attributes:
        image_list (dict): Dictionary storing image data with image_id as keys
        _data_lock (threading.Lock): Lock for thread-safe access to image_list
        _lock (threading.Lock): Class-level lock for singleton initialization
    Methods:
        add_data(image_id, data): Add new image data to storage
        get_data(image_id): Retrieve image data by ID (returns deep copy)
        remove_data(image_id): Remove image data from storage
        update_data(image_id, channel, new_img): Update specific channel of image data
        clear_data(): Clear all stored image data
        update_name(uuid, new_name): Update the name field of stored image data
    Note:
        - Automatically triggers garbage collection after data modifications
        - Returns deep copies of data to prevent external modifications
        - Thread-safe for concurrent access from multiple threads
    """

    _instance = None
    _lock = threading.Lock()  # Class-level lock for singleton init

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_instance()
        return cls._instance

    def _init_instance(self):
        self.image_list = {}
        self._data_lock = threading.Lock()  # Instance-level lock for image_list

    def __len__(self):
        with self._data_lock:
            return len(self.image_list)

    def __getitem__(self, image_id):
        return self.get_data(str(image_id))

    def __setitem__(self, image_id, data):
        self.add_data(str(image_id), data)

    def add_data(self, image_id, data):
        with self._data_lock:
            self.image_list[str(image_id)] = data

    def get_data(self, image_id):
        with self._data_lock:
            data = self.image_list.get(str(image_id))
            return data

    def remove_data(self, image_id):
        with self._data_lock:
            if str(image_id) in self.image_list:
                del self.image_list[str(image_id)]
                gc.collect()

    def update_data(self, image_id, channel="Channel 1", new_img=None):
        with self._data_lock:
            if image_id in self.image_list:
                self.image_list[str(image_id)][channel] = new_img

    def clear_data(self):
        with self._data_lock:
            self.image_list = {}
            gc.collect()

    def update_name(self, uuid, new_name):
        self.update_data(uuid, "name", new_name)


# Could maybe use collections.namedtuple to represent ImageWrapper
class ImageWrapper:
    def __init__(self, data, name="", cmap="gray"):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")

        self.name = name
        self.cmap = cmap
        self.data = data.copy()
        self.contrast_min = 0
        self.contrast_max = 255

    def copy(self):
        arr = copy.copy(self.data)
        return ImageWrapper(data=arr, name=self.name, cmap=self.cmap)

    def get_uint8_data(self):
        if self.data.dtype == np.uint8:
            return self.data
        arr = to_uint8(self.data)
        assert arr.dtype == np.uint8
        return arr

    def get_uint16_data(self):
        if self.data.dtype == np.uint16:
            return self.data
        arr = to_uint16(self.data)
        assert arr.dtype == np.uint16
        return arr

    def __repr__(self):
        return f"ImageWrapper(name={self.name}, shape={self.data.shape}, dtype={self.data.dtype}, cmap={self.cmap})"

    def __str__(self):
        return self.__repr__()

    def __bool__(self):
        return self.data.any()


class BaseGraphicsView(QWidget):
    image_signal = pyqtSignal(dict, bool)
    update_progress = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)
    fill_metadata = pyqtSignal(dict)
    update_manager = pyqtSignal(dict, str)
    add_image_to_storage = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(QSize(300, 300))
        self.scene = QGraphicsView()
        self.scene.setScene(QGraphicsScene(self))
        self.reset_pixmap = None
        self.reset_pixmap_item = None
        self.pixmap = None
        self.pixmap_item = None
        self.np_channels: dict[str, ImageWrapper] = {}
        self.display_channels: dict[str, QPixmap] = {}
        self.reset_np_channels = {}
        self.current_channel = 0
        self.image_cache = {}
        self.lut_cache = {}
        self.image_count = 0
        self.storage = ImageStorage()
        self.image_wrapper = ImageWrapper(
            np.array([]), ""
        )  # Added for single image compatibility
        self.uuid = None
        self.num_channels = 0

    def set_uuid(self, uuid):
        self.uuid = uuid

    @property
    def is_layered(self):
        return len(self.np_channels.items()) > 1

    def dragEnterEvent(self, event: QDragEnterEvent | None):  # type: ignore
        self._accept_if_valid(event)

    def dragMoveEvent(self, event: QDragMoveEvent | None):  # type: ignore
        self._accept_if_valid(event)

    def _accept_if_valid(
        self, event: QDragEnterEvent | QDragMoveEvent | QDropEvent | None
    ):
        if event and event.mimeData():
            mime_data = event.mimeData()
            if mime_data is not None and mime_data.hasUrls():
                event.acceptProposedAction()
                return True
        return False

    def _read_tiff_pages(self, file_path):
        """Generator that yields pages one at a time to avoid memory overload."""
        with tiff.TiffFile(file_path) as tif:
            total_pages = len(tif.pages)
            valid_page_count = 0
            self.num_channels = total_pages
            for i, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # Check if the image is blank
                    if np.all(image == image.flat[0]):
                        continue

                    valid_page_count += 1

                    yield image, valid_page_count

                except Exception as e:
                    print(f"Error reading page {i}: {e}")
                    continue

    def _subsample_for_display(
        self, image: np.ndarray, max_dimension: int = 1024
    ) -> np.ndarray:
        """Simple subsampling for display purposes."""
        h, w = image.shape[:2]

        scale_factor = min(max_dimension / h, max_dimension / w, 1.0)

        if scale_factor >= 1.0:
            return image

        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        step_h = max(1, h // new_h)
        step_w = max(1, w // new_w)
        return image[::step_h, ::step_w]

    def _add_to_canvas(
        self,
        file_name: str,
        adjust_contrast=False,
        subsample_for_emit=False,
        max_display_size=1024,
    ) -> np.ndarray:
        if file_name.endswith((".tiff", ".tif")):
            return self._process_tiff_image(
                file_name, adjust_contrast, subsample_for_emit, max_display_size
            )
        else:
            return self._process_single_image(
                file_name, subsample_for_emit, max_display_size
            )

    def _process_tiff_image(
        self,
        file_name: str,
        adjust_contrast: bool,
        subsample_for_emit: bool,
        max_display_size: int,
    ) -> np.ndarray:
        """Process multi-channel TIFF images."""
        # Handle metadata
        metadata_widget = MetaData()
        metadata = metadata_widget.parse_metadata(file_name)
        self.fill_metadata.emit(metadata)

        emit_data = {}
        channel_one_image = None

        # Process pages one at a time using generator
        for image, channel_num in self._read_tiff_pages(file_name):
            channel_name = f"Channel {channel_num}"
            image_adjusted = self._apply_contrast_adjustment(image, adjust_contrast)

            print(
                f"Processing {channel_name}, shape: {image.shape}, dtype: {image.dtype}"
            )

            # Store full resolution version
            self._store_channel_data(channel_name, image_adjusted)

            # Prepare display version
            display_image = self._prepare_display_image(
                image_adjusted, subsample_for_emit, max_display_size
            )
            emit_data[channel_name] = display_image

            # Store first channel for return value
            if channel_num == 1:
                channel_one_image = display_image

            self._log_memory_usage(
                channel_name, image_adjusted, display_image, subsample_for_emit
            )
            self._update_progress(channel_num, self.num_channels)

        self._emit_multichannel_data(emit_data, subsample_for_emit)
        self._add_to_manager(file_name)
        assert channel_one_image is not None, "No valid channels found in TIFF."
        return channel_one_image

    def _process_single_image(
        self, file_name: str, subsample_for_emit: bool, max_display_size: int
    ) -> np.ndarray:
        """Process single channel images."""
        self._prepare_channels_for_new_image()

        print("Processing single image")
        emit_data = {}
        channel_one_image = np.array(Image.open(file_name))
        channel_name = "Channel 1"

        # Convert to uint8 for consistency
        self._store_channel_data(channel_name, channel_one_image)
        display_image = self._prepare_display_image(
            channel_one_image, subsample_for_emit, max_display_size
        )
        emit_data[channel_name] = display_image
        self._emit_multichannel_data(emit_data, subsample_for_emit)
        self._add_to_manager(file_name)
        return display_image

    def _replace_canvas(
        self,
        data: Union[np.ndarray, Dict[str, ImageWrapper]],
        as_new_image: bool = False,
        new_image_name: Optional[str] = None,
        target_channel: str = "Channel 1",
        subsample_for_emit: bool = False,
        max_display_size: int = 1024,
    ) -> np.ndarray:
        """Automatically determine whether to use single or multichannel replacement."""

        if isinstance(data, dict):
            if not data:
                raise ValueError("Data dict must be non empty")
            processed_data = {}
            for key, value in data.items():
                if not isinstance(key, str):
                    raise TypeError(f"Key {key} must be str")
                # If ImageWrapper —> convert to ndarray
                if isinstance(value, ImageWrapper):
                    new_value = value.data
                    print(f"Converted ImageWrapper at key {key} to ndarray")
                elif isinstance(value, np.ndarray):
                    new_value = value
                else:
                    raise TypeError(
                        f"Value for key {key} must be np.ndarray or ImageWrapper"
                    )
                processed_data[key] = new_value
            # single channels also use this
            ret = self._replace_canvas_multichannel(
                processed_data, target_channel, subsample_for_emit, max_display_size
            )
            if as_new_image:
                assert (
                    new_image_name is not None
                ), "Image name must be provided for new image"
                self._add_to_manager(new_image_name)

            return ret
        elif isinstance(data, np.ndarray):
            # shouldn't be used
            return self._replace_canvas_single(
                data, subsample_for_emit, max_display_size
            )
        else:
            raise ValueError(
                "Data must be either numpy array or dictionary of numpy arrays"
            )

    def _replace_canvas_multichannel(
        self,
        channels_data: Dict[str, np.ndarray],
        target_channel: str = "Channel 1",
        subsample_for_emit: bool = False,
        max_display_size: int = 1024,
    ) -> np.ndarray:
        """Replace canvas with multichannel image data - target channel first, others in background."""
        self._prepare_channels_for_new_image()

        # Initialize thread lock for background processing

        if hasattr(self, "_background_worker") and self._background_worker.isRunning():
            print("Warning: Previous channel processing still ongoing. Canceling it.")
            self._background_worker.terminate()
        # Process target channel first for immediate display
        if target_channel not in channels_data:
            raise ValueError(f"{target_channel} was not found in the data")

        target_image_data = channels_data[target_channel]

        # Process target channel immediately
        # fill channels with dummy data:
        for channel_name, channel_data in channels_data.items():
            if channel_name != target_channel:
                self._store_channel_data(
                    channel_name, channel_data, replace_image_wrapper=False
                )
        self._store_channel_data(target_channel, target_image_data)
        display_channel_data = self._prepare_display_image(
            target_image_data, subsample_for_emit, max_display_size
        )

        print(
            f"Processing {target_channel} (primary), shape: {target_image_data.shape}, dtype: {target_image_data.dtype}"
        )

        # Emit target channel immediately for display
        emit_data = {target_channel: display_channel_data}
        self._emit_multichannel_data(emit_data, subsample_for_emit)
        self.current_channel = int(target_channel[-1]) - 1

        # Process remaining channels in background if there are any
        remaining_channels = {
            k: v for k, v in channels_data.items() if k != target_channel
        }

        if remaining_channels:
            background_worker = Worker(
                self._process_remaining_channels,
                remaining_channels,
                subsample_for_emit,
                max_display_size,
            )
            self._background_worker = background_worker
            background_worker.signal.connect(self._on_background_channels_completed)
            background_worker.finished.connect(background_worker.quit)
            background_worker.finished.connect(background_worker.deleteLater)
            background_worker.start()

        self._clear_caches()
        self.image_count += 1
        print(
            f"Canvas replaced - {target_channel} ready, {len(remaining_channels)} channels processing in background"
        )

        return display_channel_data

    def _process_remaining_channels(
        self,
        remaining_channels: Dict[str, np.ndarray],
        subsample_for_emit: bool,
        max_display_size: int,
    ) -> Dict[str, np.ndarray]:
        """Process remaining channels in background thread."""
        processed_channels = {}

        for channel_name, image_data in remaining_channels.items():
            try:
                # Store full resolution version
                self._store_channel_data(channel_name, image_data)

                # Prepare display version
                display_image = self._prepare_display_image(
                    image_data, subsample_for_emit, max_display_size
                )
                processed_channels[channel_name] = display_image

                print(
                    f"Background processing {channel_name}, shape: {image_data.shape}, dtype: {image_data.dtype}"
                )

            except Exception as e:
                print(f"Error processing {channel_name} in background: {e}")
                continue

        return processed_channels

    @pyqtSlot(object)
    def _on_background_channels_completed(
        self, processed_channels: Dict[str, np.ndarray]
    ):
        """Handle completion of background channel processing."""
        if processed_channels:
            # Update the full multichannel data
            all_emit_data = {
                **{
                    f"Channel {self.current_channel + 1}": self.np_channels[
                        f"Channel {self.current_channel + 1}"
                    ].data
                },
                **processed_channels,
            }

            # Create display wrappers for all channels
            display_wrappers = {
                name: ImageWrapper(data, name) for name, data in all_emit_data.items()
            }
            # self.image_signal.emit(display_wrappers, True)

            print(
                f"Background processing completed - {len(processed_channels)} additional channels ready"
            )

    def _prepare_channels_for_new_image(self):
        self.np_channels = {}
        self.reset_np_channels = {}
        self.display_channels = {}

    def _replace_canvas_single(
        self,
        image_data: Union[np.ndarray, Dict[str, ImageWrapper]],
        subsample_for_emit: bool = False,
        max_display_size: int = 1024,
    ) -> np.ndarray:
        """Replace canvas with single channel image data."""
        self._prepare_channels_for_new_image()
        if isinstance(image_data, dict):
            image_data = image_data["Channel 1"].data
        channel_name = "Channel 1"

        # Convert to uint8 for consistency
        img_data = to_uint8(image_data)

        # Store full resolution
        self.image_wrapper = ImageWrapper(img_data, "Channel 1")
        self.np_channels[channel_name] = self.image_wrapper

        # Handle display emission
        display_img = self._handle_single_image_display(
            img_data, subsample_for_emit, max_display_size
        )

        self._clear_caches()

        # Update manager without filename
        print("Emitting to update manager")
        # self.update_manager.emit(self.np_channels, "replaced_single")
        self.image_count += 1
        print(f"Canvas replaced with single image, shape: {image_data.shape}")
        return display_img

    def _apply_contrast_adjustment(
        self, image: np.ndarray, adjust_contrast: bool
    ) -> np.ndarray:
        """Apply contrast adjustment if requested."""
        if adjust_contrast:
            scaled = scale_adjust(image)
            return adjustContrast(scaled)
        return image

    def _store_channel_data(
        self, channel_name: str, image_data: np.ndarray, replace_image_wrapper=True
    ) -> None:
        """Store channel data in full resolution containers."""
        self.np_channels[channel_name] = ImageWrapper(image_data, channel_name)
        self.reset_np_channels[channel_name] = ImageWrapper(image_data, channel_name)
        if replace_image_wrapper:
            self.image_wrapper = ImageWrapper(image_data, channel_name)

    def _prepare_display_image(
        self,
        image_data: np.ndarray,
        subsample_for_emit: bool = False,
        max_display_size: int = 0,
    ) -> np.ndarray:
        """Prepare image for display (subsample if needed)."""
        if subsample_for_emit:
            if not max_display_size:
                raise ValueError(
                    "If subsample_for_emit, max_display_size must be specified."
                )
            subsampled = self._subsample_for_display(image_data, max_display_size)
            print(f"  Original: {image_data.shape}, Subsampled: {subsampled.shape}")
            image_data = subsampled
        image_data = to_uint8(image_data)
        return image_data

    def _handle_single_image_display(
        self, image_data: np.ndarray, subsample_for_emit: bool, max_display_size: int
    ) -> np.ndarray:
        """Handle single image display emission."""
        if subsample_for_emit and image_data.size > max_display_size * max_display_size:
            subsampled = self._subsample_for_display(image_data, max_display_size)
            self.image_signal.emit(subsampled, True)
            print(
                f"Single image subsampled from {image_data.shape} to {subsampled.shape}"
            )
            return subsampled
        else:
            self.image_signal.emit(image_data, True)
            return image_data

    def _emit_multichannel_data(
        self, emit_data: Dict[str, np.ndarray], subsample_for_emit: bool
    ) -> None:
        """Notify listeners that number of channels has changed and emit data."""
        if emit_data:
            if subsample_for_emit:
                # Create wrappers for subsampled data
                display_wrappers = {
                    name: ImageWrapper(data, name) for name, data in emit_data.items()
                }
                # self.np_channels.update(display_wrappers)
                self.image_signal.emit(display_wrappers, True)
            else:
                # Emit full resolution

                self.image_signal.emit(self.np_channels, True)

    def _log_memory_usage(
        self,
        channel_name: str,
        full_image: np.ndarray,
        display_image: np.ndarray,
        subsample_for_emit: bool,
    ) -> None:
        """Log memory usage for debugging."""
        full_size_mb = full_image.nbytes / (1024 * 1024)
        if subsample_for_emit:
            display_size_mb = display_image.nbytes / (1024 * 1024)
            print(
                f"{channel_name} - Full: {full_size_mb:.2f} MB, Display: {display_size_mb:.2f} MB"
            )
        else:
            print(f"{channel_name} - Size: {full_size_mb:.2f} MB")

    def _update_progress(self, channel_num: int, total_channels) -> None:
        """Update processing progress."""
        progress = 10 + int(channel_num / total_channels * 70)
        self.update_progress.emit(progress, f"Processing Channel {channel_num}")

    def _add_to_manager(self, file_name: str) -> None:
        """Finalize processing with cleanup and emissions."""
        self._clear_caches()
        self.update_progress.emit(100, "Image Loaded")

        print("Emitting to update manager")
        self.np_channels = {
            k: self.np_channels[k] for k in sorted(self.np_channels.keys())
        }
        self.update_manager.emit(self.np_channels, file_name)
        self.image_count += 1

    def _clear_caches(self) -> None:
        """Clear image and LUT caches."""
        self.image_cache.clear()
        self.lut_cache.clear()


class ReferenceGraphicsView(BaseGraphicsView):
    update_reference = pyqtSignal(QPixmap, bool)

    def dropEvent(self, event: QDropEvent):  # type: ignore
        if self._accept_if_valid(event):
            for url in event.mimeData().urls():  # type: ignore ;_accept_if_valid ensures mimeData is not None
                file_path = url.toLocalFile()
                if file_path:
                    self.add_to_canvas(file_path)

    def add_to_canvas(self, file_path: str):
        self.reference_worker = Worker(self._add_to_canvas, file_path, False)
        self.reference_worker.start()
        self.reference_worker.signal.connect(self.display_result)
        self.reference_worker.finished.connect(self.reference_worker.quit)
        self.reference_worker.finished.connect(self.reference_worker.deleteLater)

    def set_uuid(self, uuid):
        """Set UUID for the reference image."""
        self.uuid = uuid
        self.storage.add_data("reference_uuid", {"value": uuid})

    def display_result(self, image):
        qimage = numpy_to_qimage(image)
        self.pixmap = QPixmap(qimage)
        self.update_reference.emit(self.pixmap, self.is_layered)


import typing

if typing.TYPE_CHECKING:
    from controller import Controller


##########################################################
class ImageGraphicsView(BaseGraphicsView):

    canvas_updated = pyqtSignal(QPixmap)
    new_image_added = pyqtSignal(QGraphicsPixmapItem)
    save_image = pyqtSignal(QGraphicsPixmapItem)
    change_slider = pyqtSignal(tuple)
    update_cmap = pyqtSignal(str)
    crop_signal = pyqtSignal(bool)

    def __init__(self, controller: "Controller"):
        super().__init__()
        self.controller = controller
        self.begin_crop = False
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.memory_cache = MemoryEfficientImageCache(max_cache_size_mb=300)
        self.uuid = None

    def set_uuid(self, uuid):
        """Set UUID for the current image (Image Tab)."""
        print("Setting image UUID:", uuid)
        self.uuid = uuid
        self.storage.add_data("canvas_uuid", {"value": uuid})

    def to_pixmap(self, data: QPixmap | np.ndarray | QImage):
        """Sends a pixmap to the canvas for display"""
        # convert pixmap to pixmapItem
        if isinstance(data, QPixmap):
            self.pixmap = data
        elif isinstance(data, QImage):
            self.pixmap = QPixmap(data)
        elif isinstance(data, np.ndarray):
            self.pixmap = QPixmap(numpy_to_qimage(data))
        self.canvas_updated.emit(self.pixmap)
        return self.pixmap

    def clear_canvas(self):
        scene = self.scene.scene()
        if scene:
            scene.clear()
        self.pixmap = QPixmap()
        self.pixmap_item = QGraphicsPixmapItem()
        self.reset_pixmap = None
        self.reset_pixmap_item = None
        self.np_channels = {}
        self.reset_np_channels = {}
        self.current_channel = 0
        self.image_cache = {}
        self.lut_cache = {}
        self.image_wrapper = ImageWrapper(np.array([]), "")
        self.uuid = None
        self.num_channels = 0
        self.canvas_updated.emit(self.pixmap)

    def swap_channel(self, index):
        """Modified swap_channel to wait for background processing if needed."""
        # not 100% so need the len check later
        if (
            getattr(self, "_background_worker", None)
            and self._background_worker.isRunning()
        ):
            print("Waiting for background channel processing to complete...")
            self._background_worker.finished.connect(lambda: self.swap_channel(index))
            return
        else:
            print("no background worker or already finished")
        self.current_channel = index
        channel_num = f"Channel {index+1}"

        self.image_wrapper = self.np_channels.get(
            channel_num, ImageWrapper(np.array([]), "")
        )
        if len(self.image_wrapper.data) == 0:
            print("no data, background worker still workin")
            self._background_worker.finished.connect(lambda: self.swap_channel(index))
            return
        # self.image_wrapper.contrast_max
        display = self._prepare_display_image(self.image_wrapper.data.copy())
        q_pixmap = QPixmap(numpy_to_qimage(display))

        self.canvas_updated.emit(q_pixmap)

    def update_contrast_memory_efficient(self, values):
        """Memory-efficient version of update_contrast method."""
        if self.image_wrapper is None:
            self.error_signal.emit("Canvas is empty")
            return

        contrast_min, contrast_max = int(values[0]), int(values[1])
        self.image_wrapper.contrast_min = contrast_min
        self.image_wrapper.contrast_max = contrast_max

        # Initialize memory-efficient cache if not exists
        if not hasattr(self, "memory_cache"):
            self.memory_cache = MemoryEfficientImageCache()

        contrast_key = (contrast_min, contrast_max)
        cmap_key = self.image_wrapper.cmap
        cache_key = (cmap_key, contrast_key)
        contrast_pixmap = None
        if self.is_layered:
            print("Processing layered image with memory management")
            channel_num = f"Channel {self.current_channel + 1}"
            self.image_wrapper = self.np_channels[channel_num]

            # Check cache first
            cached_image = self.memory_cache.get(self.uuid, channel_num, cache_key)
            if cached_image is not None:
                print(f"Using cached image for {channel_num}")
                contrast_pixmap = QPixmap(numpy_to_qimage(cached_image))
                self.canvas_updated.emit(contrast_pixmap)
            else:
                print(f"Processing new contrast for {channel_num}")
                contrast_pixmap = self._apply_contrast_memory_efficient(
                    channel_num, cache_key, contrast_min, contrast_max
                )
        else:
            # Single layer processing
            cached_image = self.memory_cache.get(self.uuid, "single", cache_key)
            if cached_image is not None:
                contrast_pixmap = QPixmap(numpy_to_qimage(cached_image))
                self.canvas_updated.emit(contrast_pixmap)
            else:
                contrast_pixmap = self._apply_contrast_memory_efficient(
                    "single", cache_key, contrast_min, contrast_max
                )

        # Update slider
        self.change_slider.emit(
            (self.image_wrapper.contrast_min, self.image_wrapper.contrast_max)
        )
        return contrast_pixmap

    def _apply_contrast_memory_efficient(
        self, channel_key, cache_key, contrast_min, contrast_max
    ):
        """Memory-efficient contrast application with caching."""
        try:
            # Apply contrast
            image_to_display = self.apply_contrast(contrast_min, contrast_max)

            # Cache the result
            self.memory_cache.put(
                self.uuid, channel_key, cache_key, image_to_display.copy()
            )

            # Display
            contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))
            self.canvas_updated.emit(contrast_pixmap)

            # Clean up local reference
            del image_to_display

        except MemoryError:
            print("Memory error - clearing cache and retrying")
            self.memory_cache.clear_all()
            gc.collect()

            # Retry with no caching
            image_to_display = self.apply_contrast(contrast_min, contrast_max)
            contrast_pixmap = QPixmap(numpy_to_qimage(image_to_display))
            self.canvas_updated.emit(contrast_pixmap)
            del image_to_display

    def change_cmap(self, cmap_text="default"):
        """changes the colormap given a colormap str valid in matplotlib"""

        if self.image_wrapper is None:
            self.error_signal.emit("Canvas is empty")
            return

        if self.is_layered:
            channel_num = f"Channel {self.current_channel + 1}"
            self.image_wrapper = self.np_channels[channel_num]  # wrapper

        if cmap_text == "default":
            cmap_text = self.image_wrapper.cmap

        if cmap_text not in self.lut_cache:
            lut = self.generate_lut(cmap_text)
            self.lut_cache[cmap_text] = lut  # cache to avoid recalculating LUT
        else:
            lut = self.lut_cache[cmap_text]  # Reuse the cached LUT

        self.image_wrapper.cmap = cmap_text

        image_wrapper_copy = (self.image_wrapper.data).copy()

        return self.label2rgb(scale_adjust(image_wrapper_copy), lut).astype(np.uint8)

    def update_image(self, cmap_text="default"):
        """Updates the current image using the current colormap and contrast settings.
        This only changes the display and does not change the underlying data."""
        print("updating image")

        # update the color map
        image_to_display = self.change_cmap(cmap_text)
        assert image_to_display is not None, "Updating empty image"
        self.to_pixmap(image_to_display)
        self.update_cmap.emit(cmap_text)
        # update the contrast
        assert self.image_wrapper is not None, "Updating empty image wrapper"
        contrast_min, contrast_max = (
            self.image_wrapper.contrast_min,
            self.image_wrapper.contrast_max,
        )  # read contrast settings
        self.update_contrast((contrast_min, contrast_max))

    def generate_lut(self, cmap: str):
        """generate a 8 bit look-up table and converts to rgb space"""
        label_range = np.linspace(0, 1, 256)
        color_map = colormaps.get_cmap(cmap)
        temp = color_map(label_range)
        uint8_temp = np.uint8(temp[:, 2::-1] * 256)
        return uint8_temp.reshape(256, 1, 3)

    def update_contrast(self, values):
        return self.update_contrast_memory_efficient(values)

    def _apply_contrast_and_cache(
        self, channel_num, cache_key, contrast_min, contrast_max
    ):
        return self._apply_contrast_memory_efficient(
            channel_num, cache_key, contrast_min, contrast_max
        )

    def label2rgb(self, labels, lut):
        """applys the look-up table and merges r, g, b channels to form colored image"""
        print(type(labels))
        if len(labels.shape) == 3 and labels.shape[2] == 3:
            r, g, b = cv2.split(labels)
            return cv2.LUT(cv2.merge((r, g, b)), lut)
        else:
            # Ensure labels is 2D before merging
            if len(labels.shape) > 2:
                labels = labels[:, :, 0]  # Take first channel if multi-channel
            return cv2.LUT(cv2.merge((labels, labels, labels)), lut)  # gray to color

    def load_stardist_labels(self, stardist: ImageWrapper):

        self.stardist_labels = stardist.data
        cmap = self.np_channels[f"Channel {self.current_channel+1}"].cmap
        self.image_wrapper = ImageWrapper(
            self.stardist_labels.copy(), name="stardist_label", cmap=cmap
        )
        self.update_image(cmap_text=cmap)
        # !TODO: need to fix; think is broken now after changing update_manger behavior
        item = self.storage.get_data(self.uuid)
        if item:
            name = item["name"]
        else:
            name = f"Unnamed"
        self.update_manager.emit(
            cmap,
            f"Stardist_{name}",
        )

    def add_to_canvas(
        self,
        i: str | ImageWrapper | uuid.UUID | dict[str, ImageWrapper],
        as_new_image=True,
        new_image_name=None,
        target_channel="Channel 1",
    ):
        """add a new image if input is a filename, or can choose to only replace the canvas
        if input is an ImageWrapper or dict of ImageWrapper"""
        self._prepare_channels_for_new_image()
        if hasattr(self, "memory_cache"):
            self.memory_cache.clear_all()
        self.scene.resetTransform()

        # str is filepath
        if isinstance(i, str):
            assert self.storage.get_data(i) is None, "Convert str to UUID instance"
            if not as_new_image:
                raise ValueError(
                    "Cannot replace canvas with a filename, UUID replace not supported yet."
                )
            self.image_worker = Worker(self._add_to_canvas, i)
        elif isinstance(i, ImageWrapper):
            self.image_worker = Worker(
                self.array_to_image, i, as_new_image, new_image_name
            )
        elif isinstance(i, uuid.UUID):
            # !TODO: If memory allows, we should save current canvas by using a stackwidget,
            # such that new canvas just goes ontop of old, and if switch back, just need to go back in stack,
            # which is essentially like cached performance
            self.set_uuid(i)
            item = self.storage.get_data(str(i))
            assert item is not None, "UUID not found in storage"
            image_data = item.get("data", None)
            assert image_data is not None, "UUID not found in storage"
            self.image_worker = Worker(
                self._replace_canvas,
                image_data,
                as_new_image,
                new_image_name,
                target_channel,
            )
        elif isinstance(i, dict):
            self.image_worker = Worker(
                self._replace_canvas,
                i,
                as_new_image,
                new_image_name,
                target_channel,
            )

        self.image_worker.signal.connect(self.set_pixmap)
        self.image_worker.error.connect(self.on_error)
        self.image_worker.finished.connect(self.image_worker.quit)
        self.image_worker.start()

    def array_to_image(self, img: ImageWrapper, as_new_image, image_name=None):
        channel_name = "Channel 1"
        subsample_for_emit = False
        emit_data = {}
        self._store_channel_data(channel_name, img.data)
        self.np_channels[channel_name] = img
        self.reset_np_channels[channel_name] = img.copy()
        self.image_wrapper = img
        display_image = self._prepare_display_image(img.data)
        emit_data[channel_name] = display_image
        if as_new_image:
            assert image_name is not None, "Image name must be provided for new image"
            self._emit_multichannel_data(emit_data, subsample_for_emit)
            self._add_to_manager(image_name)
        return display_image

    @pyqtSlot(object)
    def set_pixmap(self, image: np.ndarray):
        """handles operation after the file is loaded into the canvas"""
        if image is not None and image.dtype != np.uint8:
            image = scale_adjust(image)
        # self.set_uuid(str(uuid.uuid4()))
        qimage = numpy_to_qimage(image)
        pixmap = QPixmap(qimage)
        self.reset_pixmap = pixmap
        self.reset_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap = pixmap
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.new_image_added.emit(
            self.pixmap_item
        )  # emit uint16, change to uint8 in canvas_ui

    def reset_image(self):
        """resets the image to original state"""
        if self.pixmap_item:

            self.np_channels = copy.deepcopy(self.reset_np_channels)
            self.image_signal.emit(self.np_channels, False)

            channel_num = f"Channel {self.current_channel + 1}"
            self.image_wrapper = self.np_channels.get(
                channel_num, ImageWrapper(np.array([]), "")
            )

            self.image_cache.clear()
            self.update_image("gray")

    def rotate_image_task(self, channels: dict, angle):

        for channel_num, wrapper in channels.items():
            try:
                arr = wrapper.data
                if not arr.data.contiguous:
                    arr = np.ascontiguousarray(arr, dtype="uint16")
            except Exception as e:
                self.on_error(f"Error processing {channel_num}: {e}")
                raise e

            # Rotate image with padding and center correction
            h, w = arr.shape
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)

            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])

            updated_w = int((h * sin) + (w * cos))
            updated_h = int((h * cos) + (w * sin))
            max_w = self.reset_np_channels[channel_num].data.shape[1]
            max_h = self.reset_np_channels[channel_num].data.shape[0]
            max_side = max(max_w, max_h)
            max_side *= np.sqrt(2)  # ensure it fits after rotation
            max_side = int(max_side)
            updated_w = min(updated_w, max_side)
            updated_h = min(updated_h, max_side)

            rotation_matrix[0, 2] += (updated_w / 2) - center[0]
            rotation_matrix[1, 2] += (updated_h / 2) - center[1]

            rotated_arr = cv2.warpAffine(arr, rotation_matrix, (updated_w, updated_h))
            self.np_channels[channel_num].data = rotated_arr

        return True

    def rotate_image(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            print("Error: Please enter a valid number.")  # use QMessageBox for GUI
            return

        if self.pixmap and angle is not None:
            self.rotation_worker = Worker(
                self.rotate_image_task, self.np_channels, angle
            )
            self.rotation_worker.signal.connect(self.on_rotation_completed)
            self.rotation_worker.error.connect(self.on_error)
            self.rotation_worker.finished.connect(self.rotation_worker.quit)
            self.rotation_worker.finished.connect(self.rotation_worker.deleteLater)
            self.rotation_worker.start()

    @pyqtSlot(object)
    def on_rotation_completed(self, success):
        if success:
            print("completing rotation")
            for channel_name, wrapper in self.np_channels.items():
                if "Channel" in channel_name:
                    self.storage.update_data(self.uuid, channel_name, wrapper.data)
            self.image_wrapper = self.np_channels.get(
                f"Channel {self.current_channel + 1}", ImageWrapper(np.array([]), "")
            )
            self.image_cache.clear()  # Clear cache to force redraw
            self.to_pixmap(self.image_wrapper.data)
        else:
            raise RuntimeError("Rotation failed")

    @pyqtSlot(str)
    def on_error(self, error_message):
        print(f"Error: {error_message}")

    def update_channels(
        self, channels: dict[str, ImageWrapper], clear: bool
    ) -> None:  # cropsignal will update this
        self.np_channels = channels  # replace channels with new, cropped/rotated, etc
        self.image_signal.emit(self.np_channels, clear)

    def update_current_image(self, data_dict):
        self.image = data_dict[f"Channel {self.current_channel + 1}"].data

    def auto_contrast(self, lower=0.1, upper=0.9):
        if self.is_layered:
            channel_num = f"Channel {self.current_channel + 1}"
            channel = scale_adjust(self.np_channels[channel_num].data)
        else:
            channel = scale_adjust(self.image_wrapper.data)

        flat_channel = channel.flatten()

        hist, _ = np.histogram(flat_channel, bins=256, range=(0, 255))
        total_pixels = flat_channel.size
        cumulative_hist = np.cumsum(hist) / total_pixels
        new_min = np.argmax(cumulative_hist > lower)
        new_max = np.argmax(cumulative_hist > upper)

        self.update_contrast((new_min, new_max))

    def apply_contrast(self, new_min, new_max):
        qimage = self.pixmap.toImage()  # get current image
        image = qimage_to_numpy(qimage)  # returns uint8
        lut = create_lut(new_min, new_max)
        return cv2.LUT(image, lut)

    def blur_layer(self, blur_percentage: float, confirm=False):
        """start gaussian blur in a separate thread"""
        self.blur_worker = Worker(self.blur_layer_task, blur_percentage, confirm)
        # self.blur_worker.signal.connect() # result is rotated_channels
        self.blur_worker.error.connect(self.on_error)
        self.blur_worker.finished.connect(self.blur_worker.quit)
        self.blur_worker.finished.connect(self.blur_worker.deleteLater)
        self.blur_worker.start()

    def blur_layer_task(self, blur_percentage: float, confirm=False):
        """
        Applies Gaussian blur chosen of the image stack and subtracts
        the specified percentage of the blurred image from the original.
        """

        self.image_cache.clear()
        self._blur_layer = f"Channel {self.current_channel+ 1}"
        if not confirm:

            # blur_percentage = self._blur_percentage
            layer_to_blur = (self.np_channels[self._blur_layer].data).copy()
            blurred_mask = cv2.GaussianBlur(layer_to_blur, (101, 101), 0)
            blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
            self.corrected_layer = cv2.subtract(layer_to_blur, blurred_mask_adjusted)
            self.corrected_layer = np.clip(self.corrected_layer, 0, 65535).astype(
                np.uint16
            )
            # cmap = self.np_channels[self._blur_layer].cmap
            self.to_pixmap(self.corrected_layer)
            print("blurring")
            # self.update_image(cmap)

        else:
            print("Error from gaussian blur")

        if (
            confirm
            and hasattr(self, "corrected_layer")
            and (self.np_channels.get(self._blur_layer) is not None)
        ):
            self.np_channels[self._blur_layer].data = (
                self.corrected_layer
            )  # Replace with the corrected version
            cmap = self.np_channels[self._blur_layer].cmap
            self.update_image(cmap)
            self.image_signal.emit(self.np_channels, False)
            self.update_progress.emit(100, f"Replaced {self._blur_layer}")

    def crop(self, image_rect):
        """Safely crop current image using image_rect and save to disk"""

        print("Starting crop...")

        left = max(0, image_rect.left())
        top = max(0, image_rect.top())
        right = min(self.image_wrapper.data.shape[1], image_rect.right())
        bottom = min(self.image_wrapper.data.shape[0], image_rect.bottom())
        cropped_array = self.image_wrapper.data[
            top : bottom + 1, left : right + 1
        ]  # this is the current image. if layered then its the current channel

        cropped_array_copy = cropped_array.copy()

        contrast = (self.image_wrapper.contrast_min, self.image_wrapper.contrast_max)

        self.crop_dialog = ImageDialog(
            self, cropped_array_copy, contrast, self.image_wrapper.cmap
        )
        self.crop_dialog.exec()
        item = self.storage.get_data(self.uuid)
        assert item is not None, "UUID not found in storage while cropping"
        image_name = item["name"]
        name = f"cropped_{image_name}"

        if self.crop_dialog.confirm_crop:
            channels = {}
            for channel_name, wrapper in self.np_channels.items():
                arr = wrapper.data
                cropped_array = arr[top : bottom + 1, left : right + 1].copy()
                wrapper_copy = ImageWrapper(
                    cropped_array, name=channel_name, cmap=wrapper.cmap
                )
                channels[channel_name] = wrapper_copy

            self.crop_worker = Worker(self.add_to_canvas, channels, True, name)
            self.crop_worker.finished.connect(self.crop_worker.quit)
            self.crop_worker.finished.connect(self.crop_worker.deleteLater)
            self.crop_worker.start()
        else:
            self.crop_signal.emit(False)
        if right <= left or bottom <= top:
            print("❌ Invalid crop region: empty or out-of-bounds")

    def flip_horizontal(self):
        """Flip the image horizontally"""
        for channel_name, wrapper in self.np_channels.items():
            if "Channel" in channel_name:
                wrapper.data = cv2.flip(wrapper.data, 1)
                self.storage.update_data(self.uuid, channel_name, wrapper.data)
        self.image_wrapper = self.np_channels.get(
            f"Channel {self.current_channel + 1}", ImageWrapper(np.array([]), "")
        )
        self.image_cache.clear()  # Clear cache to force redraw
        self.to_pixmap(self.image_wrapper.data)
        print("Image flipped horizontally")

    def flip_vertical(self):
        """Flip the image vertically"""

        for channel_name, wrapper in self.np_channels.items():
            if "Channel" in channel_name:
                wrapper.data = cv2.flip(wrapper.data, 0)
                self.storage.update_data(self.uuid, channel_name, wrapper.data)
        self.image_wrapper = self.np_channels.get(
            f"Channel {self.current_channel + 1}", ImageWrapper(np.array([]), "")
        )
        self.image_cache.clear()  # Clear cache to force redraw
        self.to_pixmap(self.image_wrapper.data)
        print("Image flipped vertically")


class MetaData(QWidget):
    """Class to handle metadata of images"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.row_count = 8
        self.column_count = 1
        self.table = QTableWidget()
        self.table.setRowCount(8)
        self.table.setColumnCount(1)

        self.headers = [
            "Name",
            "URI",
            "Pixel Type",
            "Width",
            "Height",
            "Dimension (CZT)",
            "PhysicalSizeX",
            "PhysicalSizeY",
        ]

        self.table.setVerticalHeaderLabels(self.headers)

        self.table.setHorizontalHeaderLabels(["Value"])
        self.table.setColumnWidth(0, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def set_item(self, row, value):
        item = QTableWidgetItem(value)
        self.table.setItem(row, 0, item)

    def populate_table(self, metadata: dict):
        self.table.setRowCount(len(self.headers))

        for row, key in enumerate(self.headers):
            value = metadata.get(key, "Unknown")
            value_item = QTableWidgetItem(str(value))

            # key_item.setFlags(key_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, value_item)

    def parse_metadata(self, filename):
        file_name = os.path.basename(filename)
        name = os.path.splitext(file_name)[0]
        metadata = {}

        with tiff.TiffFile(filename) as tif:
            raw_meta_data = {}
            page = tif.pages[0]
            if isinstance(page, tiff.TiffFrame):
                page = page.aspage()
            for tag in page.tags.values():
                raw_meta_data[tag.name] = tag.value

        try:
            desc = raw_meta_data["ImageDescription"]
            root = ET.fromstring(desc)
            namespace_uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")]
            ns = {"ome": namespace_uri}
            pixels = root.find(".//ome:Pixels", namespaces=ns)

            if pixels is not None:
                metadata = {
                    "Name": name,
                    "URI": filename,
                    "Width": pixels.attrib.get("SizeX"),
                    "Height": pixels.attrib.get("SizeY"),
                    "Dimension (CZT)": f"{pixels.attrib.get('SizeC')} x {pixels.attrib.get('SizeZ')} x {pixels.attrib.get('SizeT')}",
                    "Pixel Type": pixels.attrib.get("Type"),
                    "PhysicalSizeX": f"{pixels.attrib.get('PhysicalSizeX')} {pixels.attrib.get('PhysicalSizeXUnit')}",
                    "PhysicalSizeY": f"{pixels.attrib.get('PhysicalSizeY')} {pixels.attrib.get('PhysicalSizeYUnit')}",
                    "DimensionOrder": pixels.attrib.get("DimensionOrder"),
                }

                for k, v in metadata.items():
                    print(f"{k}: {v}")
            else:
                print("Pixels element not found.")

        except ET.ParseError as e:
            print("Parse error has occurred:", e)

        finally:
            if not metadata:
                metadata["Name"] = name
                metadata["URI"] = filename
                metadata["Width"] = raw_meta_data["ImageWidth"]
                metadata["Height"] = raw_meta_data["ImageLength"]
                metadata["Pixel Type"] = f'uint{raw_meta_data["BitsPerSample"]}'
                metadata["Dimension (CZT)"] = "Unknown"
                metadata["PhysicalSizeX"] = "Unknown"
                metadata["PhysicalSizeY"] = "Unknown"
                metadata["DimensionOrder"] = "Unknown"

        return metadata


class ImageDialog(QDialog):
    """Popup window to confirm the cropped image"""

    def __init__(
        self,
        canvas: ImageGraphicsView,
        cropped_image: np.ndarray,
        contrast: tuple,
        cmap: str,
    ):
        super().__init__()
        self.canvas = canvas
        self.cropped_image = cropped_image
        self.contrast = contrast
        self.cmap = cmap
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cropped Image")

        self._layout = QVBoxLayout()

        self.image_view = QGraphicsView()

        self.image_scene = QGraphicsScene(self)  # Create a QGraphicsScene
        self.image_view.setScene(self.image_scene)  # Set the scene on the view

        if self.cropped_image.dtype != np.uint8:
            self.cropped_image = scale_adjust(self.cropped_image)

        im = self.apply_contrast(self.cropped_image, self.contrast[0], self.contrast[1])
        self.pix = QPixmap(numpy_to_qimage(im))
        self.cropped_pixmap_item = QGraphicsPixmapItem(self.pix)
        scene = self.image_view.scene()
        assert scene is not None
        scene.addItem(self.cropped_pixmap_item)
        self.image_view.setSceneRect(0, 0, self.pix.width(), self.pix.height())
        item_rect = self.cropped_pixmap_item.boundingRect()
        self.image_view.setSceneRect(item_rect)
        self.image_view.fitInView(
            self.cropped_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_view.centerOn(self.cropped_pixmap_item)
        self._layout.addWidget(self.image_view)

        # Add buttons
        self.button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm)
        self.button_layout.addWidget(self.confirm_button)

        self.reject_button = QPushButton("Reject", self)
        self.reject_button.clicked.connect(self.cancel)
        self.button_layout.addWidget(self.reject_button)

        self._layout.addLayout(self.button_layout)

        self.setLayout(self._layout)

    def confirm(self):
        self.confirm_crop = True
        self.canvas.to_pixmap(self.pix)
        self.accept()

    def cancel(self):
        self.confirm_crop = False
        self.reject()

    def apply_contrast(self, image, new_min, new_max):

        lut = self.create_lut(new_min, new_max)
        return LUT(image, lut)

    def create_lut(self, new_min, new_max):

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
