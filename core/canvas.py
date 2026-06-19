# pylint: disable=no-name-in-module, no-member, too-many-lines

import copy
import gc
import logging
import threading
import typing
from collections import deque

# Standard library imports
from queue import Queue
from typing import Dict, Optional, OrderedDict, Union
from uuid import UUID as UUID_Type

import cv2
import numpy as np
import tifffile as tiff

# Third-party imports
from PIL import Image

# PyQt6 imports
from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QPixmap
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
from skimage.color import label2rgb as sk_label2rgb

from core.image_utils import (
    adjustContrast,
    auto_contrast_helper,
    create_lut,
    generate_lut,
    label2rgb,
    numpy_to_qimage,
    scale_adjust,
    to_pixmap,
)

# Local/project imports
from core.metadata_utils import parse_metadata
from core.project_naming import SEGMENTATION_BASE_NAME, is_segmentation_channel
from core.Worker import Worker

logger = logging.getLogger("core.canvas")

if typing.TYPE_CHECKING:
    from controller import Controller


class MemoryEfficientImageCache:
    """Memory-efficient cache with size limits and automatic cleanup."""

    def __init__(self, max_cache_size_mb=500, max_entries_per_channel=5):
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_entries_per_channel = max_entries_per_channel
        self.cache = {}  # {uuid: {channel: OrderedDict of {cache_key: image_data}}}
        self.current_size_bytes = 0

    def get(self, image_uuid, channel, cache_key):
        """Get cached image if available."""
        if image_uuid not in self.cache or channel not in self.cache[image_uuid]:
            return None

        channel_cache = self.cache[image_uuid][channel]
        if cache_key in channel_cache:
            # Move to end (most recently used)
            image_data = channel_cache[cache_key]
            del channel_cache[cache_key]
            channel_cache[cache_key] = image_data
            return image_data
        return None

    def put(self, image_uuid, channel, cache_key, image_data):
        """Cache image data with memory management."""
        if image_uuid not in self.cache:
            self.cache[image_uuid] = {}
        if channel not in self.cache[image_uuid]:
            self.cache[image_uuid][channel] = OrderedDict()

        channel_cache = self.cache[image_uuid][channel]
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
                # Try to find another channel in this uuid or other uuids to
                # evict from
                evicted = False
                for other_channel_cache in self.cache[image_uuid].values():
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

    def clear_channel(self, image_uuid, channel):
        """Clear cache for specific channel of specific uuid."""
        if image_uuid in self.cache and channel in self.cache[image_uuid]:
            channel_cache = self.cache[image_uuid][channel]
            for image_data in channel_cache.values():
                self.current_size_bytes -= image_data.nbytes
            channel_cache.clear()
            gc.collect()

    def clear_uuid(self, image_uuid):
        """Clear all cache for specific uuid."""
        if image_uuid in self.cache:
            for channel_cache in self.cache[image_uuid].values():
                for image_data in channel_cache.values():
                    self.current_size_bytes -= image_data.nbytes
                channel_cache.clear()
            del self.cache[image_uuid]
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


class _ImageStorageSignals(QObject):
    data_changed = pyqtSignal(str, str)  # (image_id, channel)


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
        self._signals = _ImageStorageSignals()

    @property
    def data_changed(self):
        return self._signals.data_changed

    def __len__(self):
        with self._data_lock:
            return len(self.image_list)

    def __getitem__(self, image_id):
        return self.get_data(str(image_id))

    def __setitem__(self, image_id, data):
        self.add_data(str(image_id), data)

    def get_reference_image(self):
        with self._data_lock:
            ref_uuid = self.image_list[str("reference_uuid")]
            assert ref_uuid is not None
            data = self.image_list.get(str(ref_uuid["value"]))
            assert data is not None and "data" in data.keys(), "ref image is none"

            return data["data"]

    def get_canvas_image(self):
        with self._data_lock:
            canvas_uuid = self.image_list[str("canvas_uuid")]
            assert canvas_uuid is not None
            data = self.image_list.get(str(canvas_uuid["value"]))
            assert data is not None and "data" in data.keys(), "canvas image is none"
            return data["data"]

    def get_alignment_ref_image(self):
        with self._data_lock:
            entry = self.image_list["alignment_ref_uuid"]
            data = self.image_list.get(str(entry["value"]))
            assert data is not None and "data" in data, "alignment ref image is none"
            return data["data"]

    def get_alignment_moving_image(self):
        with self._data_lock:
            entry = self.image_list["alignment_moving_uuid"]
            data = self.image_list.get(str(entry["value"]))
            assert data is not None and "data" in data, "alignment moving image is none"
            return data["data"]

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

    def update_data(self, image_id, channel="Channel 1", new_wrapper=None):
        with self._data_lock:
            image_id = str(image_id)
            if image_id in self.image_list:
                self.image_list[image_id]["data"][channel] = new_wrapper
            else:
                raise ValueError(f"UUID: {image_id} doesnt exist in storage")
        self.data_changed.emit(str(image_id), channel)

    def clear_data(self):
        with self._data_lock:
            self.image_list = {}
            gc.collect()

    def update_name(self, image_uuid, new_name):
        with self._data_lock:
            image_id = str(image_uuid)
            if image_id in self.image_list:
                self.image_list[str(image_id)]["name"] = new_name
            else:
                raise ValueError(f"UUID: {image_id} doesnt exist in storage")


# Could maybe use collections.namedtuple to represent ImageWrapper
class ImageWrapper:
    def __init__(self, data, name="", cmap="gray", copy_data=True):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array.")

        self.name = name
        self.cmap = cmap
        self.data = data.copy() if copy_data else data
        self.contrast_min = 0
        self.contrast_max = 255
        self.is_virtual_segmentation = False
        # Keyed by render_size (pixels). Populated lazily by create_thumbnail;
        # intentionally NOT copied so that a new wrapper always gets a fresh cache.
        self._thumb_cache: dict[int, np.ndarray] = {}

    def copy(self, copy_data=True):
        new_obj = object.__new__(ImageWrapper)
        new_obj.name = self.name
        new_obj.cmap = self.cmap
        new_obj.data = self.data.copy() if copy_data else self.data
        new_obj.contrast_min = self.contrast_min
        new_obj.contrast_max = self.contrast_max
        new_obj.is_virtual_segmentation = self.is_virtual_segmentation
        new_obj._thumb_cache = {}
        return new_obj

    def __copy__(self):
        return self.copy(copy_data=True)

    def __deepcopy__(self, memo):
        new_obj = object.__new__(ImageWrapper)
        new_obj.name = self.name
        new_obj.cmap = self.cmap
        new_obj.data = self.data.copy()
        new_obj.contrast_min = self.contrast_min
        new_obj.contrast_max = self.contrast_max
        new_obj.is_virtual_segmentation = self.is_virtual_segmentation
        new_obj._thumb_cache = {}
        memo[id(self)] = new_obj
        return new_obj

    def get_uint8_data(self):
        if self.data.dtype == np.uint8:
            return self.data
        arr = scale_adjust(self.data)
        assert arr.dtype == np.uint8
        return arr

    def get_uint16_data(self):
        if self.data.dtype == np.uint16:
            return self.data
        arr = to_uint16(self.data)
        assert arr.dtype == np.uint16
        return arr

    def __repr__(self):
        return f"ImageWrapper(name={self.name}, shape={self.data.shape}, dtype={self.data.dtype}, cmap={self.cmap}, contrast=({self.contrast_min}, {self.contrast_max}))"

    def __str__(self):
        return self.__repr__()

    def __bool__(self):
        return bool(self.data.any())


# Be able to reset to first version, contrast, crop
# use object for signal parameter if can be None
class BaseGraphicsView(QWidget):
    image_signal = pyqtSignal(dict, bool)
    update_progress = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)
    fill_metadata = pyqtSignal(dict)
    update_manager = pyqtSignal(dict, str, object)
    add_image_to_storage = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(QSize(300, 300))
        self.reset_pixmap = None
        self.reset_pixmap_item = None
        self._worker_queue = []
        self._current_worker = None

        self.working_channels: dict[str, ImageWrapper] = {}
        self.display_channels: dict[str, QPixmap] = {}
        self.reset_working_channels = {}
        self.current_channel = 0
        self.lut_cache = {}
        self.image_count = 0
        self.storage = ImageStorage()
        self.image_wrapper = ImageWrapper(
            np.array([]), ""
        )  # Added for single image compatibility
        self.uuid = None
        self.num_channels = 0
        self.file_queue = deque()
        self.queue_lock = threading.Lock()
        self.caching_queue = Queue()
        self.memory_cache = MemoryEfficientImageCache()

        self._caching_worker = None

    def set_uuid(self, uuid):
        self.uuid = uuid

    def _process_next_worker(self):
        """Processes the next worker in the queue if no worker is currently running."""
        if self._current_worker is not None and self._current_worker.isRunning():
            return
        if not self._worker_queue:
            self._current_worker = None
            return
        self._current_worker = self._worker_queue.pop(0)
        self._current_worker.start()

    @property
    def is_layered(self):
        return len(self.working_channels.items()) > 1

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
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
                return True
        if event:
            event.ignore()
        return False

    def _read_tiff_pages(self, file_path):
        """Generator that yields pages one at a time to avoid memory overload."""
        with tiff.TiffFile(file_path) as tif:
            total_pages = len(tif.pages)
            valid_page_count = 0
            self.num_channels = total_pages
            for _, page in enumerate(tif.pages):
                try:
                    image = page.asarray()
                    # Check if the image is blank
                    if np.all(image == image.flat[0]):
                        continue

                    valid_page_count += 1

                    yield image, valid_page_count

                except Exception as e:
                    logger.warning("Skipping corrupted TIFF page %d: %s", _, e)
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

    def _process_new_file(
        self,
        file_name: str,
        adjust_contrast=False,
        subsample_for_emit=False,
        max_display_size=1024,
    ) -> np.ndarray:
        with self.queue_lock:
            self.file_queue.append(file_name)
        if file_name.endswith((".tiff", ".tif")):
            return self._process_tiff_image(
                file_name, adjust_contrast, subsample_for_emit, max_display_size
            )
        else:
            return self._process_single_image(file_name, subsample_for_emit)

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
        metadata = parse_metadata(file_name)
        self.fill_metadata.emit(metadata)

        emit_data = {}
        channel_one_image = np.array(0)
        working_channels = {}
        # Process pages one at a time using generator
        with self.queue_lock:
            for image, channel_num in self._read_tiff_pages(file_name):
                channel_name = f"Channel {channel_num}"
                image_adjusted = self._apply_contrast_adjustment(image, adjust_contrast)
                working_channels[channel_name] = ImageWrapper(
                    image_adjusted, channel_name, copy_data=False
                )
                self._update_progress(channel_num, self.num_channels)
                # print("File queue:", self.file_queue, "Current file:", file_name)
            if self.file_queue and self.file_queue[-1] == file_name:
                # print("here 3")
                for channel_name, image_adjusted in working_channels.items():
                    display_image = self._prepare_display_image(
                        image_adjusted.data, subsample_for_emit, max_display_size
                    )
                    emit_data[channel_name] = display_image
                    if channel_name == "Channel 1":
                        channel_one_image = display_image
                self.working_channels = working_channels
                self._update_number_of_channels(emit_data, subsample_for_emit)
                self.reset_working_channels = {name: w.copy(copy_data=False) for name, w in working_channels.items()}

                self._schedule_caching_task(self.working_channels, self.uuid)

                self.file_queue.clear()

            # # Store first channel for return value
            # if channel_num == 1:
            #     channel_one_image = display_image
        self.image_wrapper.data = channel_one_image
        self.image_wrapper.cmap = "gray"
        metadata_text = metadata_widget.metadata_tooltip(metadata)
        self._add_to_manager(file_name, working_channels, metadata_text)
        return channel_one_image

    def _process_single_image(
        self, file_name: str, subsample_for_emit: bool
    ) -> np.ndarray:
        """Process single channel images."""
        # print("Processing single image")
        emit_data = {}
        try:
            channel_one_image = np.array(Image.open(file_name))
        except Exception as e:
            logger.error(
                "Failed to open image file '%s': %s", file_name, e, exc_info=True
            )
            raise
        channel_name = "Channel 1"
        channel_one_wrapper = ImageWrapper(channel_one_image, channel_name, copy_data=False)
        working_channel = {channel_name: channel_one_wrapper}
        display_image = scale_adjust(channel_one_image)
        with self.queue_lock:
            if self.file_queue and self.file_queue[-1] == file_name:
                self.working_channels = working_channel
                self.reset_working_channels = {k: v.copy(copy_data=False) for k, v in working_channel.items()}
                emit_data[channel_name] = display_image
                self._update_number_of_channels(emit_data, subsample_for_emit)
                self.memory_cache.put(
                    self.uuid,
                    channel_name,
                    contrast_key_from_wrapper(channel_one_wrapper),
                    display_image,
                )
                self.file_queue.clear()
        self._store_channel_data(channel_name, working_channel[channel_name])
        self._add_to_manager(file_name, working_channel)
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
            ret = self._replace_canvas_multichannel(
                data, target_channel, subsample_for_emit, max_display_size
            )
            if as_new_image:
                assert new_image_name is not None, (
                    "Image name must be provided for new image"
                )
                self._add_to_manager(new_image_name, self.working_channels)

            return ret
        elif isinstance(data, np.ndarray):
            # shouldn't be used
            logger.debug("Using single channel canvas replacement")
            return self._replace_canvas_single(
                data, subsample_for_emit, max_display_size
            )
        else:
            raise ValueError(
                "Data must be either numpy array or dictionary of numpy arrays"
            )

    def _replace_canvas_multichannel(
        self,
        channels_data: Dict[str, ImageWrapper],
        target_channel: str = "Channel 1",
        subsample_for_emit: bool = False,
        max_display_size: int = 1024,
    ) -> np.ndarray:
        """Replace canvas with multichannel image data - target channel first, others in background."""
        self._prepare_channels_for_new_image()

        # Process target channel first for immediate display
        if target_channel not in channels_data:
            raise ValueError(f"{target_channel} was not found in the data")

        target_image_wrapper = channels_data[target_channel]
        self.current_channel = int(target_channel.split(" ")[-1]) - 1

        for channel_name, channel_data in channels_data.items():
            if channel_name != target_channel:
                self._store_channel_data(
                    channel_name, channel_data, replace_image_wrapper=False
                )
        self._store_channel_data(target_channel, target_image_wrapper)
        # display_channel_data = self._prepare_display_image(
        #     target_image_wrapper.data, subsample_for_emit, max_display_size
        # )

        # Emit target channel immediately for display
        # emit_data = {target_channel: target_image_wrapper.data}
        # self._update_number_of_channels(emit_data, subsample_for_emit)

        # Process remaining channels in background if there are any
        remaining_channels = {
            k: v for k, v in self.working_channels.items() if k != target_channel
        }

        if remaining_channels:
            self._schedule_caching_task(remaining_channels, self.uuid)

        if hasattr(self, "_restore_stardist_state_from_channels"):
            self._restore_stardist_state_from_channels()

        # self._clear_caches()
        return target_image_wrapper.data

    def _cache_remaining_channels(
        self, remaining_channels: Dict[str, ImageWrapper], uuid
    ) -> Dict[str, np.ndarray]:
        processed_channels = {}
        for channel_name, image_wrapper in remaining_channels.items():
            try:
                cache_key = contrast_key_from_wrapper(image_wrapper)
                if self.memory_cache.get(uuid, channel_name, cache_key) is not None:
                    continue
                contrasted = self.apply_contrast(
                    image_wrapper.contrast_min,
                    image_wrapper.contrast_max,
                    image_wrapper.data,
                )
                self.memory_cache.put(uuid, channel_name, cache_key, contrasted)
            except Exception as e:
                logger.error(
                    f"Error processing {channel_name} in background: {e}", exc_info=True
                )
                continue

        return processed_channels

    def _schedule_caching_task(self, channels, uuid):
        """Add a caching task to the queue and start a worker if not running."""
        self.caching_queue.put((channels, uuid))
        if not (self._caching_worker is not None and self._caching_worker.isRunning()):
            self._caching_worker = Worker(self._process_caching_queue)
            self._caching_worker.finished.connect(self._on_caching_worker_finished)
            self._caching_worker.error.connect(
                lambda e: logger.error("Caching worker error: %s", e)
            )
            self._caching_worker.start()

    def _process_caching_queue(self):
        """Process one task from the caching queue."""
        if not self.caching_queue.empty():
            try:
                channels, uuid = self.caching_queue.get_nowait()
                self._cache_remaining_channels(channels, uuid)
                self.caching_queue.task_done()
            except Exception as e:
                logger.error("Error processing caching queue: %s", e, exc_info=True)

    def _on_caching_worker_finished(self):
        """Called when a caching worker finishes. Starts a new one if queue is not empty."""
        if not self.caching_queue.empty():
            self._caching_worker = Worker(self._process_caching_queue)
            self._caching_worker.finished.connect(self._on_caching_worker_finished)
            self._caching_worker.error.connect(
                lambda e: logger.error("Caching worker error: %s", e)
            )
            self._caching_worker.start()

    def _prepare_channels_for_new_image(self):
        self.working_channels = {}
        self.reset_working_channels = {}
        self.display_channels = {}
        self.current_channel = 0

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
        img_data = scale_adjust(image_data)

        # Store full resolution
        self.image_wrapper = ImageWrapper(img_data, "Channel 1", copy_data=False)
        self.working_channels[channel_name] = self.image_wrapper

        # Handle display emission
        display_img = self._handle_single_image_display(
            img_data, subsample_for_emit, max_display_size
        )

        self._clear_caches()

        # Update manager without filename
        # print("Emitting to update manager")
        # self.update_manager.emit(self.np_channels, "replaced_single")
        self.image_count += 1
        # print(f"Canvas replaced with single image, shape: {image_data.shape}")
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
        self, channel_name: str, image_wrapper: ImageWrapper, replace_image_wrapper=True
    ) -> None:
        """Store channel data in full resolution containers."""
        self.working_channels[channel_name] = image_wrapper.copy(copy_data=False)
        self.reset_working_channels[channel_name] = image_wrapper.copy(copy_data=False)
        if replace_image_wrapper:
            logger.debug("stored image_wrapper")
            self.image_wrapper = self.working_channels[channel_name]
            logger.debug(self.image_wrapper)

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
            # print(f"  Original: {image_data.shape}, Subsampled: {subsampled.shape}")
            image_data = subsampled
        image_data = scale_adjust(image_data)
        return image_data

    def _handle_single_image_display(
        self, image_data: np.ndarray, subsample_for_emit: bool, max_display_size: int
    ) -> np.ndarray:
        """Handle single image display emission."""
        if subsample_for_emit and image_data.size > max_display_size * max_display_size:
            subsampled = self._subsample_for_display(image_data, max_display_size)
            self.image_signal.emit(subsampled, True)

            return subsampled
        else:
            self.image_signal.emit(image_data, True)
            return image_data

    def _update_number_of_channels(
        self, emit_data: Dict[str, np.ndarray], subsample_for_emit: bool
    ) -> None:
        """Notify listeners that number of channels has changed and emit data."""
        logger.debug(self.working_channels.keys())

        self.working_channels = {
            k: self.working_channels[k] for k in sorted(self.working_channels.keys())
        }
        logger.debug(self.working_channels.keys())
        if emit_data:
            if subsample_for_emit:
                # Create wrappers for subsampled data
                display_wrappers = {
                    name: ImageWrapper(data, name, copy_data=False) for name, data in emit_data.items()
                }
                # self.np_channels.update(display_wrappers)
                self.image_signal.emit(display_wrappers, True)
            else:
                # Emit full resolution
                self.image_signal.emit(self.working_channels, True)

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
            logger.debug(
                f"{channel_name} - Full: {full_size_mb:.2f} MB, Display: {display_size_mb:.2f} MB"
            )
        else:
            logger.debug(f"{channel_name} - Size: {full_size_mb:.2f} MB")

    def _update_progress(self, channel_num: int, total_channels) -> None:
        """Update processing progress."""
        progress = 10 + int(channel_num / total_channels * 70)
        self.update_progress.emit(progress, f"Processing Channel {channel_num}")

    def _add_to_manager(self, file_name: str, image_channels, metadata=None) -> None:
        """Finalize processing with cleanup and emissions."""
        self.update_progress.emit(100, "Image Loaded")
        # Shallow copy to prevent race while minimizing memory overhead
        channels_snapshot = {k: v.copy(copy_data=False) for k, v in image_channels.items()}
        self.update_manager.emit(channels_snapshot, file_name, metadata)
        self.image_count += 1

    def _clear_caches(self) -> None:
        """Clear image and LUT caches."""
        pass

    def remove_from_canvas(self, uuid: UUID_Type):
        """Remove a specific channel from the canvas."""
        if str(self.uuid) == str(uuid):
            self.working_channels.clear()
            self.reset_working_channels.clear()
            self._clear_caches()
            self.image_signal.emit({}, True)
            # print(f"Removed channel with UUID {uuid} from canvas.")
            return True
        else:
            return False

    def apply_contrast(self, new_min, new_max, image=None):
        if image is None:
            image = self.image_wrapper.data
        image = scale_adjust(image)
        lut = create_lut(new_min, new_max)
        res = np.clip(cv2.LUT(image, lut), 0, 254, dtype=np.uint8)
        return res


class ReferenceGraphicsView(BaseGraphicsView):
    update_reference = pyqtSignal(QPixmap, int)
    reference_channel_updated = pyqtSignal()

    def dropEvent(self, event: QDropEvent):  # type: ignore
        if self._accept_if_valid(event):
            for url in event.mimeData().urls():  # type: ignore ;_accept_if_valid ensures mimeData is not None
                file_path = url.toLocalFile()
                if file_path:
                    self.add_to_canvas(file_path)

    def add_to_canvas(self, i: str | UUID_Type, target_channel="Channel 1"):
        self.reference_worker = None
        if isinstance(i, UUID_Type):
            self.set_uuid(i)
            self.storage.add_data("reference_channel", {"value": target_channel})
            item = self.storage.get_data(str(i))
            assert item is not None, "UUID not found in storage"
            image_data = item.get("data", None)
            assert image_data is not None, "UUID not found in storage"
            self.reference_worker = Worker(
                self._replace_canvas, image_data, target_channel=target_channel
            )
        else:
            self.reference_worker = Worker(self._process_new_file, i, False)

        self.reference_worker.signal.connect(self.set_pixmap)
        self.reference_worker.finished.connect(self.reference_worker.quit)
        self.reference_worker.finished.connect(self.reference_worker.deleteLater)
        self.reference_worker.finished.connect(self._process_next_worker)

        self._worker_queue.append(self.reference_worker)
        if self._current_worker is None or not self._current_worker.isRunning():
            self._process_next_worker()

    def set_uuid(self, uuid):
        """Set UUID for the reference image."""
        self.uuid = uuid
        self.storage.add_data("reference_uuid", {"value": uuid})

    def on_channel_changed(self, channel_index: int):
        """Update stored reference channel and notify badge when user navigates via arrows."""
        self.current_channel = channel_index - 1  # keep 0-indexed model in sync
        self.storage.add_data(
            "reference_channel", {"value": f"Channel {channel_index}"}
        )
        self.reference_channel_updated.emit()

    def set_pixmap(self, image):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = scale_adjust(image)

        qimage = to_pixmap(image)
        self.update_reference.emit(qimage, self.current_channel + 1)

    def remove_from_canvas(self, uuid: UUID_Type):
        if super().remove_from_canvas(uuid):
            self.update_reference.emit(QPixmap(), 1)
            return True
        return False


##########################################################
class ImageGraphicsView(BaseGraphicsView):
    update_canvas = pyqtSignal(QPixmap)
    save_image = pyqtSignal(QGraphicsPixmapItem)
    change_slider = pyqtSignal(tuple)
    update_cmap = pyqtSignal(str)
    crop_signal = pyqtSignal(bool)
    update_channel = pyqtSignal(int)
    uuid_changed = pyqtSignal(str)
    overlay_build_started = pyqtSignal()
    overlay_build_finished = pyqtSignal()

    def __init__(self, controller: "Controller"):
        super().__init__()
        self.controller = controller
        self.begin_crop = False
        self.crop_cursor = QCursor(Qt.CursorShape.CrossCursor)
        try:
            import psutil
            _avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            _cache_mb = min(int(_avail_mb * 0.20), 1200)
        except Exception:
            _cache_mb = 500
        self.memory_cache = MemoryEfficientImageCache(max_cache_size_mb=_cache_mb, max_entries_per_channel=2)
        self.blur_worker = None
        self.segmentation_labels = None
        self.stardist_overlay_enabled = False
        self.stardist_overlay_alpha = 0.45
        self._stardist_overlay_rgb = None
        self.stardist_source_channel = None
        self.stardist_virtual_channel = None
        self._overlay_worker = None
        self._blur_layer = ""
        self._blur_source_uuid = None
        self.corrected_layer = None
        self.uuid = None

    def set_uuid(self, uuid):
        """Set UUID for the current image (Image Tab)."""
        # print("Setting image UUID:", uuid)
        self.uuid = uuid
        self.storage.add_data("canvas_uuid", {"value": uuid})
        self.uuid_changed.emit(str(uuid))

    def clear_canvas(self):
        self.reset_pixmap = None
        self.reset_pixmap_item = None
        self.working_channels = {}
        self.reset_working_channels = {}
        self.current_channel = 0
        self.lut_cache = {}
        self.image_wrapper = ImageWrapper(np.array([]), "")
        self.blur_worker = None
        self.uuid = None
        self.num_channels = 0
        self.clear_stardist_overlay()
        self.update_canvas.emit(QPixmap())

    def _resolve_stardist_virtual_cmap(self, fallback_cmap: str = "gray") -> str:
        """Choose a non-label cmap for StarDist virtual label channel selection."""
        if (
            self.stardist_source_channel is not None
            and self.stardist_source_channel in self.working_channels
        ):
            source_cmap = self.working_channels[self.stardist_source_channel].cmap
            if source_cmap != "label_image":
                return source_cmap
        if fallback_cmap != "label_image":
            return fallback_cmap
        return "gray"

    def _effective_channel_cmap(
        self, channel_key: str, fallback_cmap: str = "gray"
    ) -> str:
        """Return display cmap for a selected channel, normalizing StarDist virtual labels."""
        wrapper = self.working_channels.get(channel_key)
        if wrapper is None:
            return "gray" if fallback_cmap == "label_image" else fallback_cmap

        if is_segmentation_channel(wrapper) and wrapper.cmap == "label_image":
            return self._resolve_stardist_virtual_cmap(fallback_cmap)

        return wrapper.cmap

    def swap_channel(self, index):
        """Modified swap_channel to wait for background processing if needed."""
        channel_num = f"Channel {index + 1}"
        if channel_num not in self.working_channels:
            logger.debug("Channel %s not available. swap ignored.", channel_num)
            return

        previous_cmap = self.image_wrapper.cmap
        # not 100% so need the len check later
        self.current_channel = index
        self.image_wrapper = self.working_channels[channel_num]
        target_cmap = self._effective_channel_cmap(channel_num, previous_cmap)
        self.update_image(cmap_text=target_cmap)

    def _prepare_channels_for_new_image(self):
        super()._prepare_channels_for_new_image()
        self.clear_stardist_overlay()

    def clear_stardist_overlay(self):
        self.segmentation_labels = None
        self.stardist_overlay_enabled = False
        self._stardist_overlay_rgb = None
        self.stardist_source_channel = None
        self.stardist_virtual_channel = None

    def _restore_stardist_state_from_channels(self):
        self.clear_stardist_overlay()
        for channel_key, wrapper in self.working_channels.items():
            if is_segmentation_channel(wrapper):
                self.stardist_virtual_channel = channel_key
                self.segmentation_labels = wrapper.data.copy()
                break

        if self.segmentation_labels is None:
            return

        current_channel_key = f"Channel {self.current_channel + 1}"
        if current_channel_key != self.stardist_virtual_channel:
            self.stardist_source_channel = current_channel_key
            return

        for channel_key, wrapper in self.working_channels.items():
            if channel_key == self.stardist_virtual_channel:
                continue
            if (
                wrapper.data.ndim == 2
                and wrapper.data.shape == self.segmentation_labels.shape
            ):
                self.stardist_source_channel = channel_key
                return

    def set_stardist_overlay_enabled(self, enabled: bool):
        if not enabled or self.segmentation_labels is None:
            # Disable path: fast, run inline (keep cache so re-enable is instant)
            self.stardist_overlay_enabled = False
            self.update_image()
            return

        if self._stardist_overlay_rgb is not None:
            # Cache already built: fast path
            self.stardist_overlay_enabled = True
            self.update_image()
            return

        # Slow path: build overlay on a background thread
        self._start_overlay_build()

    def _start_overlay_build(self):
        if self._overlay_worker is not None and self._overlay_worker.isRunning():
            return
        self.overlay_build_started.emit()
        self.update_progress.emit(0, "Building segmentation overlay")
        labels = self.segmentation_labels
        self._overlay_worker = Worker(self._build_stardist_overlay, labels)
        self._overlay_worker.signal.connect(self._on_overlay_built)
        self._overlay_worker.error.connect(self._on_overlay_error)
        self._overlay_worker.start()

    @pyqtSlot(object)
    def _on_overlay_built(self, overlay_rgb):
        self._overlay_worker = None
        self._stardist_overlay_rgb = overlay_rgb
        self.stardist_overlay_enabled = True
        self.update_progress.emit(100, "Overlay ready")
        self.overlay_build_finished.emit()
        self.update_image()

    @pyqtSlot(str)
    def _on_overlay_error(self, error_msg):
        self._overlay_worker = None
        logger.error("Overlay build failed: %s", error_msg)
        self.stardist_overlay_enabled = False
        self.update_progress.emit(100, "Overlay failed")
        self.overlay_build_finished.emit()

    def remove_virtual_stardist_channel(self, channel_key: str) -> bool:
        wrapper = self.working_channels.get(channel_key)
        if wrapper is None or not is_segmentation_channel(wrapper):
            return False

        self.working_channels.pop(channel_key, None)
        self.reset_working_channels.pop(channel_key, None)
        if self.uuid is not None:
            self.memory_cache.clear_channel(self.uuid, channel_key)

        self.clear_stardist_overlay()

        if not self.working_channels:
            self.image_wrapper = ImageWrapper(np.array([]), "")
            self.update_canvas.emit(QPixmap())
            return True

        available_channels = sorted(
            self.working_channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
        )
        self.current_channel = int(available_channels[0].replace("Channel ", "")) - 1
        self.image_wrapper = self.working_channels[available_channels[0]]
        self.image_signal.emit(self.working_channels, True)
        self.update_image()
        return True

    def _set_stardist_source_channel(self):
        if not hasattr(self.controller, "model_stardist"):
            return
        source_channel = self.controller.model_stardist.params.get("channel")
        if not isinstance(source_channel, str):
            return
        if source_channel not in self.working_channels:
            return
        self.stardist_source_channel = source_channel

        try:
            self.current_channel = int(source_channel.replace("Channel ", "")) - 1
        except ValueError:
            return
        self.image_wrapper = self.working_channels[source_channel]

    def _next_available_channel_key(self) -> str:
        next_index = 0
        for key in self.working_channels:
            if not key.startswith("Channel "):
                continue
            try:
                idx = int(key.replace("Channel ", ""))
            except ValueError:
                continue
            next_index = max(next_index, idx)
        return f"Channel {next_index + 1}"

    def _upsert_stardist_virtual_channel(
        self, labels: np.ndarray, label_name: str = SEGMENTATION_BASE_NAME
    ):
        if self.stardist_virtual_channel is None:
            for channel_key, wrapper in self.working_channels.items():
                if is_segmentation_channel(wrapper):
                    self.stardist_virtual_channel = channel_key
                    break
            if self.stardist_virtual_channel is None:
                self.stardist_virtual_channel = self._next_available_channel_key()

        channel_key = self.stardist_virtual_channel
        wrapper = ImageWrapper(labels, name=label_name, cmap="label_image")
        wrapper.is_virtual_segmentation = True
        wrapper.contrast_min = 0
        wrapper.contrast_max = 255
        self.working_channels[channel_key] = wrapper
        self.reset_working_channels[channel_key] = wrapper.copy()

        if self.uuid is None:
            return
        item = self.storage.get_data(str(self.uuid))
        if item is None:
            return
        self.storage.update_data(
            str(self.uuid),
            channel_key,
            wrapper.copy(),
        )

    def _build_stardist_overlay(self, labels: np.ndarray) -> np.ndarray:
        label_rgb = sk_label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
        return scale_adjust(label_rgb)

    def _apply_stardist_overlay(self, image_to_display: np.ndarray) -> np.ndarray:
        current_channel_key = f"Channel {self.current_channel + 1}"
        if (
            not self.stardist_overlay_enabled
            or self.segmentation_labels is None
            or image_to_display.size == 0
            or self.image_wrapper.data.ndim != 2
            or self.segmentation_labels.shape != self.image_wrapper.data.shape
            or self.stardist_source_channel is None
            or current_channel_key != self.stardist_source_channel
            or current_channel_key == self.stardist_virtual_channel
        ):
            return image_to_display

        if image_to_display.dtype != np.uint8:
            image_to_display = scale_adjust(image_to_display)

        if image_to_display.ndim == 2:
            base_rgb = cv2.cvtColor(image_to_display, cv2.COLOR_GRAY2RGB)
        elif image_to_display.ndim == 3 and image_to_display.shape[2] >= 3:
            base_rgb = image_to_display[:, :, :3].copy()
        else:
            return image_to_display

        if self._stardist_overlay_rgb is None:
            self._stardist_overlay_rgb = self._build_stardist_overlay(
                self.segmentation_labels
            )

        mask = self.segmentation_labels > 0
        if not np.any(mask):
            return base_rgb

        alpha = float(np.clip(self.stardist_overlay_alpha, 0.0, 1.0))
        blended = base_rgb.astype(np.float32)
        overlay = self._stardist_overlay_rgb.astype(np.float32)
        blended[mask] = (1.0 - alpha) * blended[mask] + alpha * overlay[mask]
        return np.clip(blended, 0, 255).astype(np.uint8)

    def update_contrast_memory_efficient(
        self, values, use_cache=True, is_labeled=False, cache_result=True
    ):
        """Memory-efficient version of update_contrast method."""
        if self.image_wrapper is None:
            self.error_signal.emit("Canvas is empty")
            return np.array([]), ""

        contrast_min, contrast_max = int(values[0]), int(values[1])

        contrast_key = (contrast_min, contrast_max)
        if is_labeled:
            contrast_key = ("labeled", "labeled")
        cmap_key = self.image_wrapper.cmap
        cache_key = contrast_key
        contrasted_image = np.array([])
        # if self.is_layered:
        # print("Processing layered image with memory management")
        channel_num = f"Channel {self.current_channel + 1}"
        if channel_num not in self.working_channels:
            logger.warning("Channel %s missing during contrast update.", channel_num)
            return np.array([]), cmap_key
        self.image_wrapper = self.working_channels[channel_num]
        self.image_wrapper.contrast_min = contrast_min
        self.image_wrapper.contrast_max = contrast_max

        # Check cache first
        cached_image = None
        if use_cache:
            cached_image = self.memory_cache.get(self.uuid, channel_num, cache_key)
        if cached_image is not None:
            assert isinstance(cached_image, np.ndarray), "Cached image must be ndarray"
            logger.debug(f"Using cached image for {channel_num}")
            contrasted_image = cached_image
        else:
            logger.debug(f"Processing new contrast for {channel_num}")
            if is_labeled:
                # ignore contast
                contrasted_image = sk_label2rgb(self.image_wrapper.data)
                cmap_key = "label_image"
            else:
                contrasted_image = self._apply_contrast_memory_efficient(
                    channel_num, cache_key, contrast_min, contrast_max
                )
            if cache_result:
                self.memory_cache.put(
                    self.uuid, channel_num, cache_key, contrasted_image
                )
        self.change_slider.emit(
            (self.image_wrapper.contrast_min, self.image_wrapper.contrast_max)
        )
        return contrasted_image, cmap_key

    def _apply_contrast_memory_efficient(
        self, channel_key, cache_key, contrast_min, contrast_max
    ):
        """Memory-efficient contrast application with caching."""
        try:
            # Apply contrast
            image_to_display = self.apply_contrast(contrast_min, contrast_max)
            # print("image_to_display", image_to_display.dtype, image_to_display.max())

            # Cache the result
            self.memory_cache.put(
                self.uuid, channel_key, cache_key, image_to_display.copy()
            )

            # Display
            return image_to_display

        except MemoryError:
            # print("Memory error - clearing cache and retrying")
            self.memory_cache.clear_all()
            gc.collect()

            # Retry with no caching
            image_to_display = self.apply_contrast(contrast_min, contrast_max)
            return image_to_display

    def change_cmap(self, cmap_text="default", image=None):
        """changes the colormap given a colormap str valid in matplotlib"""

        if self.image_wrapper is None:
            self.error_signal.emit("Canvas is empty")
            return
        if image is None:
            image = self.image_wrapper.data
        if cmap_text == "default":
            cmap_text = self.image_wrapper.cmap
        self.image_wrapper.cmap = cmap_text
        self.update_cmap.emit(cmap_text)

        if cmap_text not in self.lut_cache:
            lut = generate_lut(cmap_text)
            self.lut_cache[cmap_text] = lut  # cache to avoid recalculating LUT
        else:
            lut = self.lut_cache[cmap_text]  # Reuse the cached LaUT

        return np.clip(label2rgb(scale_adjust(image), lut), 0, 254, dtype=np.uint8)

    def update_image(
        self,
        cmap_text="default",
        image=None,
        use_cache=True,
        self_emit=True,
        cache_result=True,
    ):
        """Updates the current image using the current colormap and contrast settings.
        This only changes the display and does not change the underlying data."""
        if self.image_wrapper.data.size == 0:
            logger.warning("No image loaded — update ignored")
            return None

        if cmap_text == "default":
            cmap_text = self.image_wrapper.cmap
        logger.debug(f"Changing cmap to {cmap_text}")
        self.update_cmap.emit(cmap_text)
        assert self.image_wrapper is not None, "Updating empty image wrapper"
        contrast_min, contrast_max = (
            self.image_wrapper.contrast_min,
            self.image_wrapper.contrast_max,
        )  # read contrast settings
        # use image_wrapper data if image is None
        image_to_display = None
        prev_cmap = None
        if cmap_text == "label_image":
            # image = self.image_wrapper.data
            image, prev_cmap = self.update_contrast_memory_efficient(
                (contrast_min, contrast_max),
                use_cache=use_cache,
                is_labeled=True,
                cache_result=cache_result,
            )
        elif image is None:
            image, prev_cmap = self.update_contrast_memory_efficient(
                (contrast_min, contrast_max),
                use_cache=use_cache,
                cache_result=cache_result,
            )
        if image.size == 0:
            return None
        if cmap_text != "label_image" and prev_cmap != "label_image":
            if image.ndim == 3 and image.shape[2] == 3:
                image_to_display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                logger.debug(f"changing cmap to {cmap_text}")
                image_to_display = self.change_cmap(cmap_text, image)
        else:
            logger.debug("using raw image")
            image_to_display = image

        assert image_to_display is not None, "Updating empty image"
        image_to_display = self._apply_stardist_overlay(image_to_display)
        if self_emit:
            self.set_pixmap(image_to_display)

        if self.uuid is not None:
            item = self.storage.get_data(str(self.uuid))
            if item is not None:
                logger.debug(
                    "Updating {} , channel: {}".format(
                        str(self.uuid), self.current_channel
                    )
                )
                channel_num = f"Channel {self.current_channel + 1}"
                stored_wrapper = item.get("data", {}).get(channel_num)
                if stored_wrapper is not None:
                    logger.debug(
                        "Updating stored wrapper contrast and cmap for %s", channel_num
                    )
                    stored_wrapper.contrast_min = self.image_wrapper.contrast_min
                    stored_wrapper.contrast_max = self.image_wrapper.contrast_max
                    stored_wrapper.cmap = self.image_wrapper.cmap

        return image_to_display

    def update_contrast(self, values):
        if self.image_wrapper.data.size == 0:
            logger.warning("No image loaded — contrast change ignored")
            return
        # print("Updating contrast with values:", values)
        self.image_wrapper.contrast_min = int(values[0])
        self.image_wrapper.contrast_max = int(values[1])
        self.update_image()

    def _apply_contrast_and_cache(
        self, channel_num, cache_key, contrast_min, contrast_max
    ):
        return self._apply_contrast_memory_efficient(
            channel_num, cache_key, contrast_min, contrast_max
        )

    def load_segmentation_labels(self, stardist: ImageWrapper, *metadata):
        label_name = SEGMENTATION_BASE_NAME
        if metadata:
            candidate_name = metadata[-1]
            if isinstance(candidate_name, str) and candidate_name.strip():
                label_name = candidate_name
        self.segmentation_labels = stardist.data.copy()
        self.stardist_overlay_enabled = False
        self._stardist_overlay_rgb = None
        self._upsert_stardist_virtual_channel(
            self.segmentation_labels, label_name=label_name
        )
        self._set_stardist_source_channel()
        if self.image_wrapper.cmap == "label_image":
            self.image_wrapper.cmap = "gray"
        self.image_signal.emit(self.working_channels, True)
        self.update_image()

    def add_to_canvas(
        self,
        i: str | ImageWrapper | UUID_Type | dict[str, ImageWrapper],
        as_new_image=True,
        new_image_name=None,
        target_channel="Channel 1",
    ):
        """add a new image if input is a filename, or can choose to only replace the canvas
        if input is an ImageWrapper or dict of ImageWrapper"""
        # if hasattr(self, "memory_cache"):
        # self.memory_cache.clear_all()
        # str is filepath
        self.image_worker = None
        if isinstance(i, str):
            assert self.storage.get_data(i) is None, "Convert str to UUID instance"
            if not as_new_image:
                raise ValueError(
                    "Cannot replace canvas with a filename, UUID replace not supported yet."
                )
            self.image_worker = Worker(self._process_new_file, i)
        elif isinstance(i, ImageWrapper):
            self.image_worker = Worker(
                self.array_to_image, i, as_new_image, new_image_name
            )
        elif isinstance(i, UUID_Type):
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

        # self.image_worker.signal.connect(self.set_pixmap)
        self.image_worker.error.connect(self.on_error)
        self.image_worker.finished.connect(self.update_image)
        self.image_worker.finished.connect(self.image_worker.quit)
        self.image_worker.finished.connect(self.image_worker.deleteLater)
        self.image_worker.finished.connect(self._process_next_worker)

        self._worker_queue.append(self.image_worker)
        if self._current_worker is None or not self._current_worker.isRunning():
            self._process_next_worker()

    def array_to_image(self, img: ImageWrapper, as_new_image, image_name=None):
        channel_name = "Channel 1"
        subsample_for_emit = False
        emit_data = {}
        self.working_channels = {}
        self.reset_working_channels = {}
        self._store_channel_data(channel_name, img)
        # display_image = self._prepare_display_image(img.data)
        display_image = self.update_image(self_emit=False)
        emit_data[channel_name] = display_image
        if as_new_image:
            assert image_name is not None, "Image name must be provided for new image"
            self._update_number_of_channels(emit_data, subsample_for_emit)
            self._add_to_manager(image_name, self.working_channels)

        return display_image

    @pyqtSlot(object)
    def set_pixmap(self, image: np.ndarray):
        """handles operation after the file is loaded into the canvas"""
        if len(image) == 0:
            return
        if image is not None and image.dtype != np.uint8:
            image = scale_adjust(image)
        # self.set_uuid(str(UUID_Type4()))
        # print(image.dtype, image.shape, image.max())
        qimage = numpy_to_qimage(image)
        pixmap = QPixmap(qimage)
        # print("setting pixmap")
        self.update_channel.emit(self.current_channel)
        # emit uint16, change to uint8 in canvas_ui
        self.update_canvas.emit(pixmap)

    def reset_image(self):
        """resets the image to original state"""
        if self.image_wrapper.data.size == 0:
            logger.warning("No image loaded — reset ignored")
            return
        if len(self.reset_working_channels):
            self.working_channels = copy.deepcopy(self.reset_working_channels)
            self.image_signal.emit(self.working_channels, False)
            logger.debug(self.uuid)

            channel_num = f"Channel {self.current_channel + 1}"
            for channel_name, wrapper in self.working_channels.items():
                if "Channel" in channel_name:
                    self.storage.update_data(
                        str(self.uuid),
                        channel_name,
                        wrapper,
                    )
            self.image_wrapper = self.working_channels.get(
                channel_num, ImageWrapper(np.array([]), "")
            )

            self._clear_caches()
            self.update_image()

    def _replace_canvas_multichannel(
        self,
        channels_data: Dict[str, ImageWrapper],
        target_channel: str = "Channel 1",
        subsample_for_emit: bool = False,
        max_display_size: int = 1024,
    ) -> np.ndarray:
        """Replace canvas with multichannel image data - target channel first, others in background."""
        self._prepare_channels_for_new_image()

        # Initialize thread lock for background processing

        # Process target channel first for immediate display
        if target_channel not in channels_data:
            raise ValueError(f"{target_channel} was not found in the data")

        previous_cmap = self.image_wrapper.cmap
        target_image_wrapper = channels_data[target_channel]

        # Process target channel immediately
        # fill channels with dummy data:
        for channel_name, channel_data in channels_data.items():
            if channel_name != target_channel:
                self._store_channel_data(
                    channel_name, channel_data, replace_image_wrapper=False
                )
        self._store_channel_data(target_channel, target_image_wrapper)
        self.current_channel = int(target_channel.split(" ")[-1]) - 1

        display_channel_data = target_image_wrapper.data
        self.update_cmap.emit(
            self._effective_channel_cmap(target_channel, previous_cmap)
        )
        self.change_slider.emit(
            (target_image_wrapper.contrast_min, target_image_wrapper.contrast_max)
        )

        # Emit target channel immediately for display
        emit_data = {target_channel: display_channel_data}
        self._update_number_of_channels(emit_data, subsample_for_emit)
        logger.debug(f"{self.current_channel} current channel")

        # Process remaining channels in background if there are any
        remaining_channels = {
            k: v for k, v in self.working_channels.items() if k != target_channel
        }

        if remaining_channels:
            self._schedule_caching_task(remaining_channels, self.uuid)

        if hasattr(self, "_restore_stardist_state_from_channels"):
            self._restore_stardist_state_from_channels()

        self.image_count += 1

        return display_channel_data

    def rotate_image_task(
        self, current_uuid, channels: dict, reset_working_channels: dict, angle
    ):
        result = {}

        for channel_num, wrapper in channels.items():
            try:
                arr = wrapper.data
                arr = np.ascontiguousarray(arr, dtype="uint16").copy()
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
            max_w = reset_working_channels[channel_num].data.shape[1]
            max_h = reset_working_channels[channel_num].data.shape[0]
            max_side = max(max_w, max_h)
            max_side *= np.sqrt(2)  # ensure it fits after rotation
            max_side = int(max_side)
            updated_w = min(updated_w, max_side)
            updated_h = min(updated_h, max_side)

            rotation_matrix[0, 2] += (updated_w / 2) - center[0]
            rotation_matrix[1, 2] += (updated_h / 2) - center[1]

            rotated_arr = cv2.warpAffine(arr, rotation_matrix, (updated_w, updated_h))
            new_ch = channels[channel_num].copy()
            new_ch.data = rotated_arr
            result[channel_num] = new_ch

        combined_mask = None
        for channel_num in channels:
            arr = result[channel_num].data
            if combined_mask is None:
                combined_mask = arr > 0
            else:
                combined_mask |= arr > 0

        if combined_mask is not None:
            coords = cv2.findNonZero(combined_mask.astype(np.uint8))
            if coords is not None:
                x, y, w_sub, h_sub = cv2.boundingRect(coords)
                for channel_num in channels:
                    result[channel_num].data = np.ascontiguousarray(
                        result[channel_num].data[y : y + h_sub, x : x + w_sub]
                    )

        return current_uuid, result

    def rotate_image(self, angle_text: str):
        try:
            angle = float(angle_text)
        except ValueError:
            self.on_error("Please enter a number.")
            return

        if len(self.working_channels) > 0 and angle is not None:
            self.rotation_worker = Worker(
                self.rotate_image_task,
                self.uuid,
                self.working_channels,
                self.reset_working_channels,
                angle,
            )
            self.rotation_worker.signal.connect(self.on_rotation_completed)
            self.rotation_worker.error.connect(self.on_error)
            self.rotation_worker.finished.connect(self.rotation_worker.quit)
            self.rotation_worker.finished.connect(self.rotation_worker.deleteLater)
            self.rotation_worker.start()

    def same_uuid(self, other_uuid):
        return str(self.uuid) == str(other_uuid)

    @pyqtSlot(object)
    def on_rotation_completed(self, output):
        result_uuid, result = output
        logger.debug("rot complete")
        if result is not None:
            for channel_name, wrapper in result.items():
                if "Channel" in channel_name:
                    self.storage.update_data(result_uuid, channel_name, wrapper)
            if self.same_uuid(result_uuid):
                self.working_channels = self.storage.get_data(result_uuid)["data"]
                channel_key = f"Channel {self.current_channel + 1}"
                if channel_key in self.working_channels:
                    self.image_wrapper = self.working_channels[channel_key]
                else:
                    logger.warning(
                        "Channel %s missing after rotation; keeping current wrapper.",
                        channel_key,
                    )
                self.update_image()
            self._clear_caches(result_uuid)
        else:
            raise RuntimeError("Rotation failed")

    @pyqtSlot(str)
    def on_error(self, error_message):
        self.error_signal.emit(error_message)

    def _clear_caches(self, uuid=None, channel=None) -> None:
        super()._clear_caches()
        if uuid is not None:
            if channel is not None:
                self.memory_cache.clear_channel(uuid, channel)
            else:
                self.memory_cache.clear_uuid(uuid)
        else:
            self.memory_cache.clear_all()

    def auto_contrast(self, lower=1.0, upper=99.0, zero_eps=None, min_span=5):
        """
        Auto-contrast using robust percentiles on foreground (non-zero) pixels.
        lower/upper are *percentiles* (e.g., 1.0, 99.0).
        """
        if self.image_wrapper.data.size == 0:
            logger.warning("No image loaded — auto contrast ignored")
            return

        # Get the channel in its native scale (0..1 or 0..255 or whatever
        # scale_adjust gives)
        if self.is_layered:
            channel_num = f"Channel {self.current_channel + 1}"
            if channel_num not in self.working_channels:
                logger.warning("Channel %s missing during auto contrast.", channel_num)
                return
            img = scale_adjust(self.working_channels[channel_num].data)
        else:
            img = scale_adjust(self.image_wrapper.data)

        vmin, vmax = auto_contrast_helper(img, lower, upper, zero_eps, min_span)

        self.update_contrast((int(round(vmin)), int(round(vmax))))

    def blur_layer(
        self,
        blur_percentage: float,
        confirm=False,
        source_uuid=None,
        source_channel=None,
    ):
        """start gaussian blur in a separate thread"""
        if confirm and self.blur_worker is not None and self.blur_worker.isRunning():
            self.blur_worker.wait()
        elif self.blur_worker is not None and self.blur_worker.isRunning():
            # cancel previous worker if still running
            self.blur_worker.terminate()
            self.blur_worker.wait()
        self.blur_worker = Worker(
            self.blur_layer_task, blur_percentage, confirm, source_uuid, source_channel
        )
        # self.blur_worker.signal.connect() # result is rotated_channels
        self.blur_worker.error.connect(self.on_error)
        self.blur_worker.finished.connect(self.blur_worker.quit)
        self.blur_worker.start()

    def blur_layer_task(
        self,
        blur_percentage: float,
        confirm=False,
        source_uuid=None,
        source_channel=None,
    ):
        """
        Applies Gaussian blur to the specified image channel and subtracts
        the specified percentage of the blurred image from the original.

        When source_uuid/source_channel are provided the operation targets that
        image in storage rather than the current canvas image.
        """
        logger.debug("blur")

        if source_uuid is not None and source_channel is not None:
            # Non-canvas image path: read from storage
            img_entry = self.storage.get_data(source_uuid)
            if img_entry is None:
                return
            target_wrapper = img_entry["data"].get(source_channel)
            if target_wrapper is None:
                return
            self._blur_layer = source_channel
            self._blur_source_uuid = str(source_uuid)
            layer_to_blur = copy.deepcopy(target_wrapper.data)
            preview_cmap = target_wrapper.cmap
        else:
            # Default canvas path
            self._blur_layer = f"Channel {self.current_channel + 1}"
            self._blur_source_uuid = str(self.uuid)
            layer_to_blur = copy.deepcopy(self.working_channels[self._blur_layer].data)
            preview_cmap = self.image_wrapper.cmap

        if not confirm:
            blurred_mask = cv2.GaussianBlur(layer_to_blur, (101, 101), 0)
            blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(
                layer_to_blur.dtype
            )
            self.corrected_layer = cv2.subtract(layer_to_blur, blurred_mask_adjusted)
            self.corrected_layer = np.clip(
                self.corrected_layer, 0, np.iinfo(layer_to_blur.dtype).max
            )
            self.update_image(preview_cmap, self.corrected_layer, cache_result=False)

        if confirm and hasattr(self, "corrected_layer"):
            target_uuid = self._blur_source_uuid
            img_entry = self.storage.get_data(target_uuid)
            if img_entry is None:
                return
            wrapper = img_entry["data"].get(self._blur_layer)
            if wrapper is None:
                return
            wrapper_copy = copy.deepcopy(wrapper)
            wrapper_copy.data = self.corrected_layer
            logger.debug(f"saving. {target_uuid} {self._blur_layer}")
            self.storage.update_data(target_uuid, self._blur_layer, wrapper_copy)
            # If this is the current canvas image, keep working_channels in sync
            if str(target_uuid) == str(self.uuid):
                self.working_channels[self._blur_layer].data = self.corrected_layer
                if self._blur_layer in self.working_channels:
                    self.image_wrapper = self.working_channels[self._blur_layer]
            self._clear_caches(target_uuid, self._blur_layer)
            self.image_signal.emit(self.working_channels, False)
            self.update_progress.emit(100, f"Replaced {self._blur_layer}")

    def crop(self, image_rect):
        """Safely crop current image using image_rect and save to disk"""

        # print("Starting crop...")

        left = max(0, image_rect.left())
        top = max(0, image_rect.top())
        right = min(self.image_wrapper.data.shape[1], image_rect.right())
        bottom = min(self.image_wrapper.data.shape[0], image_rect.bottom())
        cropped_array = self.image_wrapper.data[
            top : bottom + 1, left : right + 1
        ]  # this is the current image. if layered then its the current channel

        cropped_array_copy = cropped_array.copy()

        contrast = (self.image_wrapper.contrast_min, self.image_wrapper.contrast_max)

        crop_dialog = ImageDialog(
            self, cropped_array_copy, contrast, self.image_wrapper.cmap
        )
        crop_dialog.exec()
        item = self.storage.get_data(self.uuid)
        assert item is not None, "UUID not found in storage while cropping"
        image_name = item["name"]
        name = f"cropped_{image_name}"

        if crop_dialog.confirm_crop:
            channels = {}
            for channel_name, wrapper in self.working_channels.items():
                arr = wrapper.data
                cropped_array = arr[top : bottom + 1, left : right + 1].copy()
                wrapper_copy = ImageWrapper(
                    cropped_array, name=wrapper.name, cmap=wrapper.cmap
                )
                wrapper_copy.contrast_min = wrapper.contrast_min
                wrapper_copy.contrast_max = wrapper.contrast_max
                channels[channel_name] = wrapper_copy

            target_channel = f"Channel {self.current_channel + 1}"
            self.add_to_canvas(channels, True, name, target_channel)
        else:
            self.crop_signal.emit(False)
        if right <= left or bottom <= top:
            raise ValueError("❌ Invalid crop region: empty or out-of-bounds")

    def crop_with_transform(
        self, moving_uuid, transform, crop_w, crop_h, source_name=None
    ):
        """Rotate-then-crop the moving image using a reference->protein transform.

        ``transform`` is a 2x3 affine T mapping reference coords into the moving
        (protein) image. We invert it so the crop's top-left is reference (0, 0)
        and the rotation is removed, then warp each channel directly into a
        (crop_w, crop_h) output -- a single targeted warp that also crops, rather
        than rotating the whole multi-gigapixel image. Saved as a new image.
        """
        item = self.storage.get_data(str(moving_uuid))
        assert item is not None, "Moving image not found in storage while cropping"
        src_channels = item["data"]

        T = np.asarray(transform, dtype=np.float64).reshape(2, 3)
        A = T[:, :2]
        t = T[:, 2]
        inv_a = np.linalg.inv(A)
        M = np.hstack([inv_a, (-inv_a @ t).reshape(2, 1)]).astype(np.float32)

        crop_w = int(round(crop_w))
        crop_h = int(round(crop_h))
        if crop_w <= 0 or crop_h <= 0:
            raise ValueError("Invalid crop size")

        name = f"cropped_{source_name or item.get('name', 'image')}"
        channels = {}
        for channel_name, wrapper in src_channels.items():
            if "Channel" not in channel_name:
                continue
            original_dtype = wrapper.data.dtype
            src = np.ascontiguousarray(wrapper.data)
            warped = cv2.warpAffine(
                src, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR
            )
            if warped.dtype != original_dtype:
                warped = warped.astype(original_dtype)
            new_wrapper = ImageWrapper(warped, name=wrapper.name, cmap=wrapper.cmap)
            new_wrapper.contrast_min = wrapper.contrast_min
            new_wrapper.contrast_max = wrapper.contrast_max
            channels[channel_name] = new_wrapper

        if not channels:
            raise ValueError("Moving image has no channels to crop")
        self.add_to_canvas(channels, True, name)

    @pyqtSlot()
    def flip_horizontal(self):
        """Flip the image horizontally"""
        for channel_name, wrapper in self.working_channels.items():
            if "Channel" in channel_name:
                wrapper.data = cv2.flip(wrapper.data, 1)
                self.storage.update_data(self.uuid, channel_name, wrapper)
        channel_key = f"Channel {self.current_channel + 1}"
        if channel_key in self.working_channels:
            self.image_wrapper = self.working_channels[channel_key]
        self._clear_caches()
        self.update_image(use_cache=False)
        # self.set_pixmap(self.image_wrapper.data)
        # print("Image flipped horizontally")

    @pyqtSlot()
    def flip_vertical(self):
        """Flip the image vertically"""
        for channel_name, wrapper in self.working_channels.items():
            if "Channel" in channel_name:
                wrapper.data = cv2.flip(wrapper.data, 0)
                self.storage.update_data(self.uuid, channel_name, wrapper)
        channel_key = f"Channel {self.current_channel + 1}"
        if channel_key in self.working_channels:
            self.image_wrapper = self.working_channels[channel_key]
        self._clear_caches()
        self.update_image(use_cache=False)

    def delete_from_canvas(self, uuid: UUID_Type):
        """Delete the current image from the canvas."""
        if super().remove_from_canvas(uuid):
            self.clear_canvas()
            return True
        return False


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
        self.table.setStyleSheet("background-color: transparent")

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

    def metadata_tooltip(self, metadata: dict) -> str:
        """
        Generate a tooltip string from the metadata dictionary.
        """
        tooltip_lines = []
        for key in self.headers:
            value = metadata.get(key, "Unknown")
            tooltip_lines.append(f"<b>{key}:</b> {value}")
        return "<br>".join(tooltip_lines)


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
        self.confirm_crop = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cropped Image")

        self._layout = QVBoxLayout()

        self.image_view = QGraphicsView()

        self.image_scene = QGraphicsScene(self)  # Create a QGraphicsScene
        self.image_view.setScene(self.image_scene)  # Set the scene on the view

        if self.cropped_image.dtype != np.uint8:
            self.cropped_image = scale_adjust(self.cropped_image)

        image = self.apply_contrast(
            self.cropped_image, self.contrast[0], self.contrast[1]
        )
        self.pix = QPixmap(numpy_to_qimage(image))
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
        self.cropped_image = None  # QPixmap owns its own buffer; numpy ref no longer needed

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "cropped_pixmap_item") and not self.cropped_pixmap_item.pixmap().isNull():
            self.image_view.fitInView(
                self.cropped_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
            )
            self.image_view.centerOn(self.cropped_pixmap_item)

    def confirm(self):
        self.confirm_crop = True
        self.accept()

    def cancel(self):
        self.confirm_crop = False
        self.reject()

    def apply_contrast(self, image, new_min, new_max):
        lut = self.create_lut(new_min, new_max)
        return cv2.LUT(image, lut)

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


def contrast_key_from_wrapper(wrapper: ImageWrapper):
    return (wrapper.contrast_min, wrapper.contrast_max)
