"""
Image Manager module.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import tifffile  # pylint: disable=import-error
from PIL import Image

# pylint: disable=no-name-in-module
from PyQt6.QtCore import QItemSelection, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from core import ImageGraphicsView, ImageStorage, StarDist
from core.canvas import ReferenceGraphicsView
from core.project_manager import ProjectManager
from core.project_naming import (
    SEGMENTATION_BASE_NAME,
    is_segmentation_channel,
    is_temp_project_name,
)
from models.image_list_model import ImageTreeItem, ImageTreeModel
from ui.status_badge_delegate import StatusBadgeDelegate

logger = logging.getLogger(__name__)


class ImageManager(QWidget):
    """
    Widget to manage the list of loaded images.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.image_tree_view = ImageTreeWidget(self)
        self.image_tree_model = ImageTreeModel()
        self.image_tree_view.setModel(self.image_tree_model)
        self.root_node = self.image_tree_model.invisibleRootItem()
        # max icon size
        self.image_tree_view.setIconSize(QSize(50, 50))
        self.image_tree_view.setItemDelegate(StatusBadgeDelegate(self.image_tree_view))
        self.image_tree_view.setHeaderHidden(True)
        self.storage = ImageStorage()
        self.__layout.addWidget(self.image_tree_view)

        # Initialize attributes
        self.model_canvas: Optional[ImageGraphicsView] = None
        self.model_stardist: Optional[StarDist] = None
        self.model_reference_canvas: Optional[ReferenceGraphicsView] = None
        self.current_project_path: Optional[Path] = None
        self.current_project_name: Optional[str] = None
        self.current_project_is_temp: bool = False
        self._unsaved_created: set = set()  # UUIDs (str) of in-session created images

        # Connect deletion signal to backend cleanup
        self.image_tree_view.item_deleted.connect(self._handle_item_deletion)
        self.image_tree_model.itemChanged.connect(self._on_tree_item_renamed)

    def set_project_path(self, project_path: Path):
        """Set the current project path."""
        self.current_project_path = project_path
        metadata = ProjectManager.load_project(project_path)
        if metadata is not None and metadata.name:
            self.current_project_name = metadata.name
            self.current_project_is_temp = bool(
                metadata.is_temporary
            ) or is_temp_project_name(self.current_project_name)
        else:
            self.current_project_name = project_path.stem
            self.current_project_is_temp = is_temp_project_name(
                self.current_project_name
            )

    def _can_save_project(self) -> bool:
        """Return True when project-backed saves are possible."""
        return bool(self.current_project_path) and not self.current_project_is_temp

    def _on_tree_item_renamed(self, item):
        print(f"[rename] itemChanged fired: text={item.text()!r} parent={item.parent()}")
        if item.parent() is not None:
            print(f"[rename] skipping — child item")
            return
        print(f"[rename] _can_save_project={self._can_save_project()} path={self.current_project_path} is_temp={self.current_project_is_temp}")
        if not self._can_save_project():
            print(f"[rename] skipping — cannot save project")
            return
        item_uuid = item.data(Qt.ItemDataRole.UserRole)
        storage_item = self.storage.get_data(item_uuid)
        print(f"[rename] uuid={item_uuid} storage_item_found={storage_item is not None}")
        if storage_item:
            print(f"[rename] calling _save_image_reference for uuid={item_uuid} name={storage_item.get('name')!r}")
            self._save_image_reference(item_uuid, storage_item)
            print(f"[rename] _save_image_reference done")

    def save_all_images(self, copy_data: bool = False):
        """Sync current in-memory images to disk.

        copy_data=False: write only project.json metadata (reference save).
        copy_data=True:  also copy channel TIFFs into the project folder.

        Any images previously saved but no longer in memory are removed from disk.
        """
        if not self._can_save_project():
            return

        model = self.image_tree_model
        current_uuids = set()
        for i in range(model.rowCount()):
            tree_item = model.item(i)
            if tree_item is None:
                continue
            item_uuid = tree_item.data(Qt.ItemDataRole.UserRole)
            if item_uuid is None:
                continue
            current_uuids.add(str(item_uuid))
            item = self.storage.get_data(item_uuid)
            if item is None:
                continue
            if copy_data:
                self._save_image_to_project(item_uuid, item)
            else:
                self._save_image_reference(item_uuid, item)

        # Reconcile: remove from disk any images no longer in memory
        saved_metadata = ProjectManager.load_metadata(self.current_project_path)
        if saved_metadata is not None:
            for img_meta in list(saved_metadata.images):
                if img_meta.uuid not in current_uuids:
                    ProjectManager.delete_image(
                        self.current_project_path, img_meta.uuid
                    )

    def _save_image_reference(self, item_uuid, item):
        """Save only metadata/reference for an image (no pixel data copied)."""
        if item is None:
            return
        item_data = item.get("data", {})
        if not item_data:
            return

        image_name = item.get("name", f"Image_{item_uuid}")
        channel_count = len(item_data)
        original_filename = item.get("original_filename", "")

        contrast_settings = {}
        channel_display_names = {}
        channel_cmaps = {}
        for channel_name, wrapper in item_data.items():
            contrast_settings[channel_name] = (
                wrapper.contrast_min,
                wrapper.contrast_max,
            )
            channel_display_names[channel_name] = wrapper.name or channel_name
            channel_cmaps[channel_name] = wrapper.cmap or "gray"

        ProjectManager.save_image_reference(
            project_path=self.current_project_path,
            image_uuid=str(item_uuid),
            image_name=image_name,
            channel_count=channel_count,
            original_filename=original_filename,
            contrast_settings=contrast_settings,
            channel_display_names=channel_display_names,
            channel_cmaps=channel_cmaps,
        )

    def set_model_canvas(self, model):
        """Set the model canvas."""
        self.model_canvas = model
        self.image_tree_view.model_canvas = model
        if hasattr(model, "update_channel"):
            model.update_channel.connect(self._refresh_canvas_thumbnail)

    def set_model_stardist(self, model):
        """Set the stardist model."""
        self.model_stardist = model
        self.image_tree_view.model_stardist = model

    def set_model_reference_canvas(self, model):
        """Set the reference canvas model."""
        self.model_reference_canvas = model
        self.image_tree_view.model_reference_canvas = model

    def finalize_badge_connections(self):
        """Call once after all three models have been set to wire badge repaints."""
        self.image_tree_view.connect_badge_refresh_signals()

    def add_item(self, item_uuid):
        """Add an item to the tree."""
        logger.debug("adding item")
        assert self.root_node is not None, "Root node is not initialized"
        main_item = ImageTreeItem(item_uuid, channel="Channel 1", useItemName=True)
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        item_data = item["data"]
        self._sync_channel_children(main_item, item_uuid, item_data)
        self.root_node.appendRow(main_item)

    def _save_image_to_project(self, item_uuid, item):
        """Save image data to the project folder."""
        if item is None:
            return False

        item_data = item.get("data", {})
        if not item_data:
            return False

        image_name = item.get("name", f"Image_{item_uuid}")
        channel_count = len(item_data)
        original_filename = item.get("original_filename", "")

        contrast_settings = {}
        channel_display_names = {}
        channel_cmaps = {}
        for channel_name, wrapper in item_data.items():
            contrast_settings[channel_name] = (
                wrapper.contrast_min,
                wrapper.contrast_max,
            )
            channel_display_names[channel_name] = wrapper.name or channel_name
            channel_cmaps[channel_name] = wrapper.cmap or "gray"

        try:
            ProjectManager.save_image(
                project_path=self.current_project_path,
                image_uuid=str(item_uuid),
                channel_data=item_data,
                image_name=image_name,
                channel_count=channel_count,
                original_filename=original_filename,
                contrast_settings=contrast_settings,
                channel_display_names=channel_display_names,
                channel_cmaps=channel_cmaps,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save image {item_uuid} to project: {e}")
            QMessageBox.warning(
                self,
                "Save Failure",
                f"Failed to save image '{image_name}' to disk. It will remain in your session, but you may need to free up space and try saving again.\n\nError: {str(e)}"
            )
            return False

    def _channel_display_text(self, channel: str, wrapper):
        display_name = getattr(wrapper, "name", "") or channel
        if is_segmentation_channel(wrapper):
            if (
                display_name == SEGMENTATION_BASE_NAME
                and self.current_project_name
                and not self.current_project_is_temp
            ):
                display_name = f"{self.current_project_name}_{SEGMENTATION_BASE_NAME}"
            return f"{display_name}"
        if display_name != channel:
            return display_name
        return channel

    @staticmethod
    def _channel_sort_key(channel_key: str):
        try:
            return (0, int(channel_key.replace("Channel ", "")))
        except ValueError:
            return (1, channel_key)

    def _sync_channel_children(self, main_item, item_uuid, image_data):
        desired_channels = []
        if len(image_data) > 1:
            desired_channels = sorted(image_data.keys(), key=self._channel_sort_key)

        existing_channels = []
        for i in range(main_item.rowCount()):
            child = main_item.child(i)
            if child is None:
                continue
            channel_name = child.data(Qt.ItemDataRole.WhatsThisRole)
            if isinstance(channel_name, str):
                existing_channels.append(channel_name)

        if existing_channels == desired_channels:
            return

        if main_item.rowCount() > 0:
            main_item.removeRows(0, main_item.rowCount())

        for channel_name in desired_channels:
            channel_item = ImageTreeItem(
                item_uuid,
                channel=channel_name,
                useItemName=False,
                display_text=self._channel_display_text(
                    channel_name, image_data[channel_name]
                ),
            )
            main_item.appendRow(channel_item)

    def _refresh_canvas_thumbnail(self, channel_int: int):
        """Lightweight thumbnail refresh triggered by canvas display updates.

        Reads from canvas.working_channels — the live in-memory wrappers whose
        contrast_min/max and cmap are kept current by update_contrast/change_cmap.
        Storage holds separate copies that are NOT updated on every contrast change,
        so reading from storage would always produce stale thumbnails.
        """
        canvas = self.model_canvas
        if canvas is None or canvas.uuid is None:
            return
        channel_name = f"Channel {channel_int + 1}"

        # working_channels is the authoritative source for current contrast/cmap.
        data = canvas.working_channels
        if channel_name not in data:
            return

        model = self.image_tree_model
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is None or str(item.data(Qt.ItemDataRole.UserRole)) != str(
                canvas.uuid
            ):
                continue
            # Parent item mirrors Channel 1; update it when that channel changes.
            if isinstance(item, ImageTreeItem) and item.channel == channel_name:
                item.set_icon(data)
            # Update the matching channel child.
            for j in range(item.rowCount()):
                child = item.child(j)
                if (
                    child is not None
                    and isinstance(child, ImageTreeItem)
                    and child.data(Qt.ItemDataRole.WhatsThisRole) == channel_name
                ):
                    child.set_icon(data)
                    break
            break

    def set_channel_icon(self, item_uuid, channel):
        """Set the icon for the channel item."""
        if self.root_node is None:
            return
        image_data = self.storage.get_data(item_uuid)
        if image_data is None:
            return  # UUID not in storage (e.g. internal bookkeeping keys)
        image_data = image_data.get("data", {})
        if not image_data or channel not in image_data:
            return

        model = self.image_tree_model
        main_item = None
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole) == item_uuid:
                main_item = item
                break
        if main_item is None:
            return  # UUID not in sidebar tree (e.g. internal storage entries)

        self._sync_channel_children(main_item, item_uuid, image_data)

        # Always refresh parent icon (parent.channel is always "Channel 1")
        if isinstance(main_item, ImageTreeItem):
            main_item.set_icon(image_data)

        # Refresh the specific child channel
        for i in range(main_item.rowCount()):
            child = main_item.child(i)
            if child is None:
                continue
            if child.data(Qt.ItemDataRole.WhatsThisRole) == channel:
                if isinstance(child, ImageTreeItem):
                    child.set_icon(image_data)
                break
        else:
            # New channel not yet in tree
            if isinstance(main_item, ImageTreeItem) and len(image_data) > 1:
                channel_item = ImageTreeItem(
                    item_uuid,
                    channel=channel,
                    useItemName=False,
                    display_text=self._channel_display_text(
                        channel, image_data[channel]
                    ),
                )
                main_item.appendRow(channel_item)

    def add_to_storage(self, item_uuid, obj):
        """Add data to storage, tracking whether this is an uploaded or created image."""
        logger.debug(f"adding {item_uuid} to storage")
        self.storage.add_data(item_uuid, obj)
        original_filename = obj.get("original_filename", "")
        if not original_filename or not os.path.isfile(original_filename):
            if self._can_save_project():
                if not self._save_image_to_project(item_uuid, obj):
                    self._unsaved_created.add(str(item_uuid))
            else:
                self._unsaved_created.add(str(item_uuid))

    def _handle_item_deletion(self, item_uuid: UUID):
        """Handle backend cleanup when an item is deleted."""
        self.storage.remove_data(item_uuid)
        self._unsaved_created.discard(str(item_uuid))
        if self._can_save_project():
            ProjectManager.delete_image(self.current_project_path, str(item_uuid))

    def has_unsaved_created_images(self) -> bool:
        """Return True if any in-session created images have not been saved or exported."""
        return bool(self._unsaved_created)

    def save_created_images_to_project(self):
        """Save only in-session created images (no real source file) to the project folder."""
        if not self._can_save_project():
            return
        for item_uuid_str in list(self._unsaved_created):
            item = self.storage.get_data(item_uuid_str)
            if item is None:
                self._unsaved_created.discard(item_uuid_str)
                continue
            if self._save_image_to_project(item_uuid_str, item):
                self._unsaved_created.discard(item_uuid_str)

    def save_all_images_to(self, folder: str):
        """Export every workspace image as a TIFF to folder and update project references."""
        if not folder:
            return
        model = self.image_tree_model
        for i in range(model.rowCount()):
            tree_item = model.item(i)
            if tree_item is None:
                continue
            item_uuid = tree_item.data(Qt.ItemDataRole.UserRole)
            if item_uuid is None:
                continue
            item = self.storage.get_data(item_uuid)
            if item is None:
                continue
            name = os.path.splitext(item.get("name", f"Image_{item_uuid}"))[0]
            channel_dict = item.get("data", {})
            arrays = [
                ch.data
                for _, ch in sorted(channel_dict.items())
                if not is_segmentation_channel(ch)
            ]
            if not arrays:
                continue
            file_path = os.path.join(folder, f"{name}.tif")
            stacked = np.stack(arrays, axis=0)
            try:
                tifffile.imwrite(
                    file_path,
                    stacked,
                    photometric="minisblack",
                    imagej=ProjectManager.imagej_compatible_dtype(stacked.dtype),
                )
                item["original_filename"] = file_path
                self._unsaved_created.discard(str(item_uuid))
                self._save_image_reference(str(item_uuid), item)
            except Exception as e:
                logger.error(f"Failed to export image {item_uuid} to {file_path}: {e}")
                QMessageBox.critical(
                    self,
                    "Export Failure",
                    f"Failed to export image '{name}' to {file_path}.\n\nError: {str(e)}"
                )


class ImageTreeWidget(QTreeView):
    """
    Tree view widget for displaying images.
    """

    protein_data = pyqtSignal(UUID, int)
    segmentation_label = pyqtSignal(UUID, int)
    generation_source_selected = pyqtSignal(object, object)
    item_deleted = pyqtSignal(UUID)
    image_dropped = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.storage = ImageStorage()
        self._model_canvas = None
        self._model_stardist = None
        self._model_reference_canvas = None
        self.doubleClicked.connect(self.show_on_canvas)
        self.setAcceptDrops(True)

    # pylint: disable=invalid-name
    def dragEnterEvent(self, event):  # type: ignore
        """Handle drag enter."""
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

    # pylint: disable=invalid-name
    def dragMoveEvent(self, event):  # type: ignore
        """Handle drag move."""
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    # pylint: disable=invalid-name
    def dropEvent(self, event):  # type: ignore
        """Handle drop."""
        if event is None:
            return
        mime = event.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                file_path = url.toLocalFile()
                if file_path is not None:
                    self.image_dropped.emit(file_path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection):
        """Emit generation context whenever image-manager selection changes."""
        super().selectionChanged(selected, deselected)
        indexes = self.selectedIndexes()
        if not indexes:
            self.generation_source_selected.emit(None, None)
            return
        self._emit_generation_source_selection(indexes[0])

    @property
    def model_canvas(self):
        """Get the model canvas."""
        return self._model_canvas

    @model_canvas.setter
    def model_canvas(self, value: ImageGraphicsView):
        self._model_canvas = value

    @property
    def model_reference_canvas(self):
        """Get the reference canvas model."""
        return self._model_reference_canvas

    @model_reference_canvas.setter
    def model_reference_canvas(self, value: ReferenceGraphicsView):
        self._model_reference_canvas = value

    @property
    def model_stardist(self):
        """Get the stardist model."""
        return self._model_stardist

    @model_stardist.setter
    def model_stardist(self, value: StarDist):
        self._model_stardist = value

    def show_on_canvas(self, item):
        """Show selected item on canvas."""
        assert self.model_canvas is not None, "model_canvas is not set"
        logger.debug("show on canvas")
        model_canvas = self.model_canvas
        _, my_uuid = self._name_and_uuid_from_item(item, tooltip=True)
        different = str(my_uuid) != str(model_canvas.uuid)
        if not different:
            channel = self._get_channel_from_item(item)
            model_canvas.swap_channel(channel)
        else:
            channel = self._get_channel_from_item(item, as_int=False)
            model_canvas.add_to_canvas(
                my_uuid, as_new_image=False, target_channel=str(channel)
            )

    def show_on_reference_canvas(self, item_uuid: UUID, channel: int):
        """Show selected item on reference canvas."""
        assert self.model_reference_canvas is not None, (
            "model_reference_canvas is not set"
        )
        cname = f"Channel {channel + 1}"
        self.model_reference_canvas.add_to_canvas(item_uuid, cname)

    # pylint: disable=invalid-name, too-many-locals, too-many-statements
    def contextMenuEvent(self, event):  # type: ignore
        """Handle context menu event."""
        menu = QMenu(self)
        item = self.indexAt(event.pos())
        if item and item.isValid():
            _, item_uuid = self._name_and_uuid_from_item(item)
            channel = int(self._get_channel_from_item(item))
            is_leaf = self._is_leaf(item)
            channel_name = self._get_channel_from_item(item, as_int=False)
            storage_item = self.storage.get_data(item_uuid)
            image_data = storage_item.get("data", {}) if storage_item else {}
            set_reference = QAction("Reference")

            set_stardist_label = QAction("Set as Segmentation Label")

            save_as_tiff = QAction("Save as TIF")

            set_reference.triggered.connect(
                lambda: self.show_on_reference_canvas(item_uuid, channel)
            )
            set_stardist_label.triggered.connect(
                lambda: self.set_as_segmentation_label(item_uuid, channel)
            )

            export_single_channel = channel_name if is_leaf else None
            save_as_tiff.triggered.connect(
                lambda: self.save_as(item, "tif", single_channel=export_single_channel)
            )
            save_as_png = QAction("Save as PNG")
            save_as_png.triggered.connect(
                lambda: self.save_as(item, "png", single_channel=export_single_channel)
            )
            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            model = self.model()
            assert isinstance(model, ImageTreeModel), "Model is not set"
            model_item = model.itemFromIndex(item)
            assert model_item is not None, "Item is None"
            is_root = model_item.parent() is None
            export_formats = self._allowed_export_formats(item)

            if not is_leaf:
                set_menu = QMenu("Set as...", self)
                menu.addMenu(set_menu)
                set_menu.addAction(set_reference)

                set_menu.addAction(set_stardist_label)

                set_default_channel = QMenu("Set Default Channel as", self)
                channel_group = QActionGroup(self)
                channel_group.setExclusive(True)

                current_default = model.itemFromIndex(item)
                assert current_default is not None, "Current default item is None"
                current_default = current_default.data(Qt.ItemDataRole.WhatsThisRole)
                if current_default is None:
                    current_default = "Channel 1"
                children = model.itemFromIndex(item)
                assert children is not None, "Children item is None"
                for i in range(children.rowCount()):
                    child_item = children.child(i)
                    assert child_item is not None, "Child item is None"
                    channel_name = child_item.data(Qt.ItemDataRole.WhatsThisRole)
                    action = QAction(channel_name, self)
                    action.setCheckable(True)
                    action.setChecked(channel_name == current_default)
                    action.triggered.connect(
                        lambda _, channel_name=channel_name: model.setData(
                            item, channel_name, Qt.ItemDataRole.WhatsThisRole
                        )
                    )
                    channel_group.addAction(action)
                    set_default_channel.addAction(action)
                menu.addMenu(set_default_channel)
            else:
                set_reference.setText("Set as Reference")
                menu.addAction(set_reference)
                menu.addAction(set_stardist_label)

                # Rename action for segmentation channels
                wrapper = image_data.get(channel_name)
                if wrapper and is_segmentation_channel(wrapper):
                    rename_action = QAction("Rename", self)
                    rename_action.triggered.connect(
                        lambda _, uid=item_uuid, ch=channel_name, w=wrapper: (
                            self._rename_segmentation_channel(uid, ch, w)
                        )
                    )
                    menu.addAction(rename_action)

                if not is_root:
                    if "tif" in export_formats:
                        menu.addAction(save_as_tiff)
                    if "png" in export_formats:
                        menu.addAction(save_as_png)
                if not is_root and channel_name in image_data:
                    menu.addAction(delete)
            if is_root:
                menu.addAction(delete)
                if "tif" in export_formats:
                    menu.addAction(save_as_tiff)
                if "png" in export_formats:
                    menu.addAction(save_as_png)
            menu.exec(event.globalPos())

    def _rename_segmentation_channel(self, item_uuid, channel_key, wrapper):
        """Allow the user to rename a segmentation channel via dialog."""
        current_name = wrapper.name or channel_key
        new_name, ok = QInputDialog.getText(
            self, "Rename Channel", "New channel name:", text=current_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()

        # Update the wrapper name in storage
        wrapper.name = new_name

        # Update canvas working_channels if this is the active image
        if self.model_canvas and str(self.model_canvas.uuid) == str(item_uuid):
            canvas_wrapper = self.model_canvas.working_channels.get(channel_key)
            if canvas_wrapper is not None:
                canvas_wrapper.name = new_name
            reset_wrapper = self.model_canvas.reset_working_channels.get(channel_key)
            if reset_wrapper is not None:
                reset_wrapper.name = new_name

        # Refresh tree display
        model = self.model()
        if isinstance(model, ImageTreeModel):
            for row in range(model.rowCount()):
                item = model.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == item_uuid:
                    storage_entry = self.storage.get_data(item_uuid)
                    if storage_entry:
                        self._sync_channel_children(
                            item, item_uuid, storage_entry.get("data", {})
                        )
                    break

        # Persist to project
        if self.current_project_path:
            storage_item = self.storage.get_data(item_uuid)
            if storage_item:
                self._save_image_reference(item_uuid, storage_item)

    def set_for_stardist(self, item):
        """Set the image for stardist model."""
        assert isinstance(self.model_stardist, StarDist), "model_stardist is not set"
        name, item_uuid = self._name_and_uuid_from_item(item)
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        data = item["data"]
        self.model_stardist.set_protein_image(data, name=name, source_uuid=item_uuid)

    def show_message(self, message):
        """Show a message box."""
        QMessageBox.information(self, "Selection", message)

    def delete_item(self, index):
        """Delete an item from the tree."""
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(index)
        assert item is not None, "Item is None"
        parent_item = item.parent()
        _, item_uuid = self._name_and_uuid_from_item(index)
        if parent_item is None:
            row = model.indexFromItem(item).row()
            self.item_deleted.emit(item_uuid)
            model.removeRow(row)
            if self.model_canvas is not None and str(self.model_canvas.uuid) == str(
                item_uuid
            ):
                self.model_canvas.clear_canvas()
            return

        channel_name = item.data(Qt.ItemDataRole.WhatsThisRole)
        if not isinstance(channel_name, str):
            return
        image_entry = self.storage.get_data(item_uuid)
        if image_entry is None:
            return
        data = image_entry.get("data", {})
        if channel_name not in data:
            return

        del data[channel_name]
        if not data:
            self.item_deleted.emit(item_uuid)
            model.removeRow(model.indexFromItem(parent_item).row())
            if self.model_canvas is not None and str(self.model_canvas.uuid) == str(
                item_uuid
            ):
                self.model_canvas.clear_canvas()
            return

        channel_mapping = self._renumber_channels(data)
        self._repair_default_channel(parent_item, data, channel_mapping, channel_name)
        manager = self.parent()
        if isinstance(manager, ImageManager):
            manager._sync_channel_children(parent_item, item_uuid, data)
        else:
            parent_item.removeRow(item.row())

        if self.model_canvas is not None and str(self.model_canvas.uuid) == str(
            item_uuid
        ):
            target_channel = parent_item.data(Qt.ItemDataRole.WhatsThisRole)
            if not isinstance(target_channel, str) or target_channel not in data:
                target_channel = next(iter(data))
            self.model_canvas.add_to_canvas(
                item_uuid, as_new_image=False, target_channel=target_channel
            )

        if isinstance(manager, ImageManager) and manager._can_save_project():
            manager._save_image_reference(str(item_uuid), image_entry)

    @staticmethod
    def _channel_sort_key(channel_key: str):
        try:
            return (0, int(channel_key.replace("Channel ", "")))
        except ValueError:
            return (1, channel_key)

    def _renumber_channels(self, data: dict) -> dict[str, str]:
        sorted_keys = sorted(data.keys(), key=self._channel_sort_key)
        remapped = {}
        key_mapping: dict[str, str] = {}

        for idx, old_key in enumerate(sorted_keys, start=1):
            new_key = f"Channel {idx}"
            wrapper = data[old_key]
            if getattr(wrapper, "name", "") == old_key:
                wrapper.name = new_key
            remapped[new_key] = wrapper
            key_mapping[old_key] = new_key

        data.clear()
        data.update(remapped)
        return key_mapping

    def _repair_default_channel(
        self,
        parent_item,
        data: dict,
        key_mapping: dict[str, str],
        deleted_channel: str,
    ):
        default_channel = parent_item.data(Qt.ItemDataRole.WhatsThisRole)
        if not isinstance(default_channel, str):
            default_channel = "Channel 1"
        if default_channel == deleted_channel:
            default_channel = next(iter(data))
        else:
            default_channel = key_mapping.get(default_channel, default_channel)

        if default_channel not in data:
            default_channel = next(iter(data))
        parent_item.setData(default_channel, Qt.ItemDataRole.WhatsThisRole)

    def _name_and_uuid_from_item(self, item, tooltip=False) -> tuple[str, uuid.UUID]:
        item = self.model().itemFromIndex(item)  # type: ignore
        if tooltip is False:
            name = item.text()
        else:
            name = item.data(Qt.ItemDataRole.WhatsThisRole)
        user_role_data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(user_role_data, uuid.UUID):
            item_uuid = user_role_data
        else:
            item_uuid = uuid.UUID(user_role_data)
        if not item_uuid:
            raise ValueError("Item does not have a valid UUID.")

        return name, item_uuid

    def _get_channel_from_item(self, item, as_int=True):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        channel = item.data(Qt.ItemDataRole.WhatsThisRole)
        if as_int is False:
            return str(channel)
        try:
            channel = int(channel.replace("Channel ", "")) - 1
            logger.debug(channel)
        except ValueError as exc:
            raise ValueError("Invalid default channel format.") from exc
        return channel

    def _is_leaf(self, item):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        return not item.hasChildren()

    def _allowed_export_formats(self, item) -> list[str]:
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        model_item = model.itemFromIndex(item)
        assert model_item is not None, "Item is None"
        is_root = model_item.parent() is None
        child_count = model_item.rowCount()
        is_leaf = child_count == 0

        if is_root and child_count > 0:
            return ["tif"]
        if is_root and child_count == 0:
            return ["tif", "png"]
        if not is_root and is_leaf:
            return ["tif", "png"]
        return []

    def _mark_export_saved(self, item_uuid, item: dict, file_path: str):
        """Update storage and notify parent ImageManager after a canonical export.

        A canonical export is a full-image TIF or a segmentation label TIF.
        Updates original_filename in storage so the image is treated as path-backed.
        """
        item["original_filename"] = file_path
        manager = self.parent()
        if isinstance(manager, ImageManager):
            manager._unsaved_created.discard(str(item_uuid))
            if manager._can_save_project():
                manager._save_image_reference(str(item_uuid), item)

    @staticmethod
    def _normalize_to_uint8(image_data: np.ndarray) -> np.ndarray:
        array = np.asarray(image_data)
        if array.dtype == np.uint8:
            return array
        if array.dtype == np.bool_:
            return (array.astype(np.uint8)) * 255
        if array.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)

        float_array = array.astype(np.float32, copy=False)
        finite_mask = np.isfinite(float_array)
        if not finite_mask.any():
            return np.zeros(array.shape, dtype=np.uint8)

        finite_values = float_array[finite_mask]
        min_val = float(finite_values.min())
        max_val = float(finite_values.max())
        if max_val <= min_val:
            return np.zeros(array.shape, dtype=np.uint8)

        normalized = np.zeros_like(float_array, dtype=np.float32)
        normalized[finite_mask] = (float_array[finite_mask] - min_val) / (
            max_val - min_val
        )
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def save_as(self, item, file_type, single_channel=None):
        """Save the item to file."""
        name, item_uuid = self._name_and_uuid_from_item(item)
        name = os.path.splitext(name)[0]
        item = self.storage.get_data(item_uuid)
        assert item is not None, f"No data found for UUID: {item_uuid}"
        channel_dict = item["data"]
        if file_type == "tif":
            logger.debug("inside")
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save TIFF"
            )
            if folder_path:
                file_path = os.path.join(folder_path, f"{name}.tif")
                try:
                    if single_channel is not None and single_channel in channel_dict:
                        channel_array = np.asarray(channel_dict[single_channel].data)
                        tifffile.imwrite(
                            file_path,
                            channel_array,
                            photometric="minisblack",
                            imagej=ProjectManager.imagej_compatible_dtype(
                                channel_array.dtype
                            ),
                        )
                        # Segmentation label export counts as a canonical save
                        exported_wrapper = channel_dict.get(single_channel)
                        if exported_wrapper is not None and is_segmentation_channel(
                            exported_wrapper
                        ):
                            self._mark_export_saved(item_uuid, item, file_path)
                        else:
                            # Regular single-channel: just remove from unsaved set
                            manager = self.parent()
                            if isinstance(manager, ImageManager):
                                manager._unsaved_created.discard(str(item_uuid))
                    else:
                        arrays = [
                            channel_obj.data
                            for _, channel_obj in sorted(channel_dict.items())
                            if not is_segmentation_channel(channel_obj)
                        ]
                        if not arrays:
                            return
                        stacked = np.stack(arrays, axis=0)  # Shape: (channels, H, W)
                        tifffile.imwrite(
                            file_path,
                            stacked,
                            photometric="minisblack",
                            imagej=ProjectManager.imagej_compatible_dtype(stacked.dtype),
                        )
                        # Full-image export: mark saved and update original_filename
                        self._mark_export_saved(item_uuid, item, file_path)
                except Exception as e:
                    logger.error(f"Failed to export TIFF for {item_uuid}: {e}")
                    QMessageBox.critical(
                        self,
                        "Save Failure",
                        f"Failed to save TIFF to {file_path}.\n\nError: {str(e)}"
                    )
        elif file_type == "png":
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save PNG"
            )
            if not folder_path:
                return

            file_path = os.path.join(folder_path, f"{name}.png")

            channel_key = single_channel if single_channel in channel_dict else None
            if channel_key is None:
                if not channel_dict:
                    return
                channel_key = sorted(channel_dict.keys(), key=self._channel_sort_key)[0]

            wrapper = channel_dict.get(channel_key)
            if wrapper is None:
                return

            # Preserve label IDs for integer label arrays.
            # PNG is limited to 16-bit (max 65535).  If an int32 stardist label
            # image has cell IDs above that ceiling, fall back to TIF so the
            # exported pixel values stay in sync with the CellIDs in the CSV.
            channel_array = np.asarray(wrapper.data)
            try:
                if channel_array.dtype == np.int32 and channel_array.max() > 65535:
                    tif_path = os.path.splitext(file_path)[0] + ".tif"
                    tifffile.imwrite(
                        tif_path,
                        channel_array,
                        photometric="minisblack",
                        imagej=False,
                    )
                elif channel_array.dtype in [np.int32, np.uint16, np.uint32, np.int16]:
                    clean_arr = np.clip(channel_array, 0, 65535).astype(np.uint16)
                    Image.fromarray(clean_arr).save(file_path)
                else:
                    if hasattr(wrapper, "get_uint8_data"):
                        png_data = wrapper.get_uint8_data()
                    else:
                        png_data = self._normalize_to_uint8(wrapper.data)
                    if not isinstance(png_data, np.ndarray) or png_data.dtype != np.uint8:
                        png_data = self._normalize_to_uint8(np.asarray(png_data))
                    Image.fromarray(png_data).save(file_path)
            except Exception as e:
                logger.error(f"Failed to export image for {item_uuid}: {e}")
                QMessageBox.critical(
                    self,
                    "Save Failure",
                    f"Failed to save image to {file_path}.\n\nError: {str(e)}"
                )

    def set_as_segmentation_label(self, i_uuid: UUID, channel: int):
        """Set the selected image as the StarDist label image for alignment"""
        self.storage.add_data("seg_label_uuid", {"value": i_uuid, "channel": channel})
        self.segmentation_label.emit(i_uuid, channel)

    # ------------------------------------------------------------------
    # Badge repaint helpers
    # ------------------------------------------------------------------

    def _refresh_badges(self, *_):
        """Repaint immediately — used by all role-state signals except stardist_done."""
        if self._model_canvas is not None:
            ch = self._model_canvas.current_channel
            self.storage.add_data("canvas_channel", {"value": f"Channel {ch + 1}"})
        self.viewport().update()

    def _refresh_badges_deferred(self, *_):
        """Repaint on next event loop tick.

        stardist_done fires before the controller writes the label channel to
        ImageStorage, so we defer one tick to pick up the new data.
        """
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, self.viewport().update)

    def connect_badge_refresh_signals(self):
        """Connect every role-state signal to a viewport repaint.

        Must be called once after set_model_canvas / set_model_stardist /
        set_model_reference_canvas have all been called.

        Connections are grouped as:
          - immediate : repaint on the same tick the state changes
          - deferred  : repaint after the event loop flushes (stardist label write)
          - self      : own signals already carry the UUID, just need repaint
        """
        immediate_signals = [
            self._model_canvas.uuid_changed,
            self._model_canvas.update_channel,
            self._model_reference_canvas.update_reference,
            self._model_reference_canvas.reference_channel_updated,
        ]
        deferred_signals = [
            self._model_stardist.stardist_done,
        ]
        self_signals = [
            self.segmentation_label,
        ]

        for sig in immediate_signals:
            sig.connect(self._refresh_badges)
        for sig in deferred_signals:
            sig.connect(self._refresh_badges_deferred)
        for sig in self_signals:
            sig.connect(self._refresh_badges)

    def _emit_generation_source_selection(self, item_index):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item_index)
        if item is None:
            self.generation_source_selected.emit(None, None)
            return

        item_uuid = item.data(Qt.ItemDataRole.UserRole)
        if item_uuid is None:
            self.generation_source_selected.emit(None, None)
            return

        stardist_channel = self._find_virtual_stardist_channel(item_uuid)
        self.generation_source_selected.emit(item_uuid, stardist_channel)

    def _find_virtual_stardist_channel(self, item_uuid):
        storage_item = self.storage.get_data(item_uuid)
        if storage_item is None:
            return None
        image_data = storage_item.get("data", {})
        for channel_key in sorted(image_data.keys(), key=self._channel_sort_key):
            wrapper = image_data.get(channel_key)
            if wrapper is None:
                continue
            if is_segmentation_channel(wrapper):
                return self._get_channel_idx(channel_key)
        return None

    @staticmethod
    def _channel_sort_key(channel_key: str):
        try:
            return (0, int(channel_key.replace("Channel ", "")))
        except ValueError:
            return (1, channel_key)

    @staticmethod
    def _get_channel_idx(channel_key: str):
        try:
            return int(channel_key.replace("Channel ", "")) - 1
        except ValueError:
            return None
