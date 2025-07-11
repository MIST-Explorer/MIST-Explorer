import os
import uuid
from uuid import UUID

import numpy as np
from pyexpat import model
from PyQt6.QtCore import QModelIndex, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QFileDialog,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from core import ImageGraphicsView, ImageStorage, StarDist
from models.image_list_model import ImageTreeItem, ImageTreeModel


class Manager(QWidget):

    def __init__(self, parent):

        super().__init__(parent)

        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.image_tree_view = ImageTreeWidget(self)
        self.image_tree_model = ImageTreeModel()
        self.image_tree_view.setModel(self.image_tree_model)
        self.root_node = self.image_tree_model.invisibleRootItem()
        # max icon size
        self.image_tree_view.setIconSize(QSize(50, 50))
        self.image_tree_view.setHeaderHidden(True)
        self.storage = ImageStorage()
        self.__layout.addWidget(self.image_tree_view)

    def set_model_canvas(self, model):
        self.model_canvas = model
        self.image_tree_view.model_canvas = model

    def set_model_stardist(self, model):
        self.model_stardist = model
        self.image_tree_view.model_stardist = model

    def add_item(self, uuid):
        print("adding item")
        assert self.root_node is not None, "Root node is not initialized"
        main_item = ImageTreeItem(uuid, channel="Channel 1", useItemName=True)
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        item_data = item["data"]
        channels = item_data.keys()
        channels = sorted(channels, key=lambda x: int(x.replace("Channel ", "")))
        if len(channels) > 1:
            for channel in channels:
                channel_item = ImageTreeItem(uuid, channel=channel, useItemName=False)
                main_item.appendRow(channel_item)
        self.root_node.appendRow(main_item)

    def set_channel_icon(self, uuid, channel):
        """Set the icon for the channel item"""
        assert self.root_node is not None, "Root node is not initialized"
        main_item = self.root_node.child(uuid)
        if main_item is None:
            raise ValueError(f"No main item found for UUID: {uuid}")
        channel_item = main_item.child(channel)
        assert isinstance(
            channel_item, ImageTreeItem
        ), "Channel item is not an instance of ImageTreeItem"
        if channel_item is None:
            raise ValueError(
                f"No channel item found for UUID: {uuid} and channel: {channel}"
            )
        channel_item.set_icon()

    def add_to_storage(self, uuid, obj):
        self.storage.add_data(uuid, obj)

    def update_item_layer(self, uuid, new_data, layer_name):
        self.storage.update_data(uuid, new_data, layer_name)

    # !TODO: move elsewhere


class ImageTreeWidget(QTreeView):
    tissue_target_selected = pyqtSignal(UUID, bool, int)
    tissue_unaligned_selected = pyqtSignal(UUID, bool, int)
    item_deleted = pyqtSignal(
        UUID
    )  # !TODO: implement this to make all other components reset to default if the deleted item is currently being used

    def __init__(self, parent):
        super().__init__(parent)
        self.storage = ImageStorage()
        self._model_canvas = None
        self._model_stardist = None
        self.doubleClicked.connect(self.display_image)

    @property
    def model_canvas(self):
        return self._model_canvas

    @model_canvas.setter
    def model_canvas(self, value: ImageGraphicsView):
        self._model_canvas = value

    @property
    def model_stardist(self):
        return self._model_stardist

    @model_stardist.setter
    def model_stardist(self, value: StarDist):
        self._model_stardist = value

    def display_image(self, item):
        assert self.model_canvas is not None, "model_canvas is not set"
        model_canvas = self.model_canvas
        channel, my_uuid = self._name_and_uuid_from_item(item, tooltip=True)
        different = str(my_uuid) != str(model_canvas.uuid)
        if not different:
            channel = self._get_channel_from_item(item)
            model_canvas.swap_channel(channel)
        else:
            model_canvas.add_to_canvas(
                my_uuid, as_new_image=False, target_channel=channel
            )

    def contextMenuEvent(self, event):  # type: ignore
        menu = QMenu(self)
        item = self.indexAt(event.pos())
        if item:
            _, uuid = self._name_and_uuid_from_item(item)
            channel = self._get_channel_from_item(item)
            is_leaf = self._is_leaf(item)
            set_reference = QAction("Reference")
            set_cell_image = QAction("Cell Image")

            set_tissue_target_image = QAction("Tissue Target Image")
            set_tissue_unaligned_image = QAction("Tissue Unaligned Image")

            save_as_tiff = QAction("Save as TIF")

            set_reference.triggered.connect(
                lambda: self.show_message("reference selected")
            )
            set_cell_image.triggered.connect(lambda: self.set_for_stardist(item))
            set_tissue_target_image.triggered.connect(
                lambda: self.set_as_tissue_target(uuid, is_leaf, channel)
            )
            set_tissue_unaligned_image.triggered.connect(
                lambda: self.set_as_tissue_unaligned(uuid, is_leaf, channel)
            )

            save_as_tiff.triggered.connect(lambda: self.save_as(item, "tif"))
            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            model = self.model()
            assert isinstance(model, ImageTreeModel), "Model is not set"

            if not is_leaf:
                set_menu = QMenu("Set as...", self)
                menu.addMenu(set_menu)
                set_menu.addAction(set_reference)
                set_menu.addAction(set_cell_image)
                set_menu.addAction(set_tissue_target_image)
                set_menu.addAction(set_tissue_unaligned_image)

                set_default_channel = QMenu("Set Default Channel as", self)
                channel_group = QActionGroup(self)
                channel_group.setExclusive(True)

                current_default = model.itemFromIndex(item)
                assert current_default is not None, "Current default item is None"
                current_default = current_default.data(Qt.ItemDataRole.ToolTipRole)
                if current_default is None:
                    current_default = "Channel 1"
                children = model.itemFromIndex(item)
                assert children is not None, "Children item is None"
                for i in range(children.rowCount()):
                    child_item = children.child(i)
                    assert child_item is not None, "Child item is None"
                    channel_name = child_item.data(Qt.ItemDataRole.ToolTipRole)
                    action = QAction(channel_name, self)
                    action.setCheckable(True)
                    action.setChecked(channel_name == current_default)
                    action.triggered.connect(
                        lambda _, channel_name=channel_name: (
                            model.setData(
                                item, channel_name, Qt.ItemDataRole.ToolTipRole
                            )
                        )
                    )
                    channel_group.addAction(action)
                    set_default_channel.addAction(action)
                menu.addMenu(set_default_channel)
                menu.addAction(delete)
            else:
                set_reference.setText("Set as Reference")
                set_cell_image.setText("Set as Cell Image")
                set_tissue_target_image.setText("Set as Tissue Target Image")
                set_tissue_unaligned_image.setText("Set as Tissue Unaligned Image")
                menu.addAction(set_reference)
                menu.addAction(set_cell_image)
                menu.addAction(set_tissue_target_image)
                menu.addAction(set_tissue_unaligned_image)

            menu.addAction(save_as_tiff)
            menu.exec(event.globalPos())

    def set_for_stardist(self, item):
        assert isinstance(self.model_stardist, StarDist), "model_stardist is not set"
        _, uuid = self._name_and_uuid_from_item(item)
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        data = item["data"]
        self.model_stardist.set_protein_image(data)

    def show_message(self, message):
        QMessageBox.information(self, "Selection", message)

    def delete_item(self, index: QModelIndex):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(index)
        _, uuid = self._name_and_uuid_from_item(index)
        assert item is not None, "Item is None"
        row = model.indexFromItem(item).row()
        self.item_deleted.emit(uuid)
        model.removeRow(row)

    def _name_and_uuid_from_item(self, item, tooltip=False) -> tuple[str, uuid.UUID]:
        item = self.model().itemFromIndex(item)  # type: ignore
        if tooltip is False:
            name = item.text()
        else:
            name = item.data(Qt.ItemDataRole.ToolTipRole)
        item_uuid = uuid.UUID(item.data(Qt.ItemDataRole.UserRole))
        if not item_uuid:
            raise ValueError("Item does not have a valid UUID.")

        return name, item_uuid

    def _get_channel_from_item(self, item):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        channel = item.data(Qt.ItemDataRole.ToolTipRole)
        try:
            channel = int(channel.replace("Channel ", "")) - 1
        except ValueError:
            raise ValueError("Invalid default channel format.")
        return channel

    def _is_leaf(self, item):
        model = self.model()
        assert isinstance(model, ImageTreeModel), "Model is not set"
        item = model.itemFromIndex(item)
        assert item is not None, "Item is None"
        return not item.hasChildren()

    def save_as(self, item, type):
        name, uuid = self._name_and_uuid_from_item(item)
        name = os.path.splitext(name)[0]
        item = self.storage.get_data(uuid)
        assert item is not None, f"No data found for UUID: {uuid}"
        channel_dict = item["data"]
        if type == "tif":
            import tifffile

            print("inside")
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save TIFF"
            )
            if folder_path:
                file_path = os.path.join(folder_path, f"{name}.tif")
                arrays = [
                    channel_obj.data for _, channel_obj in sorted(channel_dict.items())
                ]
                stacked = np.stack(arrays, axis=0)  # Shape: (channels, H, W)
                tifffile.imwrite(
                    file_path, stacked, photometric="minisblack", imagej=True
                )

    def set_as_tissue_target(self, i_uuid: UUID, is_leaf: bool, channel: int):
        """Set the selected image as the tissue target image for alignment"""

        self.tissue_target_selected.emit(i_uuid, is_leaf, channel)

    def set_as_tissue_unaligned(self, i_uuid: UUID, is_leaf: bool, channel: int):
        """Set the selected image as the tissue unaligned image for alignment"""
        self.tissue_unaligned_selected.emit(i_uuid, is_leaf, channel)
