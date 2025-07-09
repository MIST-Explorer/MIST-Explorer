from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel

from core import ImageStorage
from utils import numpy_to_qimage


class ImageTreeItem(QStandardItem):

    def __init__(self, uuid, channel="Channel 1", useItemName=False, image_ready=False):
        super().__init__()
        self.storage = ImageStorage()
        image_dict = self.storage.get_data(uuid)
        if not image_dict:
            raise ValueError(f"No image data found for UUID: {uuid}")
        name = image_dict.get("name", "")
        text = name if useItemName else channel
        data = image_dict.get("data", {})
        self.useItemName = useItemName
        self.channel = channel
        if not data:
            image_ready = False
        else:
            image_ready = True
        if image_ready:
            icon = numpy_to_qimage(data[channel].data)

            if useItemName:
                thumbnail = QPixmap(icon).scaled(
                    50, 50, Qt.AspectRatioMode.KeepAspectRatio
                )
                self.setData(QSize(0, 60), Qt.ItemDataRole.SizeHintRole)
                self.setEditable(True)
                self.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
            else:
                thumbnail = QPixmap(icon).scaled(
                    30, 30, Qt.AspectRatioMode.KeepAspectRatio
                )
                self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
                self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            icon = QIcon(thumbnail)
            self.setIcon(icon)
        self.setText(text)
        self.setData(uuid, Qt.ItemDataRole.UserRole)
        self.setData(channel, Qt.ItemDataRole.ToolTipRole)

    def set_icon(self):
        """Set the icon for the item. Always a child"""
        data = self.storage.get_data(self.data(Qt.ItemDataRole.UserRole))
        if not data:
            raise ValueError(
                f"No image data found for UUID: {self.data(Qt.ItemDataRole.UserRole)}"
            )
        channel_data = data.get(self.channel, None)
        if channel_data is None:
            raise ValueError(f"No data found for channel: {self.channel}")
        icon = numpy_to_qimage(channel_data.data)
        if icon is None:
            raise ValueError("Failed to convert image data to QImage")
        icon = QPixmap(icon).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio)
        if not icon.isNull():
            icon = QIcon(icon)
        else:
            raise ValueError("Failed to create QIcon from QPixmap")
        self.setData(QSize(0, 40), Qt.ItemDataRole.SizeHintRole)
        self.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.setIcon(icon)

    def onTextEdited(self, new_text):
        self.storage.update_name(self.data(Qt.ItemDataRole.UserRole), new_text)

    def setData(self, value, role=Qt.ItemDataRole.UserRole):
        super().setData(value, role)
        if role == Qt.ItemDataRole.EditRole:
            self.onTextEdited(value)


class ImageTreeModel(QStandardItemModel):
    def __init__(self, images=None):
        super().__init__()
