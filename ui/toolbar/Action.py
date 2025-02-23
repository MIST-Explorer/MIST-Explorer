from PyQt6.QtGui import QIcon, QPixmap, QAction
from PyQt6.QtGui import QPixmap


class Action(QAction):
    def __init__(self, window, tool_name, icon_file_path="", is_checkable=False):
        super().__init__(window)
        self.setCheckable(is_checkable)
        self.setObjectName(tool_name)
        tool_icon = QIcon()
        tool_icon.addPixmap(QPixmap(icon_file_path), QIcon.Mode.Normal, QIcon.State.Off)
        self.setIcon(tool_icon)
        self.setToolTip(tool_name)





