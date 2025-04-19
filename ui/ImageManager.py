from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from core.canvas import ImageStorage
from utils import numpy_to_qimage

class Manager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Image List")
        self.__layout = QVBoxLayout(self)
        self.list_widget = ListWidget(self)
        self.__layout.addWidget(self.list_widget)

    def add_item(self, data, name):
        print("adding item")
        item = QListWidgetItem(self.list_widget)
        h_layout = QHBoxLayout()

        thumbnail_label = QLabel(self)
        thumbnail_pixmap = QPixmap(numpy_to_qimage(data))
        thumbnail_label.setPixmap(thumbnail_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio))
        h_layout.addWidget(thumbnail_label)

        text_label = QLineEdit(name, self)

        text_label.setStyleSheet("QLineEdit { border: none; background: transparent; }")

        h_layout.addWidget(text_label)

        item_widget = QWidget()
        item_widget.setLayout(h_layout)
        
        self.list_widget.setItemWidget(item, item_widget)

        item.setSizeHint(item_widget.sizeHint())
    

class ListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        item = self.itemAt(event.pos())
        if item:
            set_menu = QMenu("Set as ...", self)
            set_reference = QAction("Reference")
            set_target = QAction("Target")
            set_cell_image = QAction("Cell Image")
            # rename = QAction("Rename")

            set_menu.addAction(set_reference)
            set_menu.addAction(set_target)
            set_menu.addAction(set_cell_image)
            set_reference.triggered.connect(lambda: self.show_message("reference selected"))
            set_target.triggered.connect(lambda: self.show_message("target selected"))
            set_cell_image.triggered.connect(lambda: self.show_message("cell image selected"))

            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self.delete_item(item))
            # rename.triggered.connect(lambda: self.rename(item))

            menu.addMenu(set_menu)
            # menu.addAction(rename)
            menu.addAction(delete)

            menu.exec(event.globalPos())

    def show_message(self, message):
        QMessageBox.information(self, "Selection", message)
        
    def delete_item(self, item:QListWidgetItem):
        row = self.row(item)
        self.takeItem(row)
