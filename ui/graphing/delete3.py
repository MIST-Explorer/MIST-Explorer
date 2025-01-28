from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QHBoxLayout
)
from PyQt6.QtCore import Qt
import sys
from PyQt6.QtCore import pyqtSlot

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

class IconListPage(QWidget):
    def __init__(self, icon_list, navigate_to_page, icon_paths):
        super().__init__()
        self.icon_list = icon_list
        self.navigate_to_page = navigate_to_page
        self.icon_paths = icon_paths  # List of file paths for the icons

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # Remove spacing
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Collapse to the top
        self.setLayout(layout)

        for index, icon_name in enumerate(self.icon_list):
            if icon_name and index < len(self.icon_paths):
                button = QPushButton(icon_name)
                button.setFixedHeight(70)
                button.setStyleSheet("margin-top: 10px;")
                button.setIcon(QIcon(self.icon_paths[index]))  # Set icon for the button
                button.clicked.connect(lambda _, idx=index: self.navigate_to_page(idx))
                layout.addWidget(button)

class IconDetailPage(QWidget):
    def __init__(self, navigate_back, open_in_new_window, encoder):
        super().__init__()
        self.navigate_back = navigate_back
        self.open_in_new_window = open_in_new_window
        self.enc = encoder
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Top layout with back and new window buttons
        self.top_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.navigate_back)
        back_button.setFixedSize(80, 40)
        self.top_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignLeft)

        new_window_button = QPushButton("⤢")
        new_window_button.setFixedSize(40, 40)
        new_window_button.clicked.connect(self.open_in_new_window)
        self.top_layout.addWidget(new_window_button, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addLayout(self.top_layout)

        # Content area for dynamically loaded widget
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_area.setLayout(self.content_layout)
        layout.addWidget(self.content_area)

    def set_icon_index(self, index):
        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            widget_to_remove = self.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        # Add new content based on encoder's graph
        widget = self.enc.get_graph(index)
        widget.setSizePolicy(widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding)
        self.content_layout.addWidget(widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.icon_list = [
            "Icon 1", "Icon 2", "Icon 3",
            "Icon 4", "Icon 5", "Icon 6",
            "Icon 7"
        ]

        ic = ["/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/addchart.png"] * 7
        self.encoder = self.MockEncoder()  # Replace with actual encoder

        self.stacked_widget = QStackedWidget()
        self.icon_list_page = IconListPage(self.icon_list, self.show_icon_detail_page, ic)
        self.icon_detail_page = IconDetailPage(self.show_icon_grid_page, self.open_in_new_window, self.encoder)

        self.stacked_widget.addWidget(self.icon_list_page)
        self.stacked_widget.addWidget(self.icon_detail_page)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.windows = []  # Track multiple new windows

    def show_icon_detail_page(self, index):
        self.icon_detail_page.set_icon_index(index)
        self.stacked_widget.setCurrentWidget(self.icon_detail_page)

    def show_icon_grid_page(self):
        self.stacked_widget.setCurrentWidget(self.icon_list_page)

    def open_in_new_window(self):
        # Create a new window to display the current graph
        new_window = RegenerateOnCloseWindow(
            regenerate_callback=self._on_new_window_closed
        )
        new_window.setWindowTitle("Icon Detail - New Window")
        layout = QVBoxLayout()

        # Retrieve the current graph widget from the encoder
        index = self.icon_detail_page.content_layout.itemAt(0).widget().text().split()[-1]
        widget = self.encoder.get_graph(int(index))
        widget.setSizePolicy(widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding)
        layout.addWidget(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        new_window.setLayout(layout)
        new_window.resize(300, 200)
        new_window.show()

        # Track the new window and update original content
        self.windows.append(new_window)
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.icon_detail_page.content_layout.addWidget(QLabel("visible in new window"))

    @pyqtSlot()
    def _on_new_window_closed(self):
        # When the new window is closed, regenerate the graph in the main window
        index = self.icon_detail_page.content_layout.itemAt(0).widget().text().split()[-1]
        new_graph = self.encoder.get_graph(int(1))

        # Clear old content and update layout with the regenerated graph
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        self.icon_detail_page.content_layout.addWidget(new_graph)

    class MockEncoder:
        def get_graph(self, index):
            widget = QLabel(f"Graph for Icon Index: {index}")
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return widget


class RegenerateOnCloseWindow(QWidget):
    def __init__(self, regenerate_callback):
        super().__init__()
        self.regenerate_callback = regenerate_callback

    def closeEvent(self, event):
        # Call the regenerate callback when the window is closed
        if self.regenerate_callback:
            self.regenerate_callback()
        super().closeEvent(event)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle("Icon List Navigation")
    main_window.resize(400, 400)
    main_window.show()
    sys.exit(app.exec())
