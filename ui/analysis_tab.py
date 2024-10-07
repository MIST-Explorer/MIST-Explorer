from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal
import random

import ui.graphing.Test as test


class AnalysisTab(QWidget):
    add_graph_signal = pyqtSignal()

    def __init__(self, pixmap_label):
        super().__init__()

        self.graphs = []
        self.current_graph_index = 0

        self.initUI()
        self.add_graph_signal.connect(self.add_graph_to_new_view)

    def initUI(self):
        main_layout = QVBoxLayout()

        # Add navigation buttons
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("< Back")
        self.next_button = QPushButton("Next >")
        self.back_button.clicked.connect(self.show_previous_graph)
        self.next_button.clicked.connect(self.show_next_graph)
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.next_button)

        main_layout.addLayout(nav_layout)

        # Add buttons for adding random graphs
        add_graph_layout = QHBoxLayout()
        self.add_random_graph_new_view_button = QPushButton("Add Random Graph to New View")
        self.add_random_graph_current_view_button = QPushButton("Add Random Graph to Current View")
        self.add_random_graph_new_view_button.clicked.connect(self.add_random_graph_to_new_view)
        self.add_random_graph_current_view_button.clicked.connect(self.add_random_graph_to_current_view)
        add_graph_layout.addWidget(self.add_random_graph_new_view_button)
        add_graph_layout.addWidget(self.add_random_graph_current_view_button)

        main_layout.addLayout(add_graph_layout)

        # Add button for adding a new line to the current graph
        self.add_line_button = QPushButton("Add Line to Current Graph")
        self.add_line_button.clicked.connect(self.add_line_to_current_graph)
        main_layout.addWidget(self.add_line_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.scroll_area)

        self.setLayout(main_layout)

        # Add initial graph
        self.add_graph_to_new_view()

    def add_graph_to_new_view(self):
        sc = test.Window()
        self.graphs.append(sc)
        self.current_graph_index = len(self.graphs) - 1
        self.show_graph(self.current_graph_index)

    def add_graph_to_current_view(self):
        sc = test.Window()
        self.scroll_layout.addWidget(sc)

    def add_random_graph_to_new_view(self):
        sc = self.create_random_graph()
        self.graphs.append(sc)
        self.current_graph_index = len(self.graphs) - 1
        self.show_graph(self.current_graph_index)

    def add_random_graph_to_current_view(self):
        sc = self.create_random_graph()
        self.scroll_layout.addWidget(sc)

    def create_random_graph(self):
        # This function should create and return a random graph
        # For demonstration purposes, we'll just return a new test.Window()
        return test.Window()

    def show_graph(self, index):
        # Clear the current layout
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Add the new graph
        self.scroll_layout.addWidget(self.graphs[index])

        # Update button states
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.graphs) - 1)

    def show_previous_graph(self):
        if self.current_graph_index > 0:
            self.current_graph_index -= 1
            self.show_graph(self.current_graph_index)

    def show_next_graph(self):
        if self.current_graph_index < len(self.graphs) - 1:
            self.current_graph_index += 1
            self.show_graph(self.current_graph_index)

    def add_line_to_current_graph(self):
        if self.graphs:
            current_graph = self.graphs[self.current_graph_index]
            current_graph.redraw()

    def get_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    pixmap_label = QLabel()
    analysis_tab = AnalysisTab(pixmap_label)
    analysis_tab.show()
    sys.exit(app.exec())
