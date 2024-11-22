


# from PyQt6.QtWidgets import QStyledItemDelegate, QComboBox, QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ui.graphing.ZScoreHeatmapWindow import ZScoreHeatmapWindow 
from ui.graphing.SpatialHeatmapUpdated import HeatmapWindow

class AnalysisTab(QWidget):

    def __init__(self, pixmap_label, enc):
        super().__init__()

        self.views = []  # List to hold views
        self.view_index = 0  # Current view index

        self.enc = enc

        self.graphs = []  # List of lists to hold graphs for each view
        self.graph_index = 0  # Current graph index for the active view

        self.rubberbands = []

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Navigation buttons for views
        nav_layout = QHBoxLayout()

        self.save_button = QPushButton("Save Plot")
        self.back_button = QPushButton("< Back")
        self.next_button = QPushButton("Next >")

        self.save_button.clicked.connect(self.save_current_plot)
        self.back_button.clicked.connect(self.prev_view)
        self.next_button.clicked.connect(self.next_view)

        nav_layout.addWidget(self.save_button)
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.next_button)

        main_layout.addLayout(nav_layout)

        # Graph navigation buttons
        self.graph_nav_layout = QHBoxLayout()

        self.back_plot_button = QPushButton("< Previous Graph")
        self.next_plot_button = QPushButton("Next Graph >")

        self.back_plot_button.clicked.connect(self.prev_graph)
        self.next_plot_button.clicked.connect(self.next_graph)

        self.graph_nav_layout.addWidget(self.back_plot_button)
        self.graph_nav_layout.addWidget(self.next_plot_button)

        # Scrollable area for the current view
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(self.graph_nav_layout)

        self.setLayout(main_layout)

    def delete_view(self):
        if self.views:
            self.graphs.pop(self.view_index)
            self.views.pop(self.view_index) 

            self.rubberbands[self.view_index].hide()
            self.rubberbands.pop(self.view_index)
            if len(self.views) == 0:
                for i in reversed(range(self.scroll_layout.count())):
                    widget = self.scroll_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
            else:
                self.set_view(self.view_index - 1)
                self.view_index -= 1
                self.graph_index -= 1
                self.rubberbands[self.view_index].setFilled(True)

            # self.update_graph_navigation()

    def set_view(self, index):
        """Sets the current view by index."""
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Add the first graph of the selected view
        if self.graphs and self.graphs[index]:
            self.scroll_layout.addWidget(self.graphs[index][0])
            self.scroll_layout.addWidget(self.views[index])

        # Update navigation button states
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.views) - 1)

        self.update_graph_navigation()

    def update_graph_navigation(self):
        """Updates the graph navigation buttons."""
        self.back_plot_button.setEnabled(self.graph_index > 0)
        self.next_plot_button.setEnabled(self.graph_index < len(self.graphs[self.view_index]) - 1)

    def add_new_view(self):
        """Adds a new view."""
        sc = QWidget()
        self.views.append(sc)
        self.graphs.append([])  # Add a new list for graphs in this view
        self.view_index = len(self.views) - 1
        self.graph_index = 0
        self.set_view(self.view_index)

    def next_view(self):
        """Navigate to the next view."""
        self.unfill_rubberband()
        if self.view_index < len(self.views) - 1:
            self.view_index += 1
            self.set_view(self.view_index)
            self.fill_rubberband()

    def prev_view(self):
        """Navigate to the previous view."""
        self.unfill_rubberband()
        if self.view_index > 0:
            self.view_index -= 1
            self.set_view(self.view_index)
            self.fill_rubberband()

    def add_graph_to_view(self, graph_widget):
        """Adds a new graph to the current view."""
        if self.views:
            self.graphs[self.view_index].append(graph_widget)
            self.graph_index = len(self.graphs[self.view_index]) - 1
            self.set_view(self.view_index)

    def next_graph(self):
        """Navigate to the next graph in the current view."""
        if self.graph_index < len(self.graphs[self.view_index]) - 1:
            self.graph_index += 1
            self.display_current_graph()

    def prev_graph(self):
        """Navigate to the previous graph in the current view."""
        if self.graph_index > 0:
            self.graph_index -= 1
            self.display_current_graph()

    def display_current_graph(self):
        """Displays the current graph in the current view."""
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        current_graph = self.graphs[self.view_index][self.graph_index]
        self.scroll_layout.addWidget(current_graph)
        self.scroll_layout.addWidget(self.views[self.view_index])
        self.update_graph_navigation()


    # if len(self.rubberbands) != 0:
    #     self.rubberbands[self.view_index].setFilled(False) 

    # self.rubberbands.append(rubberband)
    # self.add_new_view()
    # self.fill_rubberband()

    def analyze_region(self, random_data, rubberband, region):
        """Analyzes a selected region and adds corresponding graphs."""
        if len(self.rubberbands) != 0:
            self.rubberbands[self.view_index].setFilled(False) 

        self.rubberbands.append(rubberband)

        file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        data = pd.read_excel(file_path)
        # data = self.enc.view_tab.load_df()
        print(data)
        print()
    
        # Create a new widget to display the results
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the result details
        result_details_layout = QHBoxLayout()

        # Add a label saying "Selection Results"
        multiComboBox = MultiComboBox()
        multiComboBox.addItems(data.columns)
        result_details_layout.addWidget(multiComboBox)

        # Add a delete button (not hooked up to anything yet)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_view)
        result_details_layout.addWidget(delete_button)

        # Add a rectangle with the color converted to RGB
        pyqt_color_rgb = QColor(rubberband.color).getRgb()[:3]
        color_label = QLabel()
        color_label.setFixedSize(100, 50)
        color_label.setStyleSheet(f"background-color: rgb({pyqt_color_rgb[0]}, {pyqt_color_rgb[1]}, {pyqt_color_rgb[2]});")
        result_details_layout.addWidget(color_label)

        # Add the horizontal layout to the main vertical layout
        result_layout.addLayout(result_details_layout)

        self.scroll_layout.addWidget(result_widget)
        self.add_new_view()
        self.next_view()

        

        # Example: Adding graphs to the current view
        box_plot_graph = self.box_plot(random_data)
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in region])

        self.add_graph_to_view(box_plot_graph)
        self.add_graph_to_view(zscore_heatmap_graph)

        self.views[-1] = result_widget
        self.scroll_layout.addWidget(result_widget)

        self.rubberbands[-1].setFilled(True)

    def box_plot(self, data):
        """Creates a box plot."""
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot(data["Expression"], labels=data['Protein'].unique(), showfliers=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('Protein')
        ax.set_ylabel('Expression Level')
        ax.set_title('Random Protein Expression Box Plot')
        plt.subplots_adjust(bottom=0.25)

        result_layout.addWidget(FigureCanvas(fig))
        return result_widget

    def save_current_plot(self):
        if self.views:
            current_graph = self.views[self.view_index]
            if isinstance(current_graph, FigureCanvas):
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)")
                if file_path:
                    current_graph.figure.savefig(file_path)
            elif isinstance(current_graph, QMainWindow):
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)")
                if file_path:
                    current_graph.figure.savefig(file_path)

    def unfill_rubberband(self):
        try:
            self.rubberbands[self.view_index].setFilled(False)
        except Exception as e:
            print(e)

    def fill_rubberband(self):
        self.rubberbands[self.view_index].setFilled(True)

from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt

class MultiComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))

        # Connect to the dataChanged signal to update the text
        self.model().dataChanged.connect(self.updateText)

    def addItem(self, text: str, data=None):
        item = QStandardItem()
        item.setText(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, items_list: list):
        for text in items_list:
            self.addItem(text)

    def updateText(self):
        selected_items = [self.model().item(i).text() for i in range(self.model().rowCount())
                          if self.model().item(i).checkState() == Qt.CheckState.Checked]
        self.lineEdit().setText(", ".join(selected_items))

    def showPopup(self):
        super().showPopup()
        # Set the state of each item in the dropdown
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            combo_box_view = self.view()
            combo_box_view.setRowHidden(i, False)
            check_box = combo_box_view.indexWidget(item.index())
            if check_box:
                check_box.setChecked(item.checkState() == Qt.CheckState.Checked)

    def hidePopup(self):
        # Update the check state of each item based on the checkbox state
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            combo_box_view = self.view()
            check_box = combo_box_view.indexWidget(item.index())
            if check_box:
                item.setCheckState(Qt.CheckState.Checked if check_box.isChecked() else Qt.CheckState.Unchecked)
        super().hidePopup()