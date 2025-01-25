


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
from ui.graphing.CellDensityPlot import CellDensityPlot
from ui.graphing.DistributionViewer import DistributionViewer
from ui.graphing.PieChartCanvas import PieChartCanvas

class AnalysisTab(QWidget):

    def __init__(self, pixmap_label, enc):
        super().__init__()

        self.views = []  # List to hold views
        self.view_index = 0  # Current view index

        self.enc = enc

        self.graphs = []  # List of lists to hold graphs for each view
        self.graph_index = 0  # Current graph index for the active view

        self.rubberbands = []
        self.regions = []

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

            self.regions.pop(self.view_index)
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
            
            return True
        
        return False
    

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
            # self.graph_index = len(self.graphs[self.view_index]) - 1
            self.set_view(self.view_index)

    def remove_all_graphs_at_index(self):
        if self.views:
            for i in reversed(range(len(self.graphs[self.view_index]))):
                self.graphs[self.view_index][i].close()
            self.graphs[self.view_index] = []
            self.graph_index = 0
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

    def analyze_region(self, rubberband, region):
        """Analyzes a selected region and adds corresponding graphs."""
        if len(self.rubberbands) != 0:
            self.rubberbands[self.view_index].setFilled(False) 

        self.rubberbands.append(rubberband)
        self.regions.append(region)

        data = self.enc.view_tab.load_df()

        # Create a new widget to display the results
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the result details
        result_details_layout = QHBoxLayout()

        # Add a label saying "Selection Results"
        self.multiComboBox = MultiComboBox()
        self.multiComboBox.addItem("Select All")
        self.multiComboBox.addItem("Deselect All")
        self.multiComboBox.addItems(data.columns[3:])
        
        
        for i in range(0, len(data.columns[3:])):
            self.multiComboBox.model().item(i).setCheckState(Qt.CheckState.Checked)
        result_details_layout.addWidget(self.multiComboBox)

        # Create a vertical layout for the buttons
        buttons_layout = QVBoxLayout()

        # Add "Apply" button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda: self.handleComboBoxChanged(self.multiComboBox.get_checked_items()))
        buttons_layout.addWidget(apply_button)

        # Add "Delete" button
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_view)
        buttons_layout.addWidget(delete_button)

        # Create a widget to hold the buttons' layout
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)

        # Add the combined widget to the result_details_layout
        result_details_layout.addWidget(buttons_widget)

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
        print("multi", self.multiComboBox.lineEdit().text())

        box_plot_graph = self.box_plot(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in region])
        zscore_HeatmapWindow = HeatmapWindow(data, [i * 4 for i in region])
        zscore_cellDens = CellDensityPlot(data, [i * 4 for i in region]) 
        zscore_PieChartCanvas = PieChartCanvas(data, [i * 4 for i in region]) 
        zDistributionViewer = DistributionViewer(data)

        self.add_graph_to_view(box_plot_graph)
        self.add_graph_to_view(zscore_heatmap_graph)
        self.add_graph_to_view(zscore_HeatmapWindow)
        # self.add_graph_to_view(zscore_cellDens)
        self.add_graph_to_view(zscore_PieChartCanvas)
        self.add_graph_to_view(zDistributionViewer)
        self.graph_index = 0
        self.update_graph_navigation()
        # self.back_plot_button.click()

        # self.next_plot_button.click()
        # self.display_current_graph()
        
        self.views[-1] = result_widget
        self.scroll_layout.addWidget(result_widget)

        self.rubberbands[-1].setFilled(True)

    def handleComboBoxChanged(self, checked_items):
        if len(checked_items) == 0:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Alert")
            msg_box.setText("You have nothing selected! Please select at least one protein.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return

        self.remove_all_graphs_at_index()

        data = self.enc.view_tab.load_df()
        result = list(set(data.columns[3:]) - set(checked_items))

        data = data.drop(columns=result)

        region = self.regions[self.view_index]
        box_plot_graph = self.box_plot(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_HeatmapWindow = HeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_cellDens = CellDensityPlot(data, [i * 4 for i in region]) 
        zscore_PieChartCanvas = PieChartCanvas(data, [i * 4 for i in region]) 
        zDistributionViewer = DistributionViewer(data)

        self.add_graph_to_view(box_plot_graph)
        self.add_graph_to_view(zscore_heatmap_graph)
        self.add_graph_to_view(zscore_HeatmapWindow)
        # self.add_graph_to_view(zscore_cellDens)
        self.add_graph_to_view(zscore_PieChartCanvas)
        self.add_graph_to_view(zDistributionViewer)

    def box_plot(self, data, region):
        """Creates a box plot."""
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        x_min, y_min, x_max, y_max = region

        # Filter the dataset to include only cells within the specified region
        filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                            (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]
        
        filtered_data = filtered_data.iloc[:, 3:]
    
        fig, ax = plt.subplots(figsize=(12, 8))
        filtered_data.boxplot(ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('Protein')
        ax.set_ylabel('Expression Level')
        ax.set_title('Protein Expression Box Plot')
        plt.subplots_adjust(bottom=0.3)

        canvas = FigureCanvas(fig)
        result_layout.addWidget(canvas)
        result_widget.figure = fig
        
        return result_widget

    def save_current_plot(self):
        print("plot save cliked")
        
        current_graph = self.graphs[self.view_index][self.graph_index]
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
    itemsCheckedChanged = pyqtSignal(list)  # Signal to emit the list of checked items

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))

        # Connect to the dataChanged signal to update the text and emit our signal
        self.model().dataChanged.connect(self.onItemStateChanged)

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

    def onItemStateChanged(self):
        # Update the displayed text
        self.updateText()

        items = self.get_checked_items2()

        if "Select All" in items and not "Deselect All"  in items:
            for i in range(self.model().rowCount()):
                    item = self.model().item(i)
                    if item.text() != "Deselect All":
                        item.setCheckState(Qt.CheckState.Checked)

        if "Deselect All" in items:
            for i in range(self.model().rowCount()):
                    item = self.model().item(i)
                    item.setCheckState(Qt.CheckState.Unchecked)
            
            
        # self.updateText()   # Emit the custom si"Deselect All"gnal with the list of checked items
        # self.itemsCheckedChanged.emit()
    
    def get_checked_items(self):
        items = [self.model().item(i).text() for i in range(self.model().rowCount())
                if self.model().item(i).checkState() == Qt.CheckState.Checked]
        
        return [item for item in items if item not in ["Select All", "Deselect All"]]   
    
    # stupid backwards compatability issue 
    def get_checked_items2(self):
        return [self.model().item(i).text() for i in range(self.model().rowCount())
                if self.model().item(i).checkState() == Qt.CheckState.Checked]