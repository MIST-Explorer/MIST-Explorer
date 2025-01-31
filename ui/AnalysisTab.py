


# from PyQt6.QtWidgets import QStyledItemDelegate, QComboBox, QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSlot

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
from ui.graphing.delete_later import UMAPVisualizer


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
        # top three buttons
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

        # scroll area in middle
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area = QScrollArea()

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)

        # # bottom two buttons holder
        # self.graph_nav_layout = QHBoxLayout()

        # self.back_plot_button = QPushButton("< Previous Graph")
        # self.next_plot_button = QPushButton("Next Graph >")

        # self.back_plot_button.clicked.connect(self.prev_graph)
        # self.next_plot_button.clicked.connect(self.next_graph)

        # self.graph_nav_layout.addWidget(self.back_plot_button)
        # self.graph_nav_layout.addWidget(self.next_plot_button)
        
        # END ^^
        # Now add all to main layout
        main_layout = QVBoxLayout()

        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.scroll_area)
        # main_layout.addLayout(self.graph_nav_layout)

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

        # # Add the first graph of the selected view
        # if self.graphs and self.graphs[index]:
        #     self.scroll_layout.addWidget(self.graphs[index][0])
        #     self.scroll_layout.addWidget(self.views[index])

        # # Update navigation button states
        # self.back_button.setEnabled(index > 0)
        # self.next_button.setEnabled(index < len(self.views) - 1)

        # self.update_graph_navigation()


    def add_new_view(self):
        """Adds a new view."""
        sc = QWidget()
        self.views.append(sc)
        self.graphs.append([])  # Add a new list for graphs in this view
        self.view_index = len(self.views) - 1
        self.graph_index = 0
        self.set_view(self.view_index)

    # TODO: fix order of this
    def next_view(self):
        """Navigate to the next view."""
        
        if self.view_index < len(self.views) - 1:
            self.unfill_rubberband()
            self.view_index += 1
            self.set_view(self.view_index)
            self.fill_rubberband()

    # TODO: fix order of this
    def prev_view(self):
        """Navigate to the previous view."""
        
        if self.view_index > 0:
            self.unfill_rubberband()
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

    def get_graph(self, i):
        self.graph_index = i 
        # """Displays the current graph in the current view."""
        # for i in reversed(range(self.scroll_layout.count())):
        #     widget = self.scroll_layout.itemAt(i).widget()
        #     if widget is not None:
        #         widget.setParent(None)

        current_graph = self.graphs[self.view_index][i]
        # self.scroll_layout.addWidget(current_graph)
        # self.scroll_layout.addWidget(self.views[self.view_index])
        # self.update_graph_navigation()
        return current_graph
    
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
        index = self.graph_index
        widget = self.get_graph(int(index))
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
        index = self.graph_index
        new_graph = self.get_graph(int(self.graph_index))

        # Clear old content and update layout with the regenerated graph
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        self.icon_detail_page.content_layout.addWidget(new_graph)


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
        # result_layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the result details
        result_details_layout = QHBoxLayout()

        # Add a label saying "Selection Results"
        self.multiComboBox = MultiComboBox()
        self.multiComboBox.addItem("Select All")
        self.multiComboBox.addItem("Deselect All")
        self.multiComboBox.addItems(data.columns[3:])
        
        for i in range(0, len(data.columns[3:])):
            self.multiComboBox.model().item(i).setCheckState(Qt.CheckState.Checked)
        # result_details_layout.addWidget(self.multiComboBox)

        # Add "Apply" button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda: self.handleComboBoxChanged(self.multiComboBox.get_checked_items()))
        # buttons_layout.addWidget(apply_button)

        # Create a vertical layout for the buttons
        buttons_layout = QVBoxLayout()

        bounds = QLabel(f"Bounds: {region}")
        buttons_layout.addWidget(bounds)

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

        box_plot_graph = self.box_plot(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_HeatmapWindow = HeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        # zscore_cellDens = CellDensityPlot(data, [i * 4 for i in region]) 
        zscore_PieChartCanvas = PieChartCanvas(data, [i * 4 for i in region]) 
        zDistributionViewer = DistributionViewer(data)

        self.windows = []

        self.icon_list = [
            "Boxplot", "Z-Scores Heatmap", "Spatial Heatmap",
            "Pi Chart", "Histogram", "UMAP" 
        ]

        self.icon_path = ["/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/linechart.png",
                          "/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/heatmap.png",
                          "/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/heatmap.png",
                          "/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/piechart.png",
                          "/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/barchart.png",
                          "/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/scatter.png",
                          ]

        

        # self.icon_list = self.icon_grid
        self.stacked_widget = QStackedWidget()

        self.icon_list_page = IconListPage(self.icon_list, self.show_icon_detail_page, self.icon_path, result_details_layout)
        self.icon_detail_page = IconDetailPage(self.show_icon_grid_page, self.open_in_new_window, self)

        self.stacked_widget.addWidget(self.icon_list_page)
        self.stacked_widget.addWidget(self.icon_detail_page)

        layout = QVBoxLayout()
        # layout.addWidget(self.stacked_widget)
        layout.addWidget(self.stacked_widget)
        self.new_window_instance = None

        result_layout.addLayout(layout)

        self.scroll_layout.addWidget(result_widget)
        self.add_new_view()
        self.next_view()

        
        # TODO: fix static 4, shoulkd be based on whatever
        box_plot_graph = self.box_plot(data, [i * 4 for i in self.regions[self.view_index]])
        b1 = QWidget()
        boxxer = QVBoxLayout()
        boxxer.addWidget(box_plot_graph)
        boxxer.addWidget(apply_button)
        boxxer.addWidget(self.multiComboBox)

        b1.setLayout(boxxer)
        self.add_graph_to_view(b1)
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in region])
        zscore_HeatmapWindow = HeatmapWindow(data, [i * 4 for i in region])
        # zscore_cellDens = CellDensityPlot(data, [i * 4 for i in region]) 
        zscore_PieChartCanvas = PieChartCanvas(data, [i * 4 for i in region]) 
        zDistributionViewer = DistributionViewer(data)

        umaper = UMAPVisualizer()

        
        self.add_graph_to_view(zscore_heatmap_graph)
        self.add_graph_to_view(zscore_HeatmapWindow)
        self.add_graph_to_view(zscore_PieChartCanvas)
        self.add_graph_to_view(zDistributionViewer)
        self.add_graph_to_view(umaper)

        
        

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

        x_min, y_min, x_max, y_max = region
        filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                            (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]

        region = self.regions[self.view_index]
        box_plot_graph = self.box_plot(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_heatmap_graph = ZScoreHeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        zscore_HeatmapWindow = HeatmapWindow(data, [i * 4 for i in self.regions[self.view_index]])
        # zscore_cellDens = CellDensityPlot(data, [i * 4 for i in region]) 
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
    

    from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QStackedWidget
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QStackedWidget, QHBoxLayout
)
from PyQt6.QtCore import Qt
import sys

class IconGridPage(QWidget):
    def __init__(self, icon_grid, navigate_to_page):
        super().__init__()
        self.icon_grid = icon_grid
        self.navigate_to_page = navigate_to_page
        layout = QGridLayout()
        self.setLayout(layout)
        for row_idx, row in enumerate(self.icon_grid):
            for col_idx, icon_name in enumerate(row):
                if icon_name:
                    button = QPushButton(icon_name)
                    button.setFixedSize(100, 100)
                    index = row_idx * len(row) + col_idx
                    button.clicked.connect(lambda _, idx=index: self.navigate_to_page(idx))
                    layout.addWidget(button, row_idx, col_idx)
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

class IconListPage(QWidget):
    def __init__(self, icon_list, navigate_to_page, icon_paths, result_details_layout):
        super().__init__()
        self.icon_list = icon_list
        self.navigate_to_page = navigate_to_page
        self.icon_paths = icon_paths  # List of file paths for the icons

        layout = QVBoxLayout()
        

        title_label = QLabel("View Graphs")
        # title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # Remove spacing
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Collapse to the top
        self.setLayout(layout)

        for index, icon_name in enumerate(self.icon_list):
            if icon_name and index < len(self.icon_paths):
                button = QPushButton(icon_name)
                button.setFixedHeight(70)
                
                button.setStyleSheet(" text-align:left; padding: 10px; margin-top: 10px;")
                button.setIcon(QIcon(self.icon_paths[index]))  # Set icon for the button
                
                button.clicked.connect(lambda _, idx=index: self.navigate_to_page(idx))
                layout.addWidget(button)

        layout.addStretch()  # Add stretch to push the status layout to the bottom


        add_chart_button = QPushButton("Add Chart")
        add_chart_button.setFixedHeight(70)
        add_chart_button.setStyleSheet("text-align:left; padding: 10px; margin-top: 10px;")
        add_chart_button.setIcon(QIcon("/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/addchart.png"))
        # layout.addWidget(add_chart_button)

        # Add status box at the bottom
        # status_layout = QHBoxLayout()

        # delete_button = QPushButton("Delete")
        # status_layout.addWidget(delete_button)

        # color_label = QLabel()
        # color_label.setFixedSize(100, 50)
        # color_label.setStyleSheet("background-color: blue;")
        

        # coordinates_label = QLabel("400, 200 to 700, 550")
        # status_layout.addWidget(coordinates_label)
        # status_layout.addWidget(color_label)

        layout.addLayout(result_details_layout)


class RegenerateOnCloseWindow(QWidget):
    def __init__(self, regenerate_callback):
        super().__init__()
        self.regenerate_callback = regenerate_callback

    def closeEvent(self, event):
        # Call the regenerate callback when the window is closed
        if self.regenerate_callback:
            self.regenerate_callback()
        super().closeEvent(event)