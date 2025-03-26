"""
The `AnalysisTab` class is a QWidget that manages multiple views and associated
graphs for data analysis, allowing navigation between views, selection of regions for analysis, and
creation of various types of graphs based on selected regions and protein data.

Features: 
    ROI management: 
        Here you will find the code that extracts the actual region of either the rect, circle, or poly lasso.
        The data is filtered and passed to the graphing modules (which are lazy loaded)
        Display of the ROIs (data about position, rubberband color, ) is managed here

    And a crutial feature, NAVIGATION between the ROIs:
        This code can be a little tricky. Essentially, we have the ROIS in self.rois which correspond to user selected regions of interest.
        (sometimes, earlier in development, we called an ROI a "view" so if you see this in an analysis context it may be an ROI!)
        For each ROI, there many be many graphs for that ROI in particular. 
        As such, we track two indicies: the current graph # and the current ROI #.
        The code to navigate between the ROIs is a little complex
"""


from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal, QPoint
from PyQt6.QtGui import QColor, QIcon, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QFont
from PyQt6.QtCore import pyqtSlot, Qt, QRegularExpression

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QStackedWidget, QHBoxLayout
)

from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt
import sys


import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import traceback
from contextlib import redirect_stdout
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.analysis.graphing.ZScoreHeatmapWindow import ZScoreHeatmapWindow 
from ui.analysis.graphing.SpatialHeatmapUpdated import HeatmapWindow
from ui.analysis.graphing.CellDensityPlot import CellDensityPlot
from ui.analysis.graphing.DistributionViewer import DistributionViewer
from ui.analysis.graphing.PieChartCanvas import PieChartCanvas
from ui.analysis.graphing.delete_later import UMAPVisualizer


class AnalysisTab(QWidget):

    def __init__(self, pixmap_label, enc):
        super().__init__()
        self.enc = enc

        # roi management
        self.rois = []  # List to hold views
        self.current_view_index = 0

        # Graph management
        self.graphs = []  # List of lists to hold graphs for each roi
        self.current_graph_index = 0

        # Selection management
        self.rubberbands = []
        self.regions = []
        
        # Track open windows
        self.windows = []

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Navigation controls
        nav_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Plot")
        self.back_button = QPushButton("< Back")
        self.next_button = QPushButton("Next >")

        self.save_button.clicked.connect(self.save_current_plot)
        
        self.back_button.clicked.connect(self.navigate_to_previous_roi)
        self.next_button.clicked.connect(self.navigate_to_next_roi)

        nav_layout.addWidget(self.save_button)
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.next_button) 

        # Content area
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)

        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)
        
        # Create floating selection buttons
        # self.create_floating_buttons()
        
        self.update_navigation_buttons()

    def create_floating_buttons(self):
        """
        These are the buttons that float above the screen that allow the user to select a circle/rect/polygon lasso for ROI.
        """
        # Create a container widget for the buttons
        self.floating_container = QWidget(self)
        self.floating_container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Create horizontal layout for the buttons
        button_layout = QHBoxLayout(self.floating_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        
        # Create the selection buttons
        self.rect_button = QPushButton()
        self.circle_button = QPushButton()
        self.poly_button = QPushButton()
        
        # Set button sizes and styles
        for button in [self.rect_button, self.circle_button, self.poly_button]:
            button.setFixedSize(40, 40)
            button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.9);
                }
                QPushButton:pressed {
                    background-color: rgba(200, 200, 200, 0.9);
                }
            """)
        
        # Set icons for the buttons
        self.rect_button.setIcon(QIcon("ui/graphing/icons/rectangle.png"))
        self.circle_button.setIcon(QIcon("ui/graphing/icons/circle.png"))
        self.poly_button.setIcon(QIcon("ui/graphing/icons/polygon.png"))
        
        # Connect button signals
        self.rect_button.clicked.connect(lambda: self.enc.view_tab.set_selection_mode("rect"))
        self.circle_button.clicked.connect(lambda: self.enc.view_tab.set_selection_mode("circle"))
        self.poly_button.clicked.connect(lambda: self.enc.view_tab.set_selection_mode("poly"))
        
        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)
        
        # Position the container at the top of the scroll area
        self.update_floating_buttons_position()

    def update_floating_buttons_position(self):
        """Update the position of the floating buttons"""
        if hasattr(self, 'floating_container'):
            # Position at the top of the scroll area
            pos = self.scroll_area.mapTo(self, QPoint(10, 10))
            self.floating_container.move(pos)

    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_floating_buttons_position()

    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on current indices"""
        self.back_button.setEnabled(self.current_view_index > 0)
        self.next_button.setEnabled(self.current_view_index < len(self.rois) - 1)

    def navigate_to_roi(self, index):
        """
        The function `navigate_to_roi` navigates to a specific roi by index, updating the displayed
        content and rubberband visibility accordingly.
        
        :param index: The `index` parameter in the `navigate_to_roi` method represents the position of
        the roi that you want to navigate to within a list of views. It is used to determine which roi
        should be displayed based on its index in the list of views
        :return: The function `navigate_to_roi` returns a boolean value - `True` if the navigation to
        the specific roi by index was successful, and `False` if the views list is empty or the index
        is out of bounds.
        """
        if not self.rois or index < 0 or index >= len(self.rois):
            return False
            
        # Update rubberband visibility
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(False)
            
        # Clear current content
        self.clear_scroll_content()

        # Add the roi's widget
        self.scroll_layout.addWidget(self.rois[index])
        
        # Update rubberband for new roi
        if self.rubberbands:
            for i in range(len(self.rubberbands)):
                self.rubberbands[i].set_filled(False)
            self.rubberbands[index].set_filled(True)
        
        # Update current roi index
        self.current_view_index = index
            
        self.update_navigation_buttons()
        return True

    def navigate_to_next_roi(self):
        """Navigate to the next roi if available"""
        if self.current_view_index < len(self.rois) - 1:
            self.current_view_index += 1
            self.navigate_to_roi(self.current_view_index)

    def navigate_to_previous_roi(self):
        """Navigate to the previous roi if available"""
        if self.current_view_index > 0:
            self.current_view_index -= 1
            self.navigate_to_roi(self.current_view_index)

    def clear_scroll_content(self):
        """Clear all widgets from the scroll area"""
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def delete_current_view(self):            
        # Remove roi and its associated data
        self.rois.pop(self.current_view_index)
        self.graphs.pop(self.current_view_index)
        
        if self.rubberbands:
            self.rubberbands[self.current_view_index].hide()
            self.rubberbands.pop(self.current_view_index)
            
        if self.regions:
            self.regions.pop(self.current_view_index)
        
        # Update navigation
        if len(self.rois) == 0:
            self.clear_scroll_content()
            self.current_view_index = 0
        else:
            new_index = max(0, self.current_view_index - 1)
            self.navigate_to_roi(new_index)
            
        return True

    def add_graph_to_current_view(self, graph_widget):
        """Add a graph to the current roi"""
        if not self.rois:
            return False
            
        self.graphs[self.current_view_index].append(graph_widget)
        self.navigate_to_roi(self.current_view_index)
        return True

    def get_current_graph(self):
        """Get the current graph from the current roi"""
        if not self.rois or not self.graphs[self.current_view_index]:
            return None
            
        graph = self.graphs[self.current_view_index][self.current_graph_index]
        
        # Handle callable graphs (lazy loading)
        if callable(graph):
            graph = graph()
            self.graphs[self.current_view_index][self.current_graph_index] = graph
            
        return graph

    def save_current_plot(self):
        """Save the current plot to a file"""
        current_graph = self.get_current_graph()
        if not current_graph:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Plot", 
            "", 
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path and hasattr(current_graph, 'figure'):
            current_graph.figure.savefig(file_path)

    def analyze_region(self, rubberband, region):
        """Analyze a selected region and create corresponding visualizations"""
        # Handle previous rubberband
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(False)
        
        # Store selection data
        self.rubberbands.append(rubberband)
        self.regions.append(region)

        # Create result widget
        result_widget = self.create_analysis_result_widget(rubberband, region)
        
        # Add to views and navigate
        self.rois.append(result_widget)
        self.graphs.append([])
        self.current_view_index = len(self.rois) - 1
        self.navigate_to_roi(self.current_view_index)
        
        # Generate and add graphs
        self.generate_analysis_graphs(region)
        
        # Update rubberband
        self.rubberbands[-1].set_filled(True)

    def create_analysis_result_widget(self, rubberband, region):
        """Create the widget to display analysis results"""
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)  
        
        # Create controls
        controls_layout = self.create_analysis_controls(rubberband, region)
        
        # Create graph selection interface for this specific roi
        graph_selection = self.create_graph_selection_interface()
        
        # Store the graph interface widgets in a dictionary keyed by roi index
        view_index = len(self.rois)  # This will be the index of the new roi
        if not hasattr(self, 'view_graph_interfaces'):
            self.view_graph_interfaces = {}
        self.view_graph_interfaces[view_index] = {
            'stacked_widget': graph_selection,
            'icon_list_page': self.icon_list_page,
            'icon_detail_page': self.icon_detail_page
        }
        
        # Add to layout
        result_layout.addLayout(controls_layout)
        result_layout.addWidget(graph_selection)
        
        return result_widget

    def create_analysis_controls(self, rubberband, region):
        """Create the control panel for analysis results"""
        controls_layout = QHBoxLayout()
        
        # Add protein selection
        self.multiComboBox = MultiComboBox()
        self.multiComboBox.addItem("Select All")
        self.multiComboBox.addItem("Deselect All")
        
        data = self.enc.view_tab.load_df()
        self.multiComboBox.addItems(data.columns[3:])
        
        for i in range(len(data.columns[3:])):
            self.multiComboBox.model().item(i).setCheckState(Qt.CheckState.Checked)

        # Add buttons
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(
            lambda: self.handleComboBoxChanged(self.multiComboBox.get_checked_items())
        )

        
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.delete_current_view)
        
        # Add region info
        bounds_label = QLabel(f"Bounds: {region}")
        
        # Add color indicator
        color_label = QLabel()
        color_label.setFixedSize(100, 50)
        rgb = QColor(rubberband.color).getRgb()[:3]
        color_label.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
        )
        
        # Arrange controls
        button_layout = QVBoxLayout()
        button_layout.addWidget(bounds_label)
        button_layout.addWidget(delete_button)
        
        controls_layout.addWidget(QWidget().setLayout(button_layout))
        controls_layout.addWidget(self.multiComboBox)
        controls_layout.addWidget(apply_button)
        # controls_layout.addWidget(delete_button)
        controls_layout.addWidget(color_label)
        
        return controls_layout

    def create_graph_selection_interface(self):
        """Create the interface for selecting different graph types"""
        self.icon_list = [
            "Boxplot", "Z-Scores Heatmap", "Spatial Heatmap",
            "Pi Chart", "Histogram", "UMAP" 
        ]

        self.icon_paths = [
            "ui/graphing/icons/linechart.png",
            "ui/graphing/icons/heatmap.png",
            "ui/graphing/icons/heatmap.png",
            "ui/graphing/icons/piechart.png",
            "ui/graphing/icons/barchart.png",
            "ui/graphing/icons/scatter.png",
        ]
        
        self.stacked_widget = QStackedWidget()
        
        self.icon_list_page = GraphsList(
            icon_list=self.icon_list, 
            navigate_to_page=self.show_icon_detail_page, 
            icon_paths=self.icon_paths, 
            result_details_layout=None
        )
        self.icon_detail_page = GraphInDetail(
            navigate_back=self.show_icon_grid_page,
            open_in_new_window=self.open_in_new_window,
            parent=self
        )

        self.stacked_widget.addWidget(self.icon_list_page)
        self.stacked_widget.addWidget(self.icon_detail_page)

        return self.stacked_widget
    

    

    def generate_analysis_graphs(self, region):
        # Get filtered data
        if self.regions[self.current_view_index][0] == "rect":
            data = self.get_rect_data(region[1])
        if self.regions[self.current_view_index][0] == "circle":
            data = self.get_circle_data(region[1])
        if self.regions[self.current_view_index][0] == "poly":
            data = self.get_poly_data(region[1])
        
        # Create and add graphs
        box_plot = self.create_box_plot(data)
        self.add_graph_to_current_view(box_plot)
        
        graph_generators = [
            lambda: ZScoreHeatmapWindow(data),
            lambda: HeatmapWindow(data),
            lambda: PieChartCanvas(data),
            lambda: DistributionViewer(data),
            lambda: UMAPVisualizer()
        ]
        
        for generator in graph_generators:
            self.add_graph_to_current_view(generator)


    def create_box_plot(self, data):
        """Create a box plot widget"""
        result_widget = QWidget()
        layout = QVBoxLayout(result_widget)
        
        filtered_data = data.iloc[:, 3:]
    
        fig, ax = plt.subplots(figsize=(12, 8))
        filtered_data.boxplot(ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('Protein')
        ax.set_ylabel('Expression Level')
        ax.set_title('Protein Expression Box Plot')
        plt.subplots_adjust(bottom=0.3)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        result_widget.figure = fig
        
        return result_widget


    def get_rect_data(self, region):
        """Get data filtered by the selected region"""
        data = self.enc.view_tab.load_df()
        x_min, y_min, x_max, y_max = [i * 4 for i in region]
        
        return data[
            (data['Global X'] >= x_min) & 
            (data['Global X'] <= x_max) &
            (data['Global Y'] >= y_min) & 
            (data['Global Y'] <= y_max)
        ]

    def get_circle_data(self, region):
        """Get data filtered by the selected circular/oval region"""
        data = self.enc.view_tab.load_df()
        x_min, y_min, x_max, y_max = [i * 4 for i in region]

        # Calculate circle center and radius
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius_x = (x_max - x_min) / 2
        radius_y = (y_max - y_min) / 2

        # Apply elliptical equation filter
        return data[
            ((data['Global X'] - center_x) ** 2 / radius_x ** 2) +
            ((data['Global Y'] - center_y) ** 2 / radius_y ** 2) <= 1
        ]
     
    def get_poly_data(self, region):
        """Get data filtered by the selected polygon region using ray casting algorithm"""
        data = self.enc.view_tab.load_df()
        
        def point_in_polygon(point, polygon):
            """Check if a point is inside a polygon using ray casting algorithm"""
            x, y = point
            inside = False
            
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                
                # Check if point is between the y-coordinates of the edge
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                    
            return inside

        # Convert polygon points to list of tuples and scale them to match data coordinates
        # The data coordinates are 4x the widget coordinates
        poly_points = [(p.x() * 4, p.y() * 4) for p in region]
        
        # Get x,y coordinates from data
        points = data[['Global X', 'Global Y']].values
        
        # Filter data to only points inside polygon
        mask = [point_in_polygon(point, poly_points) for point in points]
        
        return data[mask]

    def handleComboBoxChanged(self, checked_items):
        """Handle changes in protein selection. """
        if not checked_items:
            QMessageBox.warning(
                self,
                "Alert",
                "You have nothing selected! Please select at least one protein."
            )
            return
            
        # Clear current graphs
        self.graphs[self.current_view_index] = []
        
        # Get filtered data
        region = self.regions[self.current_view_index]
        if self.regions[self.current_view_index][0] == "rect":
            data = self.get_rect_data(region)
        if self.regions[self.current_view_index][0] == "circle":
            data = self.get_circle_data(region)
        if self.regions[self.current_view_index][0] == "poly":
            data = self.get_poly_data(region)

        data = data.drop(columns=list(set(data.columns[3:]) - set(checked_items)))
        
        # Regenerate graphs
        self.generate_analysis_graphs(self.regions[self.current_view_index])
        
        # Update roi
        self.navigate_to_roi(self.current_view_index)

    def show_icon_detail_page(self, index):
        print("show_icon_detail_page1,", "current roi:", self.current_view_index, "graph:", index)
        
        # Get the graph interface for the current roi
        if self.current_view_index not in self.view_graph_interfaces:
            print("Error: No graph interface found for current roi")
            return
            
        interface = self.view_graph_interfaces[self.current_view_index]
        
        # Update the graph index and display
        self.current_graph_index = index
        interface['icon_detail_page'].set_icon_index(index)
        interface['stacked_widget'].setCurrentWidget(interface['icon_detail_page'])

    def show_icon_grid_page(self):
        # Get the graph interface for the current roi
        if self.current_view_index not in self.view_graph_interfaces:
            print("Error: No graph interface found for current roi")
            return
            
        interface = self.view_graph_interfaces[self.current_view_index]
        interface['stacked_widget'].setCurrentWidget(interface['icon_list_page'])

    def open_in_new_window(self):
        """
        Allows us the ability to pop out a graph and view it in a new window.  
        """
        # Create a new window to display the current graph
        new_window = RegenerateOnCloseWindow(
            regenerate_callback=self._on_new_window_closed
        )
        new_window.setWindowTitle("Icon Detail - New Window")
        layout = QVBoxLayout()

        # Retrieve the current graph widget from the encoder
        index = self.current_graph_index
        widget = self.get_current_graph()
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
        """
        As of writing this, this is the regenerate_callback called in open_in_new_window that the RegenerateOnCloseWindowCalls.
        """
        # When the new window is closed, regenerate the graph in the main window
        index = self.current_graph_index
        new_graph = self.get_current_graph()

        # Clear old content and update layout with the regenerated graph
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        self.icon_detail_page.content_layout.addWidget(new_graph)

    def get_graph(self, index):
        """
        The function `get_graph` retrieves a graph widget based on the provided index, handling lazy
        loading for callable graphs.
        
        :param index: The `index` parameter in the `get_graph` method is used to specify which graph
        widget to retrieve from the list of graphs. It is an integer value that represents the position
        of the graph widget within the list of graphs associated with the current view index
        :return: The `get_graph` method returns a graph widget based on the provided index. If there are
        no regions of interest (`rois`) or if there are no graphs available for the current view index,
        it returns a QLabel widget with the message "No graphs available". If the index is out of range
        for the graphs list, it returns a QLabel widget with the message "Graph index out of range".
        """
        """Get a graph widget based on the index."""
        self.current_graph_index = index
        if not self.rois or not self.graphs[self.current_view_index]:
            return QLabel("No graphs available")
            
        if index >= len(self.graphs[self.current_view_index]):
            return QLabel("Graph index out of range")
            
        graph = self.graphs[self.current_view_index][index]
        
        # Handle callable graphs (lazy loading)
        if callable(graph):
            graph = graph()
            self.graphs[self.current_view_index][index] = graph
        
        return graph


class MultiComboBox(QComboBox):
    
    itemsCheckedChanged = pyqtSignal(list)  # Signal to emit the list of checked items

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))

        # Connect to the dataChanged signal to update the text and emit our signal
        self.model().dataChanged.connect(self.onItemStateChanged)

    def addItem(self, text: str):
        """
        Add a single choice to the MCB.

        Args:
            text (str): the items we want to be made available as choices in our dropdown.
        """
        item = QStandardItem()
        item.setText(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, items_list: list):
        """
         Useful for when we have to add a lot of items to a MCB, like populating all the proteins, for example.

        Args:
            items_list (list): the items we want to be made available as choices in our dropdown.
        """
        for text in items_list:
            self.addItem(text)

    def updateText(self):
        """
        The function `updateText` retrieves the text of checked items in a model and sets it as the text
        of a line edit widget, separated by commas.
        """
        selected_items = [self.model().item(i).text() for i in range(self.model().rowCount())
                          if self.model().item(i).checkState() == Qt.CheckState.Checked]
        self.lineEdit().setText(", ".join(selected_items))

    def onItemStateChanged(self):
        """
        The function `onItemStateChanged` updates displayed text and checks all items except "Deselect
        All" when "Select All" is checked, and deselects all items when "Deselect All" is checked.
        """
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
        """
        This function retrieves checked items from a model, excluding "Select All" and "Deselect All"
        items.
        :return: The function `get_checked_items` returns a list of items that are checked in the model,
        excluding the items "Select All" and "Deselect All".
        """
        items = [self.model().item(i).text() for i in range(self.model().rowCount())
                if self.model().item(i).checkState() == Qt.CheckState.Checked]
        
        return [item for item in items if item not in ["Select All", "Deselect All"]]   
    
    # stupid 
    def get_checked_items2(self):
        """
        Started as a result of some backwards compatability issue, not sure if this is still needed....
        """
        return [self.model().item(i).text() for i in range(self.model().rowCount())
                if self.model().item(i).checkState() == Qt.CheckState.Checked]
    

    from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QStackedWidget
)


class GraphInDetail(QWidget):
    """
    This is the actual pane that comes up when you select a graph. 
    """
    def __init__(self, navigate_back, open_in_new_window, parent):
        super().__init__()
        self.navigate_back = navigate_back
        self.open_in_new_window = open_in_new_window
        self.enc = parent
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

        # Add new content based on get_graph
        widget = self.enc.get_graph(index)
        widget.setSizePolicy(widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding)
        self.content_layout.addWidget(widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)


class GraphsList(QWidget):
    """
    The `GraphsList` class creates a widget displaying a list of icons with buttons that can be clicked to navigate to different graphs.
    This is mostly a UI class -- just plugs into stuff from the analysis tab.
    One per ROI.

    Args:
        QWidget (_type_): _description_
    """
    def __init__(self, icon_list, navigate_to_page, icon_paths, result_details_layout):
        super().__init__()
        self.icon_list = icon_list
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
                
                button.clicked.connect(lambda _, idx=index: navigate_to_page(idx))
                layout.addWidget(button)

        layout.addStretch()  # Add stretch to push the status layout to the bottom


        add_chart_button = QPushButton("Add Chart")
        add_chart_button.setFixedHeight(70)
        add_chart_button.setStyleSheet("text-align:left; padding: 10px; margin-top: 10px;")
        add_chart_button.setIcon(QIcon("/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/icons/addchart.png"))
        add_chart_button.clicked.connect(navigate_to_page)
        layout.addWidget(add_chart_button)
 
class RegenerateOnCloseWindow(QWidget):
    """
    Essentially, this is used for one very specific feature -- the ability to pop out a graph and view it in a new window. 

    Once that window is closed, the graph should return into the frame of the analysis tab.
    
    """
    def __init__(self, regenerate_callback):
        """
        Initalizes a new window.

        Args:
            regenerate_callback (func): called when the window is closed.
        """
        super().__init__()
        self.regenerate_callback = regenerate_callback

    def closeEvent(self, event):
        # Call the regenerate callback when the window is closed
        if self.regenerate_callback:
            self.regenerate_callback()
        super().closeEvent(event)


