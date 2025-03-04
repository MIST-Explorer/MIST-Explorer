# from PyQt6.QtWidgets import QStyledItemDelegate, QComboBox, QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QFont
from PyQt6.QtCore import pyqtSlot, Qt, QRegularExpression

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
        
        # View management
        self.views = []  # List to hold views
        self.current_view_index = 0
        
        # Graph management
        self.graphs = []  # List of lists to hold graphs for each view
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
        self.back_button.clicked.connect(self.navigate_to_previous_view)
        self.next_button.clicked.connect(self.navigate_to_next_view)
        
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
        
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on current indices"""
        self.back_button.setEnabled(self.current_view_index > 0)
        self.next_button.setEnabled(self.current_view_index < len(self.views) - 1)

    def navigate_to_view(self, index):
        """Navigate to a specific view by index"""
        if not self.views or index < 0 or index >= len(self.views):
            return False
            
        # Update rubberband visibility
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(False)
            
        # Clear current content
        self.clear_scroll_content()
        
        # Set new content
        self.current_view_index = index
        self.scroll_layout.addWidget(self.views[index])
        
        # Update rubberband for new view
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(True)
            
        self.update_navigation_buttons()
        return True

    def navigate_to_next_view(self):
        """Navigate to the next view if available"""
        if self.current_view_index < len(self.views) - 1:
            self.navigate_to_view(self.current_view_index + 1)

    def navigate_to_previous_view(self):
        """Navigate to the previous view if available"""
        if self.current_view_index > 0:
            self.navigate_to_view(self.current_view_index - 1)

    def clear_scroll_content(self):
        """Clear all widgets from the scroll area"""
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def add_new_view(self):
        """Add a new view and make it current"""
        new_view = QWidget()
        self.views.append(new_view)
        self.graphs.append([])
        self.current_view_index = len(self.views) - 1
        self.current_graph_index = 0
        self.navigate_to_view(self.current_view_index)
        return new_view

    def delete_current_view(self):
        """Delete the current view and update navigation"""
        if not self.views:
            return False
            
        # Remove view and its associated data
        self.views.pop(self.current_view_index)
        self.graphs.pop(self.current_view_index)
        
        if self.rubberbands:
            self.rubberbands[self.current_view_index].hide()
            self.rubberbands.pop(self.current_view_index)
            
        if self.regions:
            self.regions.pop(self.current_view_index)
        
        # Update navigation
        if len(self.views) == 0:
            self.clear_scroll_content()
            self.current_view_index = 0
        else:
            new_index = max(0, self.current_view_index - 1)
            self.navigate_to_view(new_index)
            
        return True

    def add_graph_to_current_view(self, graph_widget):
        """Add a graph to the current view"""
        if not self.views:
            return False
            
        self.graphs[self.current_view_index].append(graph_widget)
        self.navigate_to_view(self.current_view_index)
        return True

    def get_current_graph(self):
        """Get the current graph from the current view"""
        if not self.views or not self.graphs[self.current_view_index]:
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
        self.views.append(result_widget)
        self.graphs.append([])
        self.navigate_to_view(len(self.views) - 1)
        
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
        
        # Create graph selection interface
        graph_selection = self.create_graph_selection_interface()
        
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
        delete_button.clicked.connect(self.delete_current_view)
        
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
        self.icon_list_page = IconListPage(
            self.icon_list, 
            self.show_icon_detail_page, 
            self.icon_paths, 
            None
        )
        self.icon_detail_page = IconDetailPage(
            self.show_icon_grid_page,
            self.open_in_new_window,
            self
        )
        
        self.stacked_widget.addWidget(self.icon_list_page)
        self.stacked_widget.addWidget(self.icon_detail_page)
        
        return self.stacked_widget
    
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
        
        # Convert polygon points to list of tuples
        poly_points = [(p.x(), p.y()) for p in region]
        
        # Get x,y coordinates from data
        points = data[['Global X', 'Global Y']].values
        
        # Filter data to only points inside polygon
        mask = [point_in_polygon(point, poly_points) for point in points]
        return data[mask]
    

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
    
    

    def handleComboBoxChanged(self, checked_items):
        """Handle changes in protein selection"""
        if not checked_items:
            QMessageBox.warning(
                self,
                "Alert",
                "You have nothing selected! Please select at least one protein."
            )
            return
            
        # Clear current graphs
        self.graphs[self.current_view_index] = []
        
        # # Get filtered data
        # if self.regions[self.current_view_index][0] == "rect":
        #     data = self.get_filtered_data(self.regions[self.current_view_index])
        # if self.regions[self.current_view_index][0] == "circle":
        #     data = self.get_circle_data(self.regions[self.current_view_index])
        # if self.regions[self.current_view_index][0] == "poly":
        #     data = self.get_poly_data(self.regions[self.current_view_index])

        data = data.drop(columns=list(set(data.columns[3:]) - set(checked_items)))
        
        # Regenerate graphs
        self.generate_analysis_graphs(self.regions[self.current_view_index])
        
        # Update view
        self.navigate_to_view(self.current_view_index)

    def show_icon_detail_page(self, index):
        print("show_icon_detail_page, index:", index)
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
        """Get a graph widget based on the index."""
        self.current_graph_index = index
        if not self.views or not self.graphs[self.current_view_index]:
            return QLabel("No graphs available")
            
        if index >= len(self.graphs[self.current_view_index]):
            return QLabel("Graph index out of range")
            
        graph = self.graphs[self.current_view_index][index]
        
        # Handle callable graphs (lazy loading)
        if callable(graph):
            graph = graph()
            self.graphs[self.current_view_index][index] = graph
        
        return graph

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
    def __init__(self, navigate_back, open_in_new_window, enclosing):
        super().__init__()
        self.navigate_back = navigate_back
        self.open_in_new_window = open_in_new_window
        self.enc = enclosing
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
        add_chart_button.clicked.connect(self.navigate_to_page)
        layout.addWidget(add_chart_button)
        
    def navigate_to_page(self):
        pass
        # Call the parent's custom plot method
        # self.parent().open_custom_plot_dialog()

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

        # layout.addLayout(result_details_layout)


class RegenerateOnCloseWindow(QWidget):
    def __init__(self, regenerate_callback):
        super().__init__()
        self.regenerate_callback = regenerate_callback

    def closeEvent(self, event):
        # Call the regenerate callback when the window is closed
        if self.regenerate_callback:
            self.regenerate_callback()
        super().closeEvent(event)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        
        self.highlighting_rules = []
        
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(120, 120, 250))
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            r'\bimport\b', r'\bfrom\b', r'\bdef\b', r'\bclass\b', r'\bfor\b', r'\bwhile\b',
            r'\bif\b', r'\belif\b', r'\belse\b', r'\btry\b', r'\bexcept\b', r'\bfinally\b',
            r'\breturn\b', r'\bpass\b', r'\bcontinue\b', r'\bbreak\b', r'\bin\b', r'\bas\b',
            r'\bglobal\b', r'\bwith\b', r'\braise\b', r'\bTrue\b', r'\bFalse\b', r'\bNone\b'
        ]
        for pattern in keywords:
            regex = QRegularExpression(pattern)
            self.highlighting_rules.append((regex, keyword_format))
        
        # Functions
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(40, 170, 40))
        function_regex = QRegularExpression(r'\b[A-Za-z0-9_]+(?=\()')
        self.highlighting_rules.append((function_regex, function_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(220, 120, 70))
        string_regex1 = QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"')
        string_regex2 = QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'")
        self.highlighting_rules.append((string_regex1, string_format))
        self.highlighting_rules.append((string_regex2, string_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(120, 120, 120))
        comment_format.setFontItalic(True)
        comment_regex = QRegularExpression(r'#[^\n]*')
        self.highlighting_rules.append((comment_regex, comment_format))
        
    def highlightBlock(self, text):
        for regex, format in self.highlighting_rules:
            match_iterator = regex.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class CustomPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Matplotlib Plot")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        
        # Default sample code
        sample_code = """# Sample code for a bar graph from cell_data.csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('/Users/clark/Desktop/wang/protein_visualization_app/assets/sample_data/cell_data.csv')

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Select first 5 rows and some protein columns for the bar chart
sample_data = data.iloc[:5, 3:8]

# Transpose so proteins become the index
sample_data = sample_data.transpose()

# Create the bar chart
sample_data.plot(kind='bar', ax=ax)

# Customize the plot
ax.set_title('Protein Expression Levels in First 5 Cells')
ax.set_xlabel('Proteins')
ax.set_ylabel('Expression Level')
ax.legend(title='Cell ID', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Return the figure for display
fig  # This figure will be displayed
"""
        
        self.code_edit.setPlainText(sample_code)
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Code editor
        code_group = QGroupBox("Python Code:")
        code_layout = QVBoxLayout()
        
        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont("Courier New", 10))
        self.code_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        code_layout.addWidget(self.code_edit)
        
        # Setup syntax highlighter
        self.highlighter = PythonSyntaxHighlighter(self.code_edit.document())
        
        code_group.setLayout(code_layout)
        
        # Output area
        output_group = QGroupBox("Output:")
        output_layout = QVBoxLayout()
        
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier New", 10))
        self.output_text.setMaximumHeight(100)
        output_layout.addWidget(self.output_text)
        
        output_group.setLayout(output_layout)
        
        # Plot area
        plot_group = QGroupBox("Plot Preview:")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        plot_group.setLayout(plot_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Code")
        self.run_button.clicked.connect(self.run_code)
        
        self.insert_button = QPushButton("Insert Plot")
        self.insert_button.clicked.connect(self.insert_plot)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.insert_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all components to main layout
        main_layout.addWidget(code_group, 3)
        main_layout.addWidget(output_group, 1)
        main_layout.addWidget(plot_group, 3)
        main_layout.addLayout(button_layout)
        
    def run_code(self):
        """Execute the matplotlib code and display the resulting figure"""
        # Clear previous output
        self.output_text.clear()
        self.figure.clear()
        
        # Get the code from the text editor
        code = self.code_edit.toPlainText()
        
        # Create a namespace for execution
        namespace = {
            'plt': plt,
            'pd': pd,
            'np': np,
            'Figure': Figure,
            'sns': sns
        }
        
        # Capture stdout
        output_buffer = io.StringIO()
        
        try:
            # Execute the code and capture output
            with redirect_stdout(output_buffer):
                exec(code, namespace)
            
            # Check if a figure object was created or used from plt
            if 'fig' in namespace:
                fig = namespace['fig']
                
                # Clear our figure and copy the content from the generated figure
                self.figure.clear()
                for ax in fig.get_axes():
                    # Copy the Axes object to our figure
                    new_ax = self.figure.add_axes(ax.get_position())
                    # Copy the content
                    for line in ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(), 
                                     color=line.get_color(),
                                     linestyle=line.get_linestyle(),
                                     marker=line.get_marker())
                    
                    # Copy other properties
                    new_ax.set_title(ax.get_title())
                    new_ax.set_xlabel(ax.get_xlabel())
                    new_ax.set_ylabel(ax.get_ylabel())
                    
                    # Copy legend if exists
                    if ax.get_legend():
                        new_ax.legend()
                    
                    # Copy bars if this is a bar chart
                    for patch in ax.patches:
                        new_ax.add_patch(patch)
                
                # Copy the tight layout if used
                self.figure.tight_layout()
                self.canvas.draw()
                
                # Store the generated figure for later use
                self.generated_figure = fig
                
                # Display success message
                self.output_text.appendPlainText("Code executed successfully!")
            else:
                # If no figure was found, show a message
                self.output_text.appendPlainText("No figure object named 'fig' found in the code. Make sure your code creates a matplotlib figure named 'fig'.")
        
        except Exception as e:
            # Handle exceptions and display error message
            self.output_text.appendPlainText(f"Error: {str(e)}")
            self.output_text.appendPlainText(traceback.format_exc())
        
        # Display any stdout output
        stdout_output = output_buffer.getvalue()
        if stdout_output:
            self.output_text.appendPlainText("\nOutput:")
            self.output_text.appendPlainText(stdout_output)
    
    def insert_plot(self):
        """Insert the current plot into the parent AnalysisTab"""
        if hasattr(self, 'generated_figure'):
            # Create a widget to hold the canvas
            plot_widget = QWidget()
            layout = QVBoxLayout(plot_widget)
            
            # Create a new canvas with the generated figure
            canvas = FigureCanvas(self.generated_figure)
            layout.addWidget(canvas)
            
            # Save reference to the figure for saving
            plot_widget.figure = self.generated_figure
            
            # Add the widget to the parent's view
            parent = self.parent()
            parent.add_graph_to_current_view(plot_widget)
            
            # Close the dialog
            self.accept()
        else:
            QMessageBox.warning(self, "No Plot", "Please run your code first to generate a plot.")