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

import logging
import traceback
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QPoint, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (QColor, QCursor, QIcon, QStandardItem,
                         QStandardItemModel)
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QScrollArea,
                             QStackedWidget, QTabWidget, QToolTip, QVBoxLayout,
                             QWidget, QDialog, QCheckBox, QSlider)


from core.dataframe_utils import get_marker_columns
from ui.analysis.graphing.DistributionViewer import DistributionViewer
from ui.analysis.graphing.PieChartCanvas import PieChartCanvas
from ui.analysis.graphing.SpatialAutocorrelationWindow import \
    SpatialAutocorrelationWindow
from ui.analysis.graphing.SpatialHeatmapUpdated import HeatmapWindow
from utils import resource_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ui.app import MainWindow


class AnalysisTab(QWidget):
    def __init__(self, pixmap_label, enc: "MainWindow"):
        super().__init__()
        self.enc = enc

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.roi_view = ROIAnalysisView(pixmap_label, enc)
        self.full_view = FullImageAnalysisView(pixmap_label, enc)

        self.tabs.addTab(self.roi_view, "ROI Analysis")
        self.tabs.addTab(self.full_view, "Full Image Analysis")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def analyze_region(self, rubberband, region):
        # Switch to ROI tab and delegate
        self.tabs.setCurrentWidget(self.roi_view)
        return self.roi_view.analyze_region(rubberband, region)

    def analyze_poly_region(self, rubberband, region):
        self.tabs.setCurrentWidget(self.roi_view)
        return self.roi_view.analyze_poly_region(rubberband, region)

    def update_roi_region(self, rubberband, region):
        """Update existing ROI region"""
        self.tabs.setCurrentWidget(self.roi_view)
        return self.roi_view.update_roi_region(rubberband, region)

    def reset_rois(self):
        """Reset all ROIs in the analysis tab."""
        self.roi_view.reset_rois()

    def on_tab_changed(self, index):
        # Index 0 is ROI Analysis, 1 is Full Image
        # Button visibility is now controlled by the main tab change handler
        # This method is kept for potential future sub-tab specific logic
        pass

    def hideEvent(self, event):
        # Button visibility is now controlled by the main tab change handler
        super().hideEvent(event)

    def showEvent(self, event):
        # Button visibility is now controlled by the main tab change handler
        super().showEvent(event)


class ROIAnalysisView(QWidget):
    def __init__(self, pixmap_label, enc: "MainWindow"):
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
        self.highlight_checkboxes = []
        self.opacity_sliders = []

        # Track open windows
        self.windows = []

        # Graph interfaces list matching the roi index
        self.view_graph_interfaces = []

        self.columns = []

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Navigation controls
        nav_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Plot")
        self.export_button = QPushButton("Export ROI Data")
        self.back_button = QPushButton("< Back")
        self.next_button = QPushButton("Next >")

        self.save_button.clicked.connect(self.save_current_plot)
        self.export_button.clicked.connect(self.export_current_roi_data)

        self.back_button.clicked.connect(self.navigate_to_previous_roi)
        self.next_button.clicked.connect(self.navigate_to_next_roi)

        nav_layout.addWidget(self.save_button)
        nav_layout.addWidget(self.export_button)
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
        self.floating_container.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        # Create horizontal layout for the buttons
        button_layout = QHBoxLayout(self.floating_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)

        # Create the selection buttons
        self.rect_button = QPushButton()
        self.circle_button = QPushButton()
        self.poly_button = QPushButton()

        # Set button sizes and styles
        from ui.theme import ThemeManager
        tc = ThemeManager.instance().get_current()
        for button in [self.rect_button, self.circle_button, self.poly_button]:
            button.setFixedSize(40, 40)
            button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {tc["bg_hover"]};
                    border: 1px solid {tc["border"]};
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {tc["bg_pressed"]};
                }}
                QPushButton:pressed {{
                    background-color: {tc["accent_dim"]};
                }}
            """
            )

        # Set icons for the buttons
        self.rect_button.setIcon(QIcon(resource_path("assets/icons/rectangle.png")))
        self.circle_button.setIcon(QIcon(resource_path("assets/icons/circle.png")))
        self.poly_button.setIcon(QIcon(resource_path("assets/icons/poly.png")))

        # Connect button signals
        self.rect_button.clicked.connect(
            lambda: self.enc.view_tab.set_selection_mode("rect")
        )
        self.circle_button.clicked.connect(
            lambda: self.enc.view_tab.set_selection_mode("circle")
        )
        self.poly_button.clicked.connect(
            lambda: self.enc.view_tab.set_selection_mode("poly")
        )

        # Add buttons to layout
        button_layout.addWidget(self.rect_button)
        button_layout.addWidget(self.circle_button)
        button_layout.addWidget(self.poly_button)

        # Position the container at the top of the scroll area
        self.update_floating_buttons_position()

    def update_floating_buttons_position(self):
        """Update the position of the floating buttons"""
        if hasattr(self, "floating_container"):
            # Position at the top of the scroll area
            pos = self.scroll_area.mapTo(self, QPoint(10, 10))
            self.floating_container.move(pos)

    def resizeEvent(self, event):
        """Handle resize events to update floating buttons position"""
        super().resizeEvent(event)
        self.update_floating_buttons_position()

    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on current indices"""
        has_rois = len(self.rois) > 0
        self.save_button.setEnabled(has_rois)
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(has_rois)
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
            if self.current_view_index < len(self.rubberbands):
                self.rubberbands[self.current_view_index].set_filled(False)
            else:
                logger.warning("current_view_index out of bounds for rubberbands")
                self.current_view_index = len(self.rubberbands) - 1
                if self.current_view_index < 0:
                    self.current_view_index = 0

        # Clear current content
        self.clear_scroll_content()

        # Add the roi's widget
        self.scroll_layout.addWidget(self.rois[index])

        # Update current roi index
        self.current_view_index = index

        # Update rubberband for new roi
        if self.rubberbands:
            for i in range(len(self.rubberbands)):
                self.rubberbands[i].set_filled(False)
            self.update_cell_highlighting()

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
        if hasattr(self, "highlight_checkboxes") and self.current_view_index < len(self.highlight_checkboxes):
            self.highlight_checkboxes.pop(self.current_view_index)
        if hasattr(self, "opacity_sliders") and self.current_view_index < len(self.opacity_sliders):
            self.opacity_sliders.pop(self.current_view_index)

        if self.rubberbands:
            rb = self.rubberbands[self.current_view_index]
            rb.hide()
            scene = rb.scene()
            if scene:
                scene.removeItem(rb)

            # Remove from canvas tracking lists
            if hasattr(self.enc, "canvas") and self.enc.canvas:
                canvas = self.enc.canvas
                if rb in canvas.rubber_bands:
                    canvas.rubber_bands.remove(rb)
                    if hasattr(rb, "color") and rb.color in canvas.rubber_band_colors:
                        canvas.rubber_band_colors.remove(rb.color)
                if rb in canvas.polygons:
                    canvas.polygons.remove(rb)
                    if hasattr(rb, "color"):
                        color_rgb = rb.color.getRgb()[:3]
                        if color_rgb in canvas.polygon_colors:
                            canvas.polygon_colors.remove(color_rgb)
            self.rubberbands.pop(self.current_view_index)

        if self.regions:
            self.regions.pop(self.current_view_index)

        if hasattr(self, "view_graph_interfaces") and isinstance(self.view_graph_interfaces, list):
            if self.current_view_index < len(self.view_graph_interfaces):
                self.view_graph_interfaces.pop(self.current_view_index)

        # Update navigation
        if len(self.rois) == 0:
            self.clear_scroll_content()
            self.current_view_index = 0
            self.update_navigation_buttons()
            if hasattr(self.enc, "canvas") and self.enc.canvas:
                self.enc.canvas.set_roi_cell_highlight(None)
        else:
            new_index = max(0, self.current_view_index - 1)
            self.navigate_to_roi(new_index)

        return True

    def show_roi_color_dialog(self, rubberband, color_button):
        """Prompt user to change the color of the selected ROI"""
        from PyQt6.QtGui import QPen, QBrush
        import pyqtgraph as pg
        from ui.view_tab import ColorDialog, color_dict
        from ui.lassos.PolyLasso import PolyLasso
        from ui.lassos.RectLasso import RectLasso
        from ui.lassos.CircleLasso import CircleLasso

        dialog = ColorDialog(color_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_color_name = dialog.get_selected_color_name()
            if selected_color_name:
                rgb = color_dict[selected_color_name]
                
                # Update the color button stylesheet
                color_button.setStyleSheet(
                    f"QPushButton {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #888; border-radius: 4px; }}"
                    f"QPushButton:hover {{ border-color: #555; }}"
                )
                
                # Track old color
                old_qcolor = rubberband.color
                old_rgb = old_qcolor.getRgb()[:3] if hasattr(rubberband, "color") else None
                
                # Update the rubberband/ROI itself and its graphics properties
                if isinstance(rubberband, PolyLasso):
                    rubberband.line_color = QColor(*rgb)
                    alpha = rubberband.color.alpha()
                    rubberband.color = QColor(*rgb, alpha)
                    rubberband.setPen(QPen(rubberband.line_color, 2))
                    rubberband.setBrush(QBrush(rubberband.color))
                    rubberband.update()
                elif isinstance(rubberband, (RectLasso, CircleLasso)):
                    rubberband.color = QColor(*rgb)
                    rubberband.setPen(pg.mkPen(color=rgb, width=2))
                    rubberband.update()
                else:
                    rubberband.color = QColor(*rgb)
                
                # Update canvas color lists
                if hasattr(self.enc, "canvas") and self.enc.canvas:
                    canvas = self.enc.canvas
                    if old_qcolor in canvas.rubber_band_colors:
                        idx = canvas.rubber_band_colors.index(old_qcolor)
                        canvas.rubber_band_colors[idx] = rubberband.color
                    if old_rgb in canvas.polygon_colors:
                        idx = canvas.polygon_colors.index(old_rgb)
                        canvas.polygon_colors[idx] = tuple(rgb)
                
                # Update highlighted cell colors
                self.update_cell_highlighting()



    def reset_rois(self):
        """Clear all ROIs and associated graphs/rubberbands from the analysis tab."""
        # Close any popped-out windows
        for window in list(self.windows):
            try:
                window.close()
            except Exception:
                pass
        self.windows = []

        self.rois = []
        self.graphs = []
        self.rubberbands = []
        self.regions = []
        self.highlight_checkboxes = []
        self.opacity_sliders = []
        self.current_view_index = 0
        self.current_graph_index = 0
        if hasattr(self, "view_graph_interfaces"):
            self.view_graph_interfaces = {}
        self.clear_scroll_content()
        self.update_navigation_buttons()

        if hasattr(self.enc, "canvas") and self.enc.canvas:
            self.enc.canvas.set_roi_cell_highlight(None)

    def update_cell_highlighting(self):
        """Update cell highlighting on the canvas based on current ROI checkbox state"""
        if not hasattr(self.enc, "canvas") or not self.enc.canvas:
            return

        # First, clear any existing highlight layer
        self.enc.canvas.set_roi_cell_highlight(None)

        if not self.rois or self.current_view_index >= len(self.rois):
            return

        if self.current_view_index >= len(self.highlight_checkboxes):
            return

        checkbox = self.highlight_checkboxes[self.current_view_index]
        rubberband = self.rubberbands[self.current_view_index]

        # Get opacity from slider
        opacity = 0.3  # default
        if hasattr(self, "opacity_sliders") and self.current_view_index < len(self.opacity_sliders):
            slider = self.opacity_sliders[self.current_view_index]
            opacity = slider.value() / 100.0

        if hasattr(rubberband, "set_fill_opacity"):
            rubberband.set_fill_opacity(opacity)
        else:
            rubberband.fill_alpha = int(opacity * 255)

        if checkbox.isChecked():
            # Disable the highlight fill within ROI
            rubberband.set_filled(False)
            # Highlight just cells
            self.apply_cell_highlight_on_canvas()
        else:
            # Re-enable the highlight within ROI
            rubberband.set_filled(True)

    def apply_cell_highlight_on_canvas(self):
        """Generates a mask image for cell boundaries and sets it on the canvas."""
        if not hasattr(self.enc, "canvas") or not self.enc.canvas:
            return

        if self.current_view_index >= len(self.rubberbands):
            return

        view_tab = self.enc.view_tab
        if not hasattr(view_tab, "reduced_cell_img") or view_tab.reduced_cell_img is None or view_tab.reduced_cell_img.size == 0:
            return

        data = self.get_current_roi_data()
        if data is None or data.empty or "CellID" not in data.columns:
            return

        # Get unique cell IDs in the current ROI
        cell_ids = data["CellID"].dropna().unique().astype(int)
        if len(cell_ids) == 0:
            return

        reduced_cell_img = view_tab.reduced_cell_img
        h, w = reduced_cell_img.shape

        # Use the current ROI's color for the highlight
        rubberband = self.rubberbands[self.current_view_index]
        color = rubberband.color if hasattr(rubberband, "color") else QColor("yellow")
        r, g, b, _ = color.getRgb()

        # Get opacity from slider
        opacity = 0.3  # default
        if hasattr(self, "opacity_sliders") and self.current_view_index < len(self.opacity_sliders):
            slider = self.opacity_sliders[self.current_view_index]
            opacity = slider.value() / 100.0
        alpha = int(opacity * 255)

        # Build lookup table for fast masking
        max_id = reduced_cell_img.max()
        lut = np.zeros(max_id + 1, dtype=bool)
        valid_cell_ids = cell_ids[cell_ids <= max_id]
        lut[valid_cell_ids] = True

        mask = lut[reduced_cell_img]

        # Generate or retrieve cached cell boundary mask
        if not hasattr(view_tab, "cell_boundary_mask") or view_tab.cell_boundary_mask is None or view_tab.cell_boundary_mask.shape != reduced_cell_img.shape:
            img = reduced_cell_img
            boundary_mask = np.zeros_like(img, dtype=bool)
            boundary_mask[:-1, :] |= (img[:-1, :] != img[1:, :])
            boundary_mask[1:, :] |= (img[1:, :] != img[:-1, :])
            boundary_mask[:, :-1] |= (img[:, :-1] != img[:, 1:])
            boundary_mask[:, 1:] |= (img[:, 1:] != img[:, :-1])
            boundary_mask &= (img > 0)
            view_tab.cell_boundary_mask = boundary_mask
        else:
            boundary_mask = view_tab.cell_boundary_mask

        # Highlight only boundary pixels of cells inside the ROI
        highlight_mask = mask & boundary_mask

        # Construct RGBA overlay image
        # Using a bright highlight color with custom opacity
        highlight_img = np.zeros((h, w, 4), dtype=np.uint8)
        highlight_img[highlight_mask] = [r, g, b, alpha]

        self.enc.canvas.set_roi_cell_highlight(highlight_img)

    def handle_opacity_change(self, value, label):
        """Handle opacity slider change."""
        opacity = value / 100.0
        label.setText(f"Opacity: {int(value)}%")
        
        # Update current rubberband fill opacity
        if self.rubberbands and self.current_view_index < len(self.rubberbands):
            rubberband = self.rubberbands[self.current_view_index]
            if hasattr(rubberband, "set_fill_opacity"):
                rubberband.set_fill_opacity(opacity)
            else:
                rubberband.fill_alpha = int(opacity * 255)
                rubberband.update()
                
        # Re-trigger update of cell highlighting / rubberband fill to redraw immediately
        self.update_cell_highlighting()

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
            self, "Save Plot", "", "PNG Files (*.png);;All Files (*)"
        )

        if file_path and hasattr(current_graph, "figure"):
            current_graph.figure.savefig(file_path)

    def export_current_roi_data(self):
        """Export the current ROI data to a CSV or XLSX file"""
        if not self.rois:
            QMessageBox.warning(self, "No Data", "No regions of interest (ROI) to export.")
            return

        data = self.get_current_roi_data()
        if data is None or data.empty:
            QMessageBox.warning(self, "No Data", "The selected ROI contains no data points.")
            return

        # Include metadata columns if they exist, plus only the selected proteins
        export_cols = [col for col in ["CellID", "Global X", "Global Y"] if col in data.columns]
        export_cols += [col for col in self.columns if col in data.columns and col not in export_cols]
        export_data = data[export_cols]

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export ROI Data",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".xlsx") or "Excel" in selected_filter:
                if not file_path.endswith(".xlsx"):
                    file_path += ".xlsx"
                export_data.to_excel(file_path, index=False)
            else:
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                export_data.to_csv(file_path, index=False)

            QMessageBox.information(
                self,
                "Export Success",
                f"Successfully exported ROI data to:\n{file_path}"
            )
        except Exception as e:
            logger.exception("EXPORT_ROI_FAIL")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export ROI data:\n{str(e)}"
            )

    def analyze_region(self, rubberband, region):
        """Analyze a selected region and create corresponding visualizations"""
        # Handle previous rubberband

        # Store selection data
        region = (region[0], tuple(int(i) for i in region[1]))
        assert len(region[1]) == 4, "invalid region definition"
        self.rubberbands.append(rubberband)
        self.regions.append(region)

        # Create result widget
        try:
            result_widget = self.create_analysis_result_widget(rubberband, region)
        except ValueError as e:
            self.rubberbands.pop()  # Remove last rubberband on error
            self.regions.pop()  # Remove last region on error
            QMessageBox.critical(
                self,
                "Error",
                e.args[0] if e.args else "An error occurred during analysis.",
            )
            return False
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(False)
        # Add to views and navigate
        self.rois.append(result_widget)
        self.graphs.append([])
        self.current_view_index = len(self.rois) - 1
        self.navigate_to_roi(self.current_view_index)

        # Generate and add graphs
        self.generate_analysis_graphs(region)

        # Update rubberband
        self.rubberbands[-1].set_filled(True)
        return True

    def analyze_poly_region(self, rubberband, region):
        """Analyze a polygon-selected region and create corresponding visualizations"""
        self.rubberbands.append(rubberband)
        self.regions.append(region)

        try:
            result_widget = self.create_analysis_result_widget(rubberband, region)
        except ValueError as e:
            self.rubberbands.pop()
            self.regions.pop()
            QMessageBox.critical(
                self,
                "Error",
                e.args[0] if e.args else "An error occurred during analysis.",
            )
            return False
        if self.rubberbands:
            self.rubberbands[self.current_view_index].set_filled(False)
        self.rois.append(result_widget)
        self.graphs.append([])
        self.current_view_index = len(self.rois) - 1
        self.navigate_to_roi(self.current_view_index)
        self.generate_analysis_graphs(region)
        self.rubberbands[-1].set_filled(True)
        return True

    def update_roi_region(self, rubberband, region):
        """Update the region of an existing ROI and regenerate graphs"""
        # Formulate tuple region — poly keeps its QPointF list, rect/circle are int-tuples
        if region[0] == "poly":
            region = (region[0], list(region[1]))
        else:
            region = (region[0], tuple(int(i) for i in region[1]))
            assert len(region[1]) == 4, "invalid region definition"

        # Find the ROI index for this rubberband
        try:
            index = self.rubberbands.index(rubberband)
        except ValueError:
            logger.warning("Rubberband not found in analysis view")
            return False

        # Update the region
        self.regions[index] = region

        # Set current index to the one being updated
        self.current_view_index = index

        # Clear existing graphs for this ROI
        self.graphs[index] = []

        # Regenerate graphs with new region data
        try:
            self.generate_analysis_graphs(region)
        except Exception as e:
            logger.error(f"Error updating ROI analysis: {e}")
            traceback.print_exc()
            return False

        # Navigate back to this ROI (updates UI)
        self.navigate_to_roi(index)

        # If we were viewing a specific graph, refresh it
        try:
            if (
                hasattr(self, "view_graph_interfaces")
                and isinstance(self.view_graph_interfaces, list)
                and index < len(self.view_graph_interfaces)
            ):
                interface = self.view_graph_interfaces[index]
                stacked_widget = interface["stacked_widget"]
                detail_page = interface["icon_detail_page"]

                # Check if detail page is the current widget
                if stacked_widget.currentWidget() == detail_page:
                    # Refresh the detail page with the current graph index
                    detail_page.set_icon_index(self.current_graph_index)
        except Exception as e:
            logger.error(f"Error refreshing graph display: {e}")

        return True

    def create_analysis_result_widget(self, rubberband, region):
        """Create the widget to display analysis results"""
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        # Create controls
        controls_layout, checkbox, opacity_slider = self.create_analysis_controls(rubberband, region)
        self.highlight_checkboxes.append(checkbox)
        self.opacity_sliders.append(opacity_slider)

        # Create graph selection interface for this specific roi
        graph_selection = self.create_graph_selection_interface()

        # Store the graph interface widgets in a list matching the roi index
        if not hasattr(self, "view_graph_interfaces") or not isinstance(self.view_graph_interfaces, list):
            self.view_graph_interfaces = []
        self.view_graph_interfaces.append({
            "stacked_widget": graph_selection,
            "icon_list_page": self.icon_list_page,
            "icon_detail_page": self.icon_detail_page,
        })

        # Add to layout
        result_layout.addLayout(controls_layout)
        result_layout.addWidget(graph_selection)

        return result_widget

    def create_analysis_controls(self, rubberband, region):
        """Create the control panel for analysis results"""
        controls_layout = QHBoxLayout()

        # Add protein selection
        multiComboBox = MultiComboBox()

        data = self.enc.view_tab.get_df()
        self.columns = get_marker_columns(data)
        multiComboBox.addItems(list(self.columns))

        for i in range(len(self.columns)):
            multiComboBox.model().item(i).setCheckState(Qt.CheckState.Checked)

        # Add buttons
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(multiComboBox.selectAll)
        
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(multiComboBox.deselectAll)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(
            lambda: self.handleComboBoxChanged(multiComboBox.get_checked_items())
        )

        delete_button = QPushButton("Delete")
        delete_button.setFixedWidth(100)
        delete_button.clicked.connect(self.delete_current_view)

        # Add color indicator
        color_button = QPushButton()
        color_button.setFixedSize(100, 50)
        color_button.setToolTip("Change ROI color")
        color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        rgb = QColor(rubberband.color).getRgb()[:3]
        color_button.setStyleSheet(
            f"QPushButton {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #888; border-radius: 4px; }}"
            f"QPushButton:hover {{ border-color: #555; }}"
        )
        color_button.clicked.connect(lambda: self.show_roi_color_dialog(rubberband, color_button))

        # Create combo and apply layout
        combo_apply_layout = QVBoxLayout()
        combo_apply_layout.addWidget(multiComboBox)
        
        # New: Horizontal layout for select all/deselect all buttons
        select_buttons_layout = QHBoxLayout()
        select_buttons_layout.addWidget(select_all_button)
        select_buttons_layout.addWidget(deselect_all_button)
        
        combo_apply_layout.addLayout(select_buttons_layout)
        combo_apply_layout.addWidget(apply_button)

        # Add cell highlight checkbox & opacity slider layout
        highlight_layout = QVBoxLayout()
        highlight_checkbox = QCheckBox("Highlight Cells")
        highlight_checkbox.setChecked(False)
        highlight_checkbox.stateChanged.connect(self.update_cell_highlighting)
        highlight_layout.addWidget(highlight_checkbox)

        # Opacity slider layout
        slider_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity: 30%")
        opacity_label.setStyleSheet("font-size: 10px; font-weight: 600;")
        
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(0)
        opacity_slider.setMaximum(50)  # Max 0.5 (represented as 50)
        opacity_slider.setValue(30)   # Default 0.3 (represented as 30)
        opacity_slider.setFixedHeight(15)
        opacity_slider.setToolTip("Adjust highlight opacity (Max 50%)")
        
        # Connect slider value change
        opacity_slider.valueChanged.connect(
            lambda val, lbl=opacity_label: self.handle_opacity_change(val, lbl)
        )
        
        slider_layout.addWidget(opacity_label)
        slider_layout.addWidget(opacity_slider)
        highlight_layout.addLayout(slider_layout)
        
        combo_apply_layout.addLayout(highlight_layout)

        # Create layout for color indicator and delete button
        color_delete_layout = QVBoxLayout()
        color_delete_layout.addWidget(color_button, alignment=Qt.AlignmentFlag.AlignCenter)
        color_delete_layout.addWidget(delete_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add layouts to main controls layout
        controls_layout.addLayout(combo_apply_layout)
        controls_layout.addLayout(color_delete_layout)

        return controls_layout, highlight_checkbox, opacity_slider

    def create_graph_selection_interface(self):
        """Create the interface for selecting different graph types"""
        self.icon_list = [
            "Boxplot",
            "Spatial Autocorrelation",
            "Spatial Heatmap",
            "Pi Chart",
            "Histogram",
        ]

        self.icon_paths = [
            resource_path("assets/icons/linechart.png"),
            resource_path("assets/icons/spatial_autocorrelation.png"),
            resource_path("assets/icons/heatmap.png"),
            resource_path("assets/icons/piechart.png"),
            resource_path("assets/icons/barchart.png"),
            resource_path("assets/icons/scatter.png"),
        ]

        self.stacked_widget = QStackedWidget()

        self.icon_list_page = GraphsList(
            icon_list=self.icon_list,
            navigate_to_page=self.show_icon_detail_page,
            icon_paths=self.icon_paths,
            result_details_layout=None,
        )
        self.icon_detail_page = GraphInDetail(
            navigate_back=self.show_icon_grid_page,
            open_in_new_window=self.open_in_new_window,
            parent=self,
        )

        self.stacked_widget.addWidget(self.icon_list_page)
        self.stacked_widget.addWidget(self.icon_detail_page)

        return self.stacked_widget

    def generate_analysis_graphs(self, region):
        # Get filtered data
        data = self.get_current_roi_data()
        assert data is not None, "Shape selection not implemeneted in analysis tab"

        # Capture current state for the lambdas
        current_data = data
        current_columns = list(self.columns)

        # Create and add graphs
        box_plot = self.create_box_plot(current_data, current_columns)
        self.add_graph_to_current_view(box_plot)

        graph_generators = [
            lambda d=current_data, c=current_columns: SpatialAutocorrelationWindow(
                self.get_z_heatmap_data(d, c)
            ),
            lambda d=current_data, c=current_columns: HeatmapWindow(
                self.get_z_heatmap_data(d, c)
            ),
            lambda d=current_data, c=current_columns: PieChartCanvas(d[c]),
            lambda d=current_data, c=current_columns: DistributionViewer(d[c]),
        ]

        for generator in graph_generators:
            self.add_graph_to_current_view(generator)

    def get_z_heatmap_data(self, data, columns):
        t = [c for c in columns if c not in ("Global X", "Global Y")]
        t = ["Global X", "Global Y"] + t
        return data[t]

    def create_box_plot(self, data, columns):
        """Create a box plot widget"""
        result_widget = QWidget()
        layout = QVBoxLayout(result_widget)
        filtered_data = data.loc[:, columns]
        filtered_data = filtered_data.melt(var_name="Protein", value_name="Expression")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            x="Expression",
            y="Protein",
            data=filtered_data,
            ax=ax,
            palette="Set2",
            flierprops=dict(marker="o", markersize=4, alpha=0.3),
        )

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Protein Expression Box Plot")
        plt.subplots_adjust(bottom=0.3, left=0.4)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        result_widget.figure = fig

        # Connect hover event
        canvas.mpl_connect(
            "motion_notify_event",
            lambda event: self.on_boxplot_hover(event, canvas, ax, data, columns)
        )
        canvas.mpl_connect("figure_leave_event", lambda event: QToolTip.hideText())

        return result_widget

    def on_boxplot_hover(self, event, canvas, ax, data, columns):
        if event.inaxes != ax:
            QToolTip.hideText()
            return

        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            num_proteins = len(columns)
            row_idx = int(round(y))
            if 0 <= row_idx < num_proteins:
                protein = columns[row_idx]
                values = data[protein].dropna()
                if len(values) > 0:
                    q1 = values.quantile(0.25)
                    median = values.median()
                    q3 = values.quantile(0.75)
                    min_val = values.min()
                    max_val = values.max()

                    tooltip_text = (
                        f"<b>Protein:</b> {protein}<br/>"
                        f"<b>Median:</b> {median:.2f}<br/>"
                        f"<b>Q1:</b> {q1:.2f} | <b>Q3:</b> {q3:.2f}<br/>"
                        f"<b>IQR:</b> {(q3-q1):.2f}<br/>"
                        f"<b>Min:</b> {min_val:.2f} | <b>Max:</b> {max_val:.2f}"
                    )
                    QToolTip.showText(QCursor.pos(), tooltip_text, canvas)
                    return
        QToolTip.hideText()

    def get_current_roi_data(self):
        """Retrieve the filtered data for the current ROI"""
        if not self.regions or self.current_view_index >= len(self.regions):
            return None
        region = self.regions[self.current_view_index]
        if region[0] == "rect":
            return self.get_rect_data(region[1])
        elif region[0] == "circle":
            return self.get_circle_data(region[1])
        elif region[0] == "poly":
            return self.get_poly_data(region[1])
        return None

    def get_rect_data(self, region):
        """Get data filtered by the selected region"""
        data = self.enc.view_tab.get_df()
        x_min, y_min, x_max, y_max = [i for i in region]
        logger.debug("%s %s %s %s", x_min, y_min, x_max, y_max)

        return data[
            (data["Global X"] >= x_min)
            & (data["Global X"] <= x_max)
            & (data["Global Y"] >= y_min)
            & (data["Global Y"] <= y_max)
        ]

    def get_circle_data(self, region):
        """Get data filtered by the selected circular region"""
        data = self.enc.view_tab.get_df()
        center_x, center_y, x2, y2 = region

        radius = np.linalg.norm(np.array([x2 - center_x, y2 - center_y]))

        # Apply circular filter
        return data[
            ((data["Global X"] - center_x) ** 2 + (data["Global Y"] - center_y) ** 2)
            <= radius**2
        ]

    def get_poly_data(self, region):
        """Get data filtered by the selected polygon region using ray casting algorithm"""
        data = self.enc.view_tab.get_df()

        def point_in_polygon(point, polygon):
            """Check if a point is inside a polygon using ray casting algorithm"""
            x, y = point
            inside = False

            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                xi, yi = polygon[i]
                xj, yj = polygon[j]

                # Check if point is between the y-coordinates of the edge
                if ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi
                ):
                    inside = not inside

            return inside

        poly_points = [(p.x(), p.y()) for p in region]

        points = data[["Global X", "Global Y"]].values
        mask = [point_in_polygon(point, poly_points) for point in points]

        return data[mask]

    def handleComboBoxChanged(self, checked_items):
        """Handle changes in protein selection."""
        if not checked_items:
            QMessageBox.warning(
                self,
                "Alert",
                "You have nothing selected! Please select at least one protein.",
            )
            return

        # Clear current graphs
        self.graphs[self.current_view_index] = []

        self.columns = checked_items

        # Regenerate graphs
        self.generate_analysis_graphs(self.regions[self.current_view_index])

        # Update roi
        self.navigate_to_roi(self.current_view_index)

        # Refresh the detail page if it is currently visible
        try:
            if (
                hasattr(self, "view_graph_interfaces")
                and isinstance(self.view_graph_interfaces, list)
                and self.current_view_index < len(self.view_graph_interfaces)
            ):
                interface = self.view_graph_interfaces[self.current_view_index]
                stacked_widget = interface["stacked_widget"]
                detail_page = interface["icon_detail_page"]

                if stacked_widget.currentWidget() == detail_page:
                    # Refresh the detail page with the current graph index
                    detail_page.set_icon_index(self.current_graph_index)
        except Exception as e:
            logger.error(f"Error refreshing graph display: {e}")

    def show_icon_detail_page(self, index):
        logger.debug(
            "show_icon_detail_page1, current roi: %s graph: %s",
            self.current_view_index,
            index,
        )

        # Get the graph interface for the current roi
        if not isinstance(self.view_graph_interfaces, list) or self.current_view_index >= len(self.view_graph_interfaces):
            logger.error("No graph interface found for current roi")
            return

        interface = self.view_graph_interfaces[self.current_view_index]

        # Update the graph index and display
        self.current_graph_index = index
        interface["icon_detail_page"].set_icon_index(index)
        interface["stacked_widget"].setCurrentWidget(interface["icon_detail_page"])

    def show_icon_grid_page(self):
        # Get the graph interface for the current roi
        if not isinstance(self.view_graph_interfaces, list) or self.current_view_index >= len(self.view_graph_interfaces):
            logger.error("No graph interface found for current roi")
            return

        interface = self.view_graph_interfaces[self.current_view_index]
        interface["stacked_widget"].setCurrentWidget(interface["icon_list_page"])

    def open_in_new_window(self):
        """
        Allows us the ability to pop out a graph and view it in a new window.
        """
        if not isinstance(self.view_graph_interfaces, list) or self.current_view_index >= len(self.view_graph_interfaces):
            return

        interface = self.view_graph_interfaces[self.current_view_index]
        icon_detail_page = interface["icon_detail_page"]
        graph_index = self.current_graph_index

        # Create a new window to display the current graph
        new_window = RegenerateOnCloseWindow(
            regenerate_callback=lambda: self._on_new_window_closed(interface, graph_index)
        )
        new_window.setWindowTitle("Graph Window")
        layout = QVBoxLayout()

        # Retrieve the current graph widget from the encoder
        widget = self.get_current_graph()
        if widget is not None:
            widget.setSizePolicy(
                widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding
            )
            layout.addWidget(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        new_window.setLayout(layout)
        new_window.resize(300, 200)
        new_window.show()

        # Track the new window and update original content
        self.windows.append(new_window)
        for i in reversed(range(icon_detail_page.content_layout.count())):
            widget_to_remove = icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        icon_detail_page.content_layout.addWidget(QLabel("visible in new window"))

    def get_graph_for_roi(self, roi_index, graph_index):
        """Retrieve graph widget for a specific ROI index and graph index."""
        if roi_index < 0 or roi_index >= len(self.graphs):
            return None
        if graph_index < 0 or graph_index >= len(self.graphs[roi_index]):
            return None

        graph = self.graphs[roi_index][graph_index]
        if callable(graph):
            graph = graph()
            self.graphs[roi_index][graph_index] = graph
        return graph

    @pyqtSlot()
    def _on_new_window_closed(self, interface, graph_index):
        """
        When the new window is closed, regenerate the graph in the main window.
        """
        # Verify the interface is still active and get its current index
        try:
            roi_index = self.view_graph_interfaces.index(interface)
        except (ValueError, AttributeError):
            # Interface is no longer tracked or list has been reset
            return

        icon_detail_page = interface.get("icon_detail_page")
        if not icon_detail_page:
            return

        # Double check C++ object survival
        try:
            icon_detail_page.parent()
        except RuntimeError:
            return

        new_graph = self.get_graph_for_roi(roi_index, graph_index)
        if new_graph is None:
            return

        # Clear old content and update layout with the regenerated graph
        for i in reversed(range(icon_detail_page.content_layout.count())):
            widget_to_remove = icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

        icon_detail_page.content_layout.addWidget(new_graph)

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


class FullImageAnalysisView(QWidget):
    def __init__(self, pixmap_label, enc: "MainWindow"):
        super().__init__()
        self.enc = enc
        self.columns = []
        self.graphs = []
        self.current_graph_index = 0
        self.initUI()

    def initUI(self):
        self.setLayout(QVBoxLayout())
        # Graph Area
        self.stacked_widget = QStackedWidget()

        self.icon_list = [
            "UMAP",
        ]

        self.icon_paths = [
            resource_path("assets/icons/scatter.png"),
        ]

        self.icon_list_page = GraphsList(
            icon_list=self.icon_list,
            navigate_to_page=self.show_icon_detail_page,
            icon_paths=self.icon_paths,
            result_details_layout=None,
        )

        self.stacked_widget.addWidget(self.icon_list_page)

        self.layout().addWidget(self.stacked_widget)

        # Try to generate initial graphs if data exists
        self.generate_analysis_graphs()

    def generate_analysis_graphs(self):
        self.graphs = []

        # Create graphs similar to ROI view but for full data
        graph_generators = [
            lambda: self.enc.view_tab.open_umap_analysis(),
        ]

        for generator in graph_generators:
            self.graphs.append(generator)

    def show_icon_detail_page(self, index):
        self.current_graph_index = index
        if index == 0:
            self.enc.view_tab.open_umap_analysis()

    def show_icon_grid_page(self):
        self.stacked_widget.setCurrentWidget(self.icon_list_page)

    def get_graph(self, index):
        if not self.graphs:
            return QLabel("No graphs available")
        if index >= len(self.graphs):
            return QLabel("Graph index out of range")

        graph = self.graphs[index]
        if callable(graph):
            graph = graph()
            self.graphs[index] = graph
        return graph

    def open_in_new_window(self):
        # Create a new window to display the current graph
        new_window = RegenerateOnCloseWindow(
            regenerate_callback=self._on_new_window_closed
        )
        new_window.setWindowTitle("Graph Window")
        layout = QVBoxLayout()

        # Retrieve the current graph widget
        widget = self.get_graph(self.current_graph_index)
        widget.setSizePolicy(
            widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding
        )
        layout.addWidget(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        new_window.setLayout(layout)
        new_window.resize(300, 200)
        new_window.show()

        # Simple hack to remove from current view while in new window
        # (similar to ROIView implementation)
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.icon_detail_page.content_layout.addWidget(QLabel("visible in new window"))

    def _on_new_window_closed(self):
        # Regenerate/restore graph
        new_graph = self.get_graph(self.current_graph_index)
        for i in reversed(range(self.icon_detail_page.content_layout.count())):
            widget_to_remove = self.icon_detail_page.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.icon_detail_page.content_layout.addWidget(new_graph)


class MultiComboBox(QComboBox):
    itemsCheckedChanged = pyqtSignal(list)  # Signal to emit the list of checked items

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))
        self._popup_visible = False

        # Connect to the dataChanged signal to update the text and emit our signal
        self.model().dataChanged.connect(self.onItemStateChanged)
        self.lineEdit().textEdited.connect(self.filter_items)

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
        selected_items = [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]
        self.lineEdit().setText(", ".join(selected_items))

    def onItemStateChanged(self):
        """
        The function `onItemStateChanged` updates displayed text when item states change.
        """
        if not self._popup_visible:
            self.updateText()

    def selectAll(self):
        """
        Checks all items in the model.
        """
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            item.setCheckState(Qt.CheckState.Checked)
        self.updateText()

    def deselectAll(self):
        """
        Unchecks all items in the model.
        """
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.updateText()

    def get_checked_items(self):
        """
        This function retrieves checked items from a model.
        :return: The function `get_checked_items` returns a list of items that are checked in the model.
        """
        return [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]

    def get_checked_items2(self):
        """
        Return all checked items.
        """
        return self.get_checked_items()

    def showPopup(self):
        self._popup_visible = True
        self.lineEdit().setReadOnly(False)
        self.lineEdit().clear()
        self.filter_items("")
        super().showPopup()

    def hidePopup(self):
        super().hidePopup()
        self._popup_visible = False
        self.lineEdit().setReadOnly(True)
        self.updateText()
        self.filter_items("")

    def filter_items(self, text):
        query = text.lower().strip()
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item:
                self.view().setRowHidden(i, query not in item.text().lower())

    from PyQt6.QtWidgets import (QLabel, QPushButton, QStackedWidget,
                                 QVBoxLayout, QWidget)


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
        self.top_layout.addWidget(
            new_window_button, alignment=Qt.AlignmentFlag.AlignRight
        )

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
        widget.setSizePolicy(
            widget.sizePolicy().Policy.Expanding, widget.sizePolicy().Policy.Expanding
        )
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

                button.setStyleSheet(
                    " text-align:left; padding: 10px; margin-top: 10px;"
                )
                button.setIcon(QIcon(self.icon_paths[index]))  # Set icon for the button

                button.clicked.connect(lambda _, idx=index: navigate_to_page(idx))
                layout.addWidget(button)


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
