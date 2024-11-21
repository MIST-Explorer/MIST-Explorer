# from PyQt6.QtWidgets import QStyledItemDelegate, QComboBox, QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
import random

import ui.graphing.Test as test

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class BoxPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(12, 8))
        super().__init__(fig)
        self.setParent(parent)
        self.plot_boxplot()

    def plot_boxplot(self):
        # Load your dataset from the Excel file
        file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
        data = pd.read_excel(file_path)

        # Define the region of interest
        x_min, y_min = 0, 0  # Adjust these coordinates as needed
        x_max, y_max = 12000, 12000

        # Filter the dataset to include only cells within the specified region
        filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                             (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]

        # Extract the list of all proteins dynamically based on the column range
        protein_columns = filtered_data.columns[13:15]  # Assuming proteins start at the 4th column

        # Filter data to include only the protein expression columns
        protein_data = filtered_data[protein_columns]

        # Reshape the data for seaborn boxplot (melt to "long" format)
        protein_data_melted = protein_data.melt(var_name='Protein', value_name='Expression')

        # Plot the boxplot
        sns.boxplot(x='Protein', y='Expression', data=protein_data_melted, palette='Set2', showfliers=False, ax=self.ax)

        # Customize the plot
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=90, )
        self.ax.set_yticklabels(self.ax.get_yticklabels(), )
        self.ax.set_xlabel('Protein', )
        self.ax.set_ylabel('Expression Level', )
        self.ax.set_title(f'Prot. Exp. Box Plot', )
        plt.subplots_adjust(bottom=0.25)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Expression Box Plot")
        self.setGeometry(100, 100, 1280, 720)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.canvas = BoxPlotCanvas(self)
        layout.addWidget(self.canvas)

class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class CellDensityPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Density Plot')
        self.setGeometry(100, 100, 800, 600)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)
        
        # Create the plot
        self.figure, self.ax = plt.subplots(2, 1, figsize=(30, 40), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.plot_cell_density()

    def plot_cell_density(self):
        sns.set_style('ticks')
        # Load your dataset
        file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
        data = pd.read_excel(file_path)

        # Define regions of interest using their coordinates (x_min, y_min, x_max, y_max)
        regions = {
            "T Cell Enriched": (7400, 6800, 10800, 7800),
            "B Cell Enriched": (4200, 2900, 6200, 3200),
            "Injured Area": (6000, 4400, 9500, 6200),
            "Glomerular": (0, 0, 3600, 3600),
        }

        # Markers of interest (e.g., CD3, CD20)
        markers = ['CD3', 'CD20', 'CD163']

        # Store the results
        region_data = []

        # Loop over each region and each marker to calculate the total number of cells expressing each marker
        for region_name, (x_min, y_min, x_max, y_max) in regions.items():
            # Filter cells within the region, correcting for y_min as top and y_max as bottom
            cells_in_region = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                                (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]

            # Calculate the total number of cells
            total_cells = len(cells_in_region)

            # Store results for each marker in the current region
            region_marker_data = {'Region': region_name, 'Total Cells': total_cells}

            for marker in markers:
                # Count cells expressing the marker above the threshold (e.g., > 2000)
                marker_positive_cells = len(cells_in_region[cells_in_region[marker] > 2000])

                # Calculate percentage of cells expressing the marker
                if total_cells > 0:
                    marker_percentage = (marker_positive_cells / total_cells) * 100
                else:
                    marker_percentage = 0

                # Store the results for the marker
                region_marker_data[f'{marker} Positive Cells'] = marker_positive_cells
                region_marker_data[f'{marker} %'] = marker_percentage

            # Add the result for the current region
            region_data.append(region_marker_data)

        # Convert results to a DataFrame for easy plotting
        region_df = pd.DataFrame(region_data)

        # Plotting the results: Stacked bar plot
        ax = self.ax

        # Define a color palette for the markers
        colors = plt.get_cmap('Set1')(np.linspace(0, 1, len(markers)))

        # Plot the percentages with improved styling
        bottom = np.zeros(len(region_df))
        for i, marker in enumerate(markers):
            ax[0].bar(region_df['Region'], region_df[f'{marker} %'], bottom=bottom, label=marker, color=colors[i])
            bottom += region_df[f'{marker} %'].values  # Update the bottom for stacking

        # Label the percentage plot
        ax[0].set_ylabel('Percentage of Marker+ Cells',)
        ax[0].set_title('Marker Density Percentage Across Regions')
        ax[0].legend(title='Markers', )
        ax[0].tick_params(axis='x', rotation=45, )
        ax[0].tick_params(axis='y', )
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        # Plot the total counts with improved styling
        width = 0.1  # Bar width for better spacing
        x = np.arange(len(region_df))

        for i, marker in enumerate(markers):
            ax[1].bar(x + i * width, region_df[f'{marker} Positive Cells'], width=width, label=marker, color=colors[i])

        # Label the total counts plot
        ax[1].set_ylabel('Number of Marker+ Cells',)
        ax[1].set_title('Total Marker+ Cells Across Regions', )
        ax[1].legend(title='Markers',)
        ax[1].set_xticks(x + width / 2 * (len(markers) - 1))
        ax[1].set_xticklabels(region_df['Region'], rotation=45)
        ax[1].tick_params(axis='y', labelsize=12)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        # Fine-tune the layout and add gridlines
        plt.grid(visible=True, which='both', axis='y', linestyle='--', linewidth=0.5)

        # Adjust margins for better spacing
        plt.subplots_adjust(left=0.15, bottom=0.15)

        # Draw the canvas
        self.canvas.draw()

        # Print the results in table format for inspection
        print(region_df)
        # plt.savefig('plot.png', dpi=300, bbox_inches='tight')



class AnalysisTab(QWidget):

    def __init__(self, pixmap_label):
        super().__init__()

        self.views = []
        self.view_index = 0

        self.rubberbands = []

        self.graphs = []

        self.initUI()

        # self.graph_views = []
        # self.current_view_index = 0
        # self.view_index = 0

        # # Add navigation buttons for views
        # view_nav_layout = QHBoxLayout()
        # self.view_back_button = QPushButton("< View Back")
        # self.view_next_button = QPushButton("Next View >")

        # self.view_back_button.clicked.connect(self.show_previous_view)
        # self.view_next_button.clicked.connect(self.show_next_view)
        # view_nav_layout.addWidget(self.view_back_button)
        # view_nav_layout.addWidget(self.view_next_button)

        # main_layout.addLayout(view_nav_layout)

        # Add initial view
        # self.add_graph_view()



    def initUI(self):
        main_layout = QVBoxLayout()

        # Add navigation buttons
        nav_layout = QHBoxLayout()

        # view based nav
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

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        

        main_layout.addWidget(self.scroll_area)

        self.setLayout(main_layout)

        # Add initial graph
        # self.add_new_view()

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


    def set_view(self, index):
        # Clear the current layout
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Add the new graph
        self.scroll_layout.addWidget(self.views[index])

        # Update button states
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.views) - 1)

    def add_new_view(self):
        sc = QWidget()

        self.views.append(sc)
        self.view_index = len(self.views) - 1
        self.set_view(self.view_index)

    def delete_view(self):
        if self.views:
            self.views.pop(self.view_index)
            self.rubberbands[self.view_index].hide()
            self.rubberbands.pop(self.view_index)
            if len(self.views) == 0:
                for i in reversed(range(self.scroll_layout.count())):
                    widget = self.scroll_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
            else:
                self.prev_view()

    def prev_view(self):
        try:
            self.rubberbands[self.view_index].setFilled(False)
        except Exception as e:
            print(e)

        if self.view_index > 0:
            self.view_index -= 1
            self.set_view(self.view_index)

            self.rubberbands[self.view_index].setFilled(True)
            

    def next_view(self):
        try:
            self.rubberbands[self.view_index].setFilled(False)
        except Exception as e:
            print(e)

        if self.view_index < len(self.views) - 1:
            self.view_index += 1
            self.set_view(self.view_index)

            self.rubberbands[self.view_index].setFilled(True)

    def add_stats(self, random_data, pyqt_color, rubberband):

        print(len(self.rubberbands))
        print(self.view_index)
        if len(self.rubberbands) != 0:
            self.rubberbands[self.view_index].setFilled(False)
        # except Exception as e:
        #     print(e)

        # Convert the PyQt color to RGB
        pyqt_color_rgb = QColor(pyqt_color).getRgb()[:3]
        
        
        # Create a new widget to display the results
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        # Create a horizontal layout for the result details
        result_details_layout = QHBoxLayout()

        # Add a label saying "Selection Results"
        result_label = QLabel("Selection Results")
        result_details_layout.addWidget(result_label)

        # Add a delete button (not hooked up to anything yet)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_view)
        result_details_layout.addWidget(delete_button)

        # Add a rectangle with the color converted to RGB
        color_label = QLabel()
        color_label.setFixedSize(100, 50)
        color_label.setStyleSheet(f"background-color: rgb({pyqt_color_rgb[0]}, {pyqt_color_rgb[1]}, {pyqt_color_rgb[2]});")
        result_details_layout.addWidget(color_label)

        # Add the horizontal layout to the main vertical layout
        result_layout.addLayout(result_details_layout)

        # Add the result widget to the scroll layout
        self.scroll_layout.addWidget(result_widget)
        self.add_new_view()
        self.next_view()

        if self.views:
            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot the boxplot with random data using matplotlib
            ax.boxplot(random_data["Expression"],
                   labels=random_data['Protein'].unique(), showfliers=False)

            # Customize the plot
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel('Protein')
            ax.set_ylabel('Expression Level')
            ax.set_title('Random Protein Expression Box Plot')
            plt.subplots_adjust(bottom=0.25)

            # Add the new figure to the layout
            result_layout.addWidget(FigureCanvas(fig))
            
            self.views[-1] = result_widget

            self.back_plot = QPushButton("< Back")
            self.next_plot = QPushButton("Next >")
            
            result_layout.addWidget(self.back_plot)
            result_layout.addWidget(self.next_plot)

            self.rubberbands.append(rubberband)
            rubberband.setFilled(True)

            self.scroll_layout.addWidget(result_widget)
            # self.rubberbands[self.view_index].setFilled(True)


            # self.prev_view()
            # self.next_view()

    def get_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    pixmap_label = QLabel()
    analysis_tab = AnalysisTab(pixmap_label)
    analysis_tab.show()
    sys.exit(app.exec())
