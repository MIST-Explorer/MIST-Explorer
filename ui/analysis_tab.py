from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal
import random

import graphing.Test as test

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
    add_graph_signal = pyqtSignal()

    def __init__(self, pixmap_label):
        super().__init__()

        self.graphs = []
        self.current_graph_index = 0

        self.initUI()
        self.add_graph_signal.connect(self.add_graph_to_new_view)

        # Add CellDensityPlot as one of the plots
        self.add_cell_density_plot()
        self.add_box_plot_canvas()
        
    def add_box_plot_canvas(self):
        box_plot_canvas = BoxPlotCanvas()
        self.graphs.append(box_plot_canvas)
        self.current_graph_index = len(self.graphs) - 1
        self.show_graph(self.current_graph_index)

    def add_cell_density_plot(self):
        cell_density_plot = CellDensityPlot()
        self.graphs.append(cell_density_plot)
        self.current_graph_index = len(self.graphs) - 1
        self.show_graph(self.current_graph_index)

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
        # self.add_random_graph_new_view_button = QPushButton("Add Random Graph to New View")
        # self.add_random_graph_current_view_button = QPushButton("Add Random Graph to Current View")
        # self.add_random_graph_new_view_button.clicked.connect(self.add_random_graph_to_new_view)
        # self.add_random_graph_current_view_button.clicked.connect(self.add_random_graph_to_current_view)
        # add_graph_layout.addWidget(self.add_random_graph_new_view_button)
        # add_graph_layout.addWidget(self.add_random_graph_current_view_button)

        main_layout.addLayout(add_graph_layout)

        # Add button for adding a new line to the current graph
        # self.add_line_button = QPushButton("Add Line to Current Graph")
        # self.add_line_button.clicked.connect(self.add_line_to_current_graph)
        # main_layout.addWidget(self.add_line_button)

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

    def add_line_to_current_graph(self, color):
        if self.graphs:
            current_graph = self.graphs[self.current_graph_index]
            current_graph.redraw(color)

    def get_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    pixmap_label = QLabel()
    analysis_tab = AnalysisTab(pixmap_label)
    analysis_tab.show()
    sys.exit(app.exec())
