import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
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
        self.figure, self.ax = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
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
        ax[0].set_ylabel('Percentage of Marker+ Cells', fontsize=14, fontweight='bold')
        ax[0].set_title('Marker Density Percentage Across Regions', fontsize=16, fontweight='bold')
        ax[0].legend(title='Markers', fontsize=12)
        ax[0].tick_params(axis='x', rotation=45, labelsize=12)
        ax[0].tick_params(axis='y', labelsize=12)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        # Plot the total counts with improved styling
        width = 0.25  # Bar width for better spacing
        x = np.arange(len(region_df))

        for i, marker in enumerate(markers):
            ax[1].bar(x + i * width, region_df[f'{marker} Positive Cells'], width=width, label=marker, color=colors[i])

        # Label the total counts plot
        ax[1].set_ylabel('Number of Marker+ Cells', fontsize=14, fontweight='bold')
        ax[1].set_title('Total Marker+ Cells Across Regions', fontsize=16, fontweight='bold')
        ax[1].legend(title='Markers', fontsize=12)
        ax[1].set_xticks(x + width / 2 * (len(markers) - 1))
        ax[1].set_xticklabels(region_df['Region'], rotation=45, fontsize=12)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = CellDensityPlot()
    main.show()
    sys.exit(app.exec_())
