import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys


class HeatmapWindow(QMainWindow):
    def __init__(self, data, parent=None):

        super().__init__(parent)


        # Filter the dataset to include only cells within the specified region
        filtered_data = data

        # Extract the list of all proteins dynamically based on the column range
        protein_columns = filtered_data.columns[3:]  # Assuming proteins start at the 4th column
        proteins = protein_columns

        # Dummy interaction matrix and correlation matrix for demonstration
        interaction_matrix = np.random.rand(len(proteins), len(proteins)) * 300
        correlation_matrix = filtered_data[proteins].corr(method='spearman')

       
        # self.setWindowTitle("Protein Correlation Heatmap")
        # self.resize(1200, 800)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create the Matplotlib figure and axes
        self.figure, self.ax = plt.subplots(figsize=(12, 10))

        self.figure.suptitle("Heatmap indicates protein-protein correlation; \nSize indicates spatial distances between protein expression cells", fontsize=7)
        
        # self.ax.text(0, -2.0, "Heatmap indicates protein-protein correlation; \nSize indicates spatial distances between protein expression cells", fontsize=7, weight="bold")

        # Plot the heatmap on the axes
        self.plot_heatmap(data, interaction_matrix, proteins, correlation_matrix)

        # Create a canvas to embed the Matplotlib figure into the PyQt6 application
        canvas = FigureCanvas(self.figure)
        layout.addWidget(canvas)

        

    def plot_heatmap(self, data, interaction_matrix, proteins, correlation_matrix):

        # Plot the heatmap
        ax = sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            vmin=-0.25,
            vmax=0.25,
            cbar=False,
            annot=False,
            xticklabels=proteins,
            yticklabels=proteins[::-1],  # Inverted y-axis labels
            ax=self.ax,
        )

        # Customize axis labels
        # self.ax.set_xticks(range(len(proteins)))
        # self.ax.set_yticks(range(len(proteins)))
        # self.ax.set_xticklabels(proteins, rotation=90, fontname="Arial")
        # self.ax.set_yticklabels(proteins[::-1], fontname="Arial")
        self.figure.subplots_adjust(bottom=0.20, left=0.20)

        # Add circles proportional to the interaction strengths (edge lengths)
        for i in range(len(proteins)):
            for j in range(len(proteins)):
                if interaction_matrix[i, j] > 0:  # Only add circles for actual interactions
                    self.ax.add_patch(
                        plt.Circle(
                            (j + 0.5, i + 0.5),
                            radius=interaction_matrix[i, j] / 1000,
                            color="turquoise",
                            fill=True,
                        )
                    )

        # Set the title
        # self.ax.set_title(
        #     f"Protein Correlation Heatmap",
        #     fontsize=16,
        # )



def main():
    app = QApplication(sys.argv)

    # Load your dataset (replace with actual file path and)
    file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
    data = pd.read_excel(file_path)

    # Example parameters
    

    # Create and show the main window
    window = HeatmapWindow(data)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()