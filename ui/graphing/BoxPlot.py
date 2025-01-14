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
        # # Load your dataset from the Excel file
        # file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
        # data = pd.read_excel(file_path)

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