import sys
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QScrollArea, QPushButton, QLabel, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DistributionViewer(QMainWindow):
    def __init__(self, data):
        super().__init__()

        cols = data.columns[3:]

        self.data = data[cols] # DataFrame with data to plot
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Line Plot Distribution Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Scroll area for the plot
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.scroll_area.setWidget(self.plot_container)
        self.layout.addWidget(self.scroll_area)

        # Matplotlib canvas for the plot
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        # Button to save the current plot
        # self.save_button = QPushButton("Save Plot")
        # self.save_button.clicked.connect(self.save_plot)
        # self.layout.addWidget(self.save_button)

        # Plot the data
        self.plot_distributions(self.data)

    def plot_distributions(self, data):
        data = trim_outliers(data).dropna()
        """Plots distributions of all columns as line plots."""
        self.ax.clear()  # Clear the previous plot

        for column in data.columns:
            # Compute histogram (values and counts)
            counts, bin_edges = np.histogram(data[column].dropna(), bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get bin centers

            # Plot as a line
            self.ax.plot(bin_centers, counts, label=column, alpha=0.7)

        self.ax.set_title("Value Distributions")
        self.ax.set_xlabel("Value")
        self.ax.set_ylabel("Count")
        self.ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw()  # Refresh the canvas with the new plot

    def save_plot(self):
        file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        data = pd.read_excel(file_path).iloc[:, 5:9]
        data = trim_outliers(data).dropna()

        data = data[data.columns[3:]]
        self.plot_distributions(data)

import pandas as pd

def trim_outliers(data, method="iqr", factor=1.5):
    """
    Trims outliers from a DataFrame or Series.
    
    Parameters:
        data (pd.DataFrame or pd.Series): The data to trim.
        method (str): The method to use for trimming ('iqr' or 'zscore').
        factor (float): The threshold for identifying outliers. Default is 1.5 for IQR.
        
    Returns:
        pd.DataFrame or pd.Series: Data with outliers trimmed.
    """
    if isinstance(data, pd.Series):
        return _trim_outliers_series(data, method, factor)
    elif isinstance(data, pd.DataFrame):
        return data.apply(lambda col: _trim_outliers_series(col, method, factor) if pd.api.types.is_numeric_dtype(col) else col)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")


def trim_outliers(data, method="iqr", factor=1.5):
    """
    Trims outliers from a DataFrame or Series.
    
    Parameters:
        data (pd.DataFrame or pd.Series): The data to trim.
        method (str): The method to use for trimming ('iqr' or 'zscore').
        factor (float): The threshold for identifying outliers. Default is 1.5 for IQR.
        
    Returns:
        pd.DataFrame or pd.Series: Data with outliers trimmed.
    """
    if isinstance(data, pd.Series):
        return _trim_outliers_series(data, method, factor)
    elif isinstance(data, pd.DataFrame):
        return data.apply(lambda col: _trim_outliers_series(col, method, factor) if pd.api.types.is_numeric_dtype(col) else col)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")

def _trim_outliers_series(series, method="iqr", factor=1.5):
    """
    Trims outliers from a pandas Series.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return series  # Skip non-numeric data
    
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    else:
        raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

    return series[(series >= lower_bound) & (series <= upper_bound)]


if __name__ == "__main__":
    # Example data for testing
    file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
    data = pd.read_excel(file_path).iloc[:, 3:7]
    # data = 
    # print(data)
    

    

    app = QApplication(sys.argv)
    viewer = DistributionViewer(data)

    

    viewer.show()
    sys.exit(app.exec())



# if __name__ == "__main__":
    

#     app = QApplication(sys.argv)
#     viewer = OverlayLinePlotViewer(data)
#     viewer.show()
#     sys.exit(app.exec())
