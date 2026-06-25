import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QScrollArea,
    QToolTip,
    QCheckBox,
)
from PyQt6.QtGui import QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from core.dataframe_utils import get_marker_columns


class DistributionViewer(QMainWindow):
    def __init__(self, data):
        super().__init__()

        cols = get_marker_columns(data)

        self.data = data[cols]  # DataFrame with data to plot

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Line Plot Distribution Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Control Panel
        controls_layout = QHBoxLayout()
        self.log_x_cb = QCheckBox("Log Scale (X-axis)")
        self.log_y_cb = QCheckBox("Log Scale (Y-axis)")
        self.normalize_cb = QCheckBox("Normalize Peaks (0-100%)")

        self.log_x_cb.stateChanged.connect(self.update_plot)
        self.log_y_cb.stateChanged.connect(self.update_plot)
        self.normalize_cb.stateChanged.connect(self.update_plot)

        controls_layout.addWidget(self.log_x_cb)
        controls_layout.addWidget(self.log_y_cb)
        controls_layout.addWidget(self.normalize_cb)
        controls_layout.addStretch()
        self.layout.addLayout(controls_layout)

        # Scroll area for the plot
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.scroll_area.setWidget(self.plot_container)
        self.layout.addWidget(self.scroll_area)

        # Matplotlib canvas for the plot - Pass constrained_layout=True to prevent overlap
        self.figure, self.ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        # Connect hover event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("figure_leave_event", lambda e: QToolTip.hideText())

        # Plot the data
        self.plot_distributions(self.data)

    def update_plot(self):
        self.plot_distributions(self.data)

    def plot_distributions(self, data):
        data = trim_outliers(data).dropna()
        """Plots distributions of all columns as line plots."""
        self.ax.clear()  # Clear the previous plot

        log_x = self.log_x_cb.isChecked()
        log_y = self.log_y_cb.isChecked()
        normalize = self.normalize_cb.isChecked()

        for column in data.columns:
            col_data = data[column].dropna()
            if col_data.empty:
                continue

            if log_x:
                col_data_pos = col_data[col_data > 0]
                if len(col_data_pos) > 1:
                    min_val = col_data_pos.min()
                    max_val = col_data_pos.max()
                    if min_val == max_val:
                        min_val /= 10.0
                        max_val *= 10.0
                    bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
                    counts, bin_edges = np.histogram(col_data_pos, bins=bins)
                    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
                else:
                    counts, bin_edges = np.histogram(col_data, bins=30)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            else:
                counts, bin_edges = np.histogram(col_data, bins=30)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            counts = counts.astype(float)
            if normalize:
                max_count = np.max(counts)
                if max_count > 0:
                    counts = (counts / max_count) * 100.0

            if log_y:
                counts[counts == 0] = np.nan

            # Plot as a line
            self.ax.plot(bin_centers, counts, label=column, alpha=0.7)

        self.ax.set_title("Value Distributions")
        
        if log_x:
            self.ax.set_xscale("log")
            self.ax.set_xlabel("Value (Log Scale)")
        else:
            self.ax.set_xscale("linear")
            self.ax.set_xlabel("Value")

        if log_y:
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")

        if normalize:
            self.ax.set_ylabel("Normalized Count (%)")
        else:
            self.ax.set_ylabel("Count")

        self.ax.legend(loc="upper right")
        self.canvas.draw()  # Refresh the canvas with the new plot

    def on_hover(self, event):
        if event.inaxes != self.ax:
            QToolTip.hideText()
            return

        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Find the closest point/line in display (pixel) coordinates
            best_dist = float("inf")
            best_info = None
            threshold_px = 15.0  # Max distance in pixels to trigger tooltip

            for line in self.ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                label = line.get_label()
                if xdata is None or len(xdata) == 0:
                    continue

                # Find index of closest x point
                if self.log_x_cb.isChecked():
                    valid = (xdata > 0) & (x > 0)
                    if np.any(valid):
                        temp_diff = np.full_like(xdata, float('inf'))
                        temp_diff[valid] = np.abs(np.log10(xdata[valid]) - np.log10(x))
                        idx = np.argmin(temp_diff)
                    else:
                        idx = np.argmin(np.abs(xdata - x))
                else:
                    idx = np.argmin(np.abs(xdata - x))

                x_pt, y_pt = xdata[idx], ydata[idx]
                if np.isnan(x_pt) or np.isnan(y_pt):
                    continue

                # Convert both point and mouse coordinates to display (pixel) coordinates
                pt_display = self.ax.transData.transform((x_pt, y_pt))
                mouse_display = self.ax.transData.transform((x, y))

                dist = np.linalg.norm(pt_display - mouse_display)
                if dist < best_dist:
                    best_dist = dist
                    best_info = (label, x_pt, y_pt)

            if best_dist <= threshold_px and best_info is not None:
                label, x_pt, y_pt = best_info
                if self.normalize_cb.isChecked():
                    count_text = f"{y_pt:.1f}%"
                else:
                    count_text = f"{int(round(y_pt))}"
                
                tooltip_text = (
                    f"<b>Protein:</b> {label}<br/>"
                    f"<b>Value:</b> {x_pt:.2f}<br/>"
                    f"<b>Count:</b> {count_text}" if not self.normalize_cb.isChecked() else f"<b>Normalized Count:</b> {count_text}"
                )
                QToolTip.showText(QCursor.pos(), tooltip_text, self.canvas)
                return

        QToolTip.hideText()

    def save_plot(self):
        file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
        data = pd.read_excel(file_path).iloc[:, 5:9]
        data = trim_outliers(data).dropna()

        data = data[data.columns[3:]]
        self.plot_distributions(data)




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
        return data.apply(
            lambda col: (
                _trim_outliers_series(col, method, factor)
                if pd.api.types.is_numeric_dtype(col)
                else col
            )
        )
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
