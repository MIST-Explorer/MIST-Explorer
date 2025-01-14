# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# class PieChartCanvas(FigureCanvas):
#     def __init__(self, parent=None, df_path=None):
#         fig, self.ax = plt.subplots(figsize=(8, 6))
#         super().__init__(fig)
#         self.setParent(parent)

#         self.df = None

#         if df_path is not None:
#             self.df = pd.read_excel(df_path)
#         self.plot_pie_chart()

#     def plot_pie_chart(self):
#         # Generate random data for demonstration

#         if self.df is None:
#             file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
#             df = pd.read_excel(file_path)
#         # df = pd.DataFrame(data)
#         columns = df.columns[3:]  # Exclude the first column (e.g., 'Region')

#         # Aggregate data for the pie chart
#         values = df[columns].sum()

#         # Create labels with raw counts
#         labels = [f"{protein} ({int(count)})" for protein, count in zip(values.index, values)]

#         # Plot the pie chart
#         wedges, texts, autotexts = self.ax.pie(
#             values,
#             labels=labels,
#             autopct='%1.1f%%',
#             startangle=90,
#             pctdistance=0.85,
#         )

#         # Add a label for the total number of rows at the bottom
#         total_rows = len(df)
#         self.ax.text(0, -1.3, f"Total Rows: {total_rows}", ha='center', fontsize=10, weight="bold")

#         # Customize the plot
#         self.ax.set_title("Protein Expression Pie Chart")
#         plt.setp(autotexts, size=10, weight="bold")
#         plt.subplots_adjust(bottom=0.25)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Protein Expression Pie Chart")
#         self.setGeometry(100, 100, 1280, 720)
#         central_widget = QWidget(self)
#         self.setCentralWidget(central_widget)
#         layout = QVBoxLayout(central_widget)
#         self.canvas = PieChartCanvas(self)
#         layout.addWidget(self.canvas)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())

# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# class PieChartCanvas(FigureCanvas):
#     def __init__(self, parent=None):
#         self.fig, self.ax = plt.subplots(figsize=(8, 6))
#         super().__init__(self.fig)
#         self.setParent(parent)
#         self.plot_pie_chart()

#     def plot_pie_chart(self):
#         # plt.rcParams.update({'font.size': 4})

#         # Generate random data for demonstration
#         file_path = r"/Users/clark/Desktop/wang/protein_visualization_app/ui/graphing/Grouped Cells Biopsy Data.xlsx"
#         df = pd.read_excel(file_path)
#         columns = df.columns[3:]  # Exclude the first column (e.g., 'Region')
#         df =df[columns]

#         # Count how many rows are dominated by each protein
#         dominant_counts = df.idxmax(axis=1).value_counts()

#         # Create labels with counts
#         labels = [f"{protein} ({count})" for protein, count in dominant_counts.items()]
#         values = dominant_counts.values

#         # Plot the pie chart
#         self.ax.clear()
#         wedges, texts, autotexts = self.ax.pie(
#             values,
#             labels=labels,
#             autopct='%1.1f%%',
#             startangle=90,
#             pctdistance=1,
#             textprops={'fontsize': 7}
#         )

#         # texts[0].set_fontsize(4)

#         # Add a label for the total number of rows at the bottom
#         total_rows = len(df)
#         self.ax.text(0, -1.3, f"Total Rows: {total_rows}", ha='center', fontsize=10, weight="bold")

#         # Customize the plot
#         self.ax.set_title("Protein Dominance Pie Chart")
#         plt.setp(autotexts, size=10, weight="bold")
#         plt.subplots_adjust(bottom=0.25)


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Protein Dominance Pie Chart")
#         self.setGeometry(100, 100, 1280, 720)
#         central_widget = QWidget(self)
#         self.setCentralWidget(central_widget)
#         layout = QVBoxLayout(central_widget)

#         # Add the pie chart canvas
#         self.canvas = PieChartCanvas(self)
#         layout.addWidget(self.canvas)

#         # Ensure the chart resizes with the window
#         self.setCentralWidget(central_widget)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(0)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PieChartCanvas(FigureCanvas):
    def __init__(self, df, region, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.region = region
        
        self.df = df

        self.plot_pie_chart()

    def plot_pie_chart(self):
        df = self.df
        x_min, y_min, x_max, y_max = self.region

        # Filter the data based on the selected region
        df = df[(df['Global X'] >= x_min) & (df['Global X'] <= x_max) &
                (df['Global Y'] >= y_min) & (df['Global Y'] <= y_max)]
        df = df[df.columns[3:]]  # Exclude irrelevant columns

        # Count how many rows are dominated by each protein
        dominant_counts = df.idxmax(axis=1).value_counts()

        # Data for the pie chart
        labels = dominant_counts.index
        values = dominant_counts.values
        total = sum(values)

        # Modify labels: Add raw counts and hide labels for slices contributing less than 1%
        final_labels = [
            f"{label} ({value})" if (value / total * 100) >= 1 else ''  # Hide labels under 1%
            for label, value in zip(labels, values)
        ]

        # Plot the pie chart
        self.ax.clear()
        wedges, _, _ = self.ax.pie(
            values,
            labels=final_labels,  # Use the modified labels
            startangle=90,
            wedgeprops={'edgecolor': 'black'},
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 1 else '',  # Show % only if >= 1%
            pctdistance=.9,
            radius=1.45,
            labeldistance=1.15
        )

        # Add total rows label at the bottom
        total_rows = len(df)
        self.ax.text(0, -2.0, f"Total cells in this region: {total_rows}", ha='center', fontsize=10, weight="bold")

        # Adjust plot layout
        plt.subplots_adjust(bottom=0.3)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("Protein Dominance Pie Chart")
        self.setGeometry(100, 100, 1280, 720)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add the pie chart canvas
        self.canvas = PieChartCanvas(self)
        layout.addWidget(self.canvas)

        # Ensure the chart resizes with the window
        self.setCentralWidget(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())