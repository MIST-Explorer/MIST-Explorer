import sys
import random
from PyQt6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.all_data = []
        self.all_colors = []

        # Create a figure instance to plot on
        self.figure = Figure()

        # Create a Canvas Widget that displays the figure
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Create a Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a button and connect it to the plot method
        self.plot_button = QtWidgets.QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot)

        # Create a button and connect it to the redraw method
        self.redraw_button = QtWidgets.QPushButton('Redraw')
        self.redraw_button.clicked.connect(self.redraw)

        # Set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.redraw_button)
        self.setLayout(layout)

    def plot(self):
        '''Plot some random data'''
        data = [random.random() for _ in range(10)]
        self.all_data.append(data)

        # Create an axis
        ax = self.figure.add_subplot(111)

        # Discard the old graph
        ax.clear()

        # Plot data
        ax.plot(data, '*-')

        # Refresh canvas
        self.canvas.draw()

    def redraw(self, color):
        
        '''Redraw the current graph with up to n different lines in new random colors'''
        # Clear the current figure
        self.figure.clear()

        # Create an axis
        ax = self.figure.add_subplot(111)

        # Add new random data
        self.all_data.append([random.random() for _ in range(10)])
        self.all_colors.append(color)
        
        # Plot all data with different colors
        for data, color in zip(self.all_data, self.all_colors):
            ax.bar(range(len(data)), data, color=[co/255 for co in color], alpha=0.7)
            # ax.plot(data, '*-', color=[co/255 for co in color])

        # Refresh canvas
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    main_window.show()
    sys.exit(app.exec())
