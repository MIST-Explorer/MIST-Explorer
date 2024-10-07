import sys
import matplotlib
# matplotlib.use('Qt6Agg')

from PyQt6 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import random

# class MplCanvas2(FigureCanvasQTAgg):

#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)
        



class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.n = 1
        self.all_data =[]
        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvasQTAgg(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QtWidgets.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)
        
        

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]
        self.all_data.append(data)
        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.clear()

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()
        
        
    def redraw(self):
        self.n += 1
        ''' redraw the current graph with up to n different lines in new random colors '''
        # clear the current figure
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)
        
        self.all_data.append([random.random() for i in range(10)])
        for data in self.all_data:
            ax.plot(data, '*-', color=(random.random(), random.random(), random.random()))

        # refresh canvas
        self.canvas.draw()