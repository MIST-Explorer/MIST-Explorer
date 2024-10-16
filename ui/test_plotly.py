import sys
import plotly.graph_objs as go
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import plotly.io as pio

import graphing.Neighbor as neighbor

# Step 1: Create the Plotly graph
def create_plotly_graph():
    return neighbor.plot()


# Step 2: Create the PyQt6 application and GUI
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set up the window layout
        self.setWindowTitle('Plotly Graph in PyQt6')
        self.setGeometry(100, 100, 800, 600)

        # Create a QVBoxLayout to hold the QWebEngineView
        layout = QVBoxLayout()

        # Create a QWebEngineView (from PyQt6.WebEngineWidgets)
        self.web_view = QWebEngineView()

        # Add the web view to the layout
        layout.addWidget(self.web_view)

        # Set the layout on the main window
        self.setLayout(layout)

        # Step 3: Load the Plotly graph into the QWebEngineView
        self.display_plot()

    def display_plot(self):
        # Get the Plotly HTML
        html_string = create_plotly_graph()
        
        with open('test.html', 'w') as f:
            f.write(html_string)

        # Load the HTML string into the QWebEngineView
        self.web_view.setHtml(html_string)

# Step 4: Run the PyQt6 application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Create an instance of the main window and show it
    main_window = MainWindow()
    main_window.show()

    # Execute the application
    sys.exit(app.exec())
