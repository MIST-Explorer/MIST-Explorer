import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from sklearn.datasets import make_blobs
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class UMAPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UMAP Visualizer")
        self.setGeometry(100, 100, 800, 600)

        self.data, self.labels = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)
        self.umap_model = None
        self.embedding = np.zeros((self.data.shape[0], 2))
        self.iteration = 0

        self.initUI()

    def initUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Sliders and labels
        self.n_neighbors_label = QLabel("n_neighbors: 15")
        self.n_neighbors_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_neighbors_slider.setMinimum(2)
        self.n_neighbors_slider.setMaximum(50)
        self.n_neighbors_slider.setValue(15)
        self.n_neighbors_slider.valueChanged.connect(self.update_n_neighbors_label)

        self.min_dist_label = QLabel("min_dist: 0.1")
        self.min_dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_dist_slider.setMinimum(1)
        self.min_dist_slider.setMaximum(100)
        self.min_dist_slider.setValue(10)
        self.min_dist_slider.valueChanged.connect(self.update_min_dist_label)

        layout.addWidget(self.n_neighbors_label)
        layout.addWidget(self.n_neighbors_slider)
        layout.addWidget(self.min_dist_label)
        layout.addWidget(self.min_dist_slider)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start UMAP")
        self.start_button.clicked.connect(self.start_umap)
        button_layout.addWidget(self.start_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

        # Timer for iterative updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_umap)

    def update_n_neighbors_label(self):
        value = self.n_neighbors_slider.value()
        self.n_neighbors_label.setText(f"n_neighbors: {value}")

    def update_min_dist_label(self):
        value = self.min_dist_slider.value() / 100
        self.min_dist_label.setText(f"min_dist: {value:.2f}")

    def start_umap(self):
        n_neighbors = self.n_neighbors_slider.value()
        min_dist = self.min_dist_slider.value() / 100

        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_epochs=1, 
            random_state=42
        )

        self.embedding = np.random.randn(self.data.shape[0], 2)  # Initialize randomly
        self.iteration = 0
        self.timer.start(50)

    def update_umap(self):
        if self.iteration < 200:
            self.umap_model.n_epochs = self.iteration + 1  # Increment epochs
            self.embedding = self.umap_model.fit_transform(self.data)  # Update embedding

            self.ax.clear()
            self.ax.scatter(
                self.embedding[:, 0], 
                self.embedding[:, 1], 
                c=self.labels, 
                cmap='viridis', 
                s=10, 
                alpha=0.5
            )
            self.ax.set_title(f"UMAP Iteration: {self.iteration}")
            self.canvas.draw()
            self.iteration += 1
        else:
            self.timer.stop()

    def reset(self):
        self.timer.stop()
        self.ax.clear()
        self.ax.set_title("UMAP Visualizer")
        self.canvas.draw()
        self.iteration = 0
        self.umap_model = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UMAPVisualizer()
    window.show()
    sys.exit(app.exec())
