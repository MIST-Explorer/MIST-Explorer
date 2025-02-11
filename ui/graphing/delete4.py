import sys
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QSlider, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scanpy as sc
import matplotlib.pyplot as plt








class UMAPComputation(QThread):
    finished = pyqtSignal(object, object)  # Emit when finished, sending embedding and labels

    def __init__(self, data, n_neighbors, min_dist, n_iter):
        super().__init__()
        self.data = data
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_iter = n_iter
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        adata = sc.AnnData(self.data)
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        sc.tl.umap(adata, min_dist=self.min_dist, n_epochs=self.n_iter)

        if not self.running:
            return
        embedding = adata.obsm['X_umap']

        # Perform clustering
        sc.tl.leiden(adata, resolution=0.5)
        labels = adata.obs['leiden'].astype(int).to_numpy()

        if self.running:
            self.finished.emit(embedding, labels)


class UMAPWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Matplotlib Canvas
        self.canvas = UMAPCanvas()
        layout.addWidget(self.canvas)

        # Controls
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        # Sliders for UMAP parameters
        sliders_layout = QVBoxLayout()
        self.n_neighbors_slider = self.create_slider("n_neighbors", 5, 50, 15)
        self.min_dist_slider = self.create_slider("min_dist (x100)", 1, 100, 10)
        self.iter_slider = self.create_slider("iterations", 10, 1000, 200)
        sliders_layout.addWidget(self.n_neighbors_slider)
        sliders_layout.addWidget(self.min_dist_slider)
        sliders_layout.addWidget(self.iter_slider)

        controls_layout.addLayout(sliders_layout)
        layout.addLayout(controls_layout)

        self.setLayout(layout)

        # Connections
        self.start_button.clicked.connect(self.start_computation)
        self.stop_button.clicked.connect(self.stop_computation)

        # Thread for UMAP
        self.thread = None

    def create_slider(self, label_text, min_value, max_value, initial_value):
        container = QVBoxLayout()
        label = QLabel(f"{label_text}: {initial_value}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_value, max_value)
        slider.setValue(initial_value)
        slider.valueChanged.connect(lambda value: label.setText(f"{label_text}: {value}"))
        container.addWidget(label)
        container.addWidget(slider)
        frame = QFrame()
        frame.setLayout(container)
        return frame

    def start_computation(self):
        if self.thread is not None and self.thread.isRunning():
            return

        # Get parameter values from sliders
        n_neighbors = self.n_neighbors_slider.findChild(QSlider).value()
        min_dist = self.min_dist_slider.findChild(QSlider).value() / 100.0
        n_iter = self.iter_slider.findChild(QSlider).value()

        # Generate dummy data
        np.random.seed(42)
        data = np.random.rand(500, 10)

        # Set up and start UMAP computation in a thread
        self.thread = UMAPComputation(data, n_neighbors, min_dist, n_iter)
        self.thread.finished.connect(self.on_computation_finished)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.canvas.show_loading()

    def stop_computation(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.canvas.hide_loading()

    def on_computation_finished(self, embedding, labels):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.canvas.hide_loading()
        self.canvas.plot_umap(embedding, labels)


class UMAPCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self.scatter = None
        self.points = None
        self.labels = None
        self.colors = None
        self.hover_cluster = None

    def show_loading(self):
        self.ax.clear()
        self.ax.text(
            0.5, 0.5, "Loading...", fontsize=20, ha="center", va="center", transform=self.ax.transAxes
        )
        self.fig.canvas.draw()

    def hide_loading(self):
        self.ax.clear()
        self.fig.canvas.draw()

    def plot_umap(self, embedding, labels):
        self.ax.clear()
        self.points = embedding
        self.labels = labels

        unique_labels = np.unique(self.labels)
        cmap = plt.cm.tab10
        self.colors = [cmap(i % 10) for i in unique_labels]  # Assign colors to each cluster
        self.colors = {label: cmap(i % 10) if label != -1 else (0.5, 0.5, 0.5, 1.0) for i, label in enumerate(unique_labels)}

        # Map labels to colors
        point_colors = [self.colors[label] for label in self.labels]

        self.scatter = self.ax.scatter(
            embedding[:, 0], embedding[:, 1], s=10, c=point_colors
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax:
            # Find the cluster under the click
            distances = np.linalg.norm(self.points - [event.xdata, event.ydata], axis=1)
            closest_point_idx = np.argmin(distances)
            closest_cluster = self.labels[closest_point_idx]
            cluster_color = self.colors[closest_cluster]

            # Print cluster information
            print(f"Cluster: {closest_cluster}, Color: {cluster_color}")

    def on_hover(self, event):
        if self.points is None or self.labels is None:
            return

        if event.inaxes == self.ax:
            distances = np.linalg.norm(self.points - [event.xdata, event.ydata], axis=1)
            closest_idx = np.argmin(distances)
            closest_cluster = self.labels[closest_idx]

            if closest_cluster != self.hover_cluster:
                sizes = np.where(self.labels == closest_cluster, 50, 10)
                self.scatter.set_sizes(sizes)
                self.hover_cluster = closest_cluster
                self.fig.canvas.draw()
        else:
            if self.hover_cluster is not None:
                self.scatter.set_sizes(np.full(len(self.labels), 10))
                self.hover_cluster = None
                self.fig.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UMAPWidget()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())