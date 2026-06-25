import logging
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QSpinBox, QToolTip,
                             QVBoxLayout, QWidget)
from scipy import sparse
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class SpatialAutocorrelationWindow(QMainWindow):
    def __init__(self, data, region=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Autocorrelation")

        font = {"size": 8}
        matplotlib.rc("font", **font)
        matplotlib.rcParams["figure.figsize"] = [5, 5]

        self.data = data.copy()
        self.region = region
        # The first two columns are Global X and Global Y, the rest are proteins
        self.proteins = self.data.columns[2:]
        logger.debug(f"Proteins for spatial autocorrelation: {list(self.proteins)}")

        # Central widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Control Panel
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 5)
        controls_layout.setSpacing(10)

        cofactor_label = QLabel("Cofactor:")
        self.cofactor_spin = QDoubleSpinBox()
        self.cofactor_spin.setRange(0.1, 10000.0)
        self.cofactor_spin.setSingleStep(0.5)
        self.cofactor_spin.setValue(5.0)
        self.cofactor_spin.setDecimals(1)

        k_label = QLabel("Neighbors (K):")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 1000)
        self.k_spin.setValue(6)

        perm_label = QLabel("Permutations:")
        self.perm_spin = QSpinBox()
        self.perm_spin.setRange(10, 10000)
        self.perm_spin.setSingleStep(100)
        self.perm_spin.setValue(1000)

        self.update_btn = QPushButton("Update Heatmap")
        self.update_btn.clicked.connect(self.update_plot)

        controls_layout.addWidget(cofactor_label)
        controls_layout.addWidget(self.cofactor_spin)
        controls_layout.addWidget(k_label)
        controls_layout.addWidget(self.k_spin)
        controls_layout.addWidget(perm_label)
        controls_layout.addWidget(self.perm_spin)
        controls_layout.addWidget(self.update_btn)
        controls_layout.addStretch()

        self.main_layout.addLayout(controls_layout)

        self.figure = None
        self.canvas = None
        self.g = None
        self.ax_heatmap = None

        # Run initial plot
        self.update_plot()

    def show_error_message(self, message):
        if self.canvas is not None:
            self.main_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        error_label = QLabel(message)
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
        
        self.canvas = error_label
        self.main_layout.addWidget(self.canvas)

    def update_plot(self):
        # Set wait cursor during computation
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.update_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            cofactor = self.cofactor_spin.value()
            k = self.k_spin.value()
            n_perm = self.perm_spin.value()

            proteins = list(self.proteins)
            if len(proteins) < 2:
                self.show_error_message("At least 2 proteins are required to generate a heatmap.")
                return

            raw = self.data[proteins].to_numpy(dtype=float)
            xy = self.data[["Global X", "Global Y"]].to_numpy(dtype=float)
            n, p = raw.shape

            if n < 2:
                self.show_error_message("At least 2 cells are required for spatial correlation.")
                return

            # Adjust K if n <= K
            k_eff = min(k, n - 1)
            if k_eff < 1:
                self.show_error_message("Insufficient cells/neighbors to run analysis.")
                return

            # Clear old canvas/error label
            if self.canvas is not None:
                self.main_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()
                self.canvas = None
                self.g = None
                self.ax_heatmap = None

            plt.close("all")

            # 1. z-scored expression (arcsinh first)
            Zexpr = np.arcsinh(raw / cofactor)
            # Avoid division by zero in standard deviation scaling
            stds = Zexpr.std(0)
            stds[stds == 0.0] = 1.0
            Zexpr = (Zexpr - Zexpr.mean(0)) / stds

            # 2. row-standardized kNN spatial weights (sparse)
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(xy)
            _, idx = nn.kneighbors(xy)
            idx = idx[:, 1:]                                  # drop self
            rows = np.repeat(np.arange(n), k_eff)
            cols = idx.ravel()
            Wsp = sparse.csr_matrix((np.full(n * k_eff, 1.0 / k_eff), (rows, cols)), shape=(n, n))

            def moran_matrix(Zmat):
                lag = Wsp @ Zmat                              # spatial lag, n x p
                M = (Zmat.T @ lag) / n                        # p x p
                return 0.5 * (M + M.T)                        # symmetrize

            # 3. observed + permutation null
            I_obs = moran_matrix(Zexpr)
            nullI = np.empty((n_perm, p, p))
            rng = np.random.default_rng(0)
            for t in range(n_perm):
                nullI[t] = moran_matrix(Zexpr[rng.permutation(n)])

            null_std = nullI.std(0)
            null_mean = nullI.mean(0)
            # Avoid division by zero
            null_std[null_std == 0.0] = 1.0

            Z = (I_obs - null_mean) / null_std
            Z_df = pd.DataFrame(Z, index=proteins, columns=proteins)

            # 4. clustered heatmap
            d = 1.0 / (1.0 + np.abs(Z_df.values))
            np.fill_diagonal(d, 0.0)
            d_sym = 0.5 * (d + d.T)
            
            # Symmetrize distance and run hierarchical clustering linkage
            link = linkage(squareform(d_sym, checks=False), "average")
            vlim = np.nanpercentile(np.abs(Z_df.values), 99)
            if vlim == 0.0 or np.isnan(vlim):
                vlim = 1.0

            g = sns.clustermap(
                Z_df, row_linkage=link, col_linkage=link,
                cmap="RdBu_r", center=0, vmin=-vlim, vmax=vlim,
                annot=False,
                linewidths=0.5, linecolor="white", figsize=(7, 6),
                dendrogram_ratio=(0.12, 0.12),
            )
            g.ax_heatmap.set_xlabel("")
            g.ax_heatmap.set_ylabel("")
            g.ax_heatmap.tick_params(labelsize=8)
            g.fig.suptitle("Spatial Autocorrelation", y=0.96, fontsize=10)

            # Adjust layout to prevent collisions and position the color legend on the left
            g.fig.subplots_adjust(top=0.88, bottom=0.12, left=0.15, right=0.85)
            g.ax_cbar.set_position([0.02, 0.80, 0.03, 0.12])
            g.ax_cbar.set_title("Moran's I\nspatial Z", fontsize=7, pad=5, loc="center")

            self.figure = g.fig
            self.canvas = FigureCanvas(self.figure)
            
            # Keep references for hover handling
            self.g = g
            self.ax_heatmap = g.ax_heatmap
            self.canvas.mpl_connect("motion_notify_event", self.on_hover)
            self.canvas.mpl_connect("figure_leave_event", lambda e: QToolTip.hideText())

            self.main_layout.addWidget(self.canvas)

        except Exception as e:
            logger.exception("Error updating spatial autocorrelation plot")
            self.show_error_message(f"Error during calculation:\n{str(e)}")

        finally:
            self.update_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def on_hover(self, event):
        if self.g is None or self.ax_heatmap is None:
            QToolTip.hideText()
            return

        if event.inaxes == self.ax_heatmap:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                p = len(self.proteins)
                col_idx = int(np.floor(x))
                row_idx = int(np.floor(y))
                if 0 <= col_idx < p and 0 <= row_idx < p:
                    # In sns.clustermap, g.data2d holds the clustered order data
                    row_name = self.g.data2d.index[row_idx]
                    col_name = self.g.data2d.columns[col_idx]
                    val = self.g.data2d.iloc[row_idx, col_idx]

                    tooltip_text = (
                        f"<b>Row:</b> {row_name}<br/>"
                        f"<b>Col:</b> {col_name}<br/>"
                        f"<b>Z-score:</b> {val:.2f}"
                    )
                    QToolTip.showText(QCursor.pos(), tooltip_text, self.canvas)
                    return

        QToolTip.hideText()


def main():
    app = QApplication(sys.argv)

    # Attempt to load demo_cell_data.csv first, then fallback to excel path
    try:
        data = pd.read_csv("demo_cell_data.csv")
    except FileNotFoundError:
        file_path = r"/mnt/user-data/uploads/cell_data_SP14_5326_Complete_15_Proteins.xlsx"
        try:
            data = pd.read_excel(file_path)
        except FileNotFoundError:
            print("Demo datasets not found. Please provide cell coordinate data.")
            sys.exit(1)

    data = data.loc[:, data.columns != "N/A"]

    window = SpatialAutocorrelationWindow(data)
    window.resize(900, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
