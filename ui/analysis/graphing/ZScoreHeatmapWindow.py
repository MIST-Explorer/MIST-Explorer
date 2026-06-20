import logging
import sys
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from scipy.cluster.hierarchy import linkage

logger = logging.getLogger(__name__)


class ZScoreHeatmapWindow(QMainWindow):
    def __init__(self, data, region=None, parent=None):
        super().__init__(parent)

        font = {"size": 8}
        matplotlib.rc("font", **font)
        matplotlib.rcParams["figure.figsize"] = [3, 3]

        filtered_data = data.copy()

        protein_columns = filtered_data.columns[2:]
        proteins = protein_columns
        logger.debug(proteins)

        def calculate_weighted_centroids(data, proteins, signal_threshold=100):
            centroids = {}
            for protein in proteins:
                protein_cells = data[data[protein] >= signal_threshold]
                if len(protein_cells) > 0:
                    weights = protein_cells[protein]
                    avg_x = np.average(protein_cells["Global X"], weights=weights)
                    avg_y = np.average(protein_cells["Global Y"], weights=weights)
                    centroids[protein] = (avg_x, avg_y)
            return centroids

        centroids = calculate_weighted_centroids(filtered_data, proteins)

        def calculate_pairwise_distances(centroids):
            distances = {}
            for protein1, protein2 in combinations(proteins, 2):
                if protein1 in centroids and protein2 in centroids:
                    x1, y1 = centroids[protein1]
                    x2, y2 = centroids[protein2]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    distances[(protein1, protein2)] = distance
                    distances[(protein2, protein1)] = distance
            return distances

        observed_distances = calculate_pairwise_distances(centroids)

        num_simulations = 1000
        random_distances = {pair: [] for pair in combinations(proteins, 2)}

        # Extract base arrays once for faster shuffling in the loop
        orig_x = filtered_data["Global X"].values
        orig_y = filtered_data["Global Y"].values

        # monte carlo simulation to generate random distributions
        print(f"Running {num_simulations} simulations...")
        for i in range(num_simulations):
            shuffled_data = filtered_data.copy()

            # --- original ---
            shuffled_x = orig_x.copy()
            shuffled_y = orig_y.copy()
            np.random.shuffle(shuffled_x)
            np.random.shuffle(shuffled_y)

            shuffled_data["Global X"] = shuffled_x
            shuffled_data["Global Y"] = shuffled_y

            # --- together ---
            # shuffled_coords = orig_coords.copy()
            # np.random.shuffle(shuffled_coords)
            #
            # shuffled_data["Global X"] = shuffled_coords[:, 0]
            # shuffled_data["Global Y"] = shuffled_coords[:, 1]

            # Recalculate centroids with shuffled data
            shuffled_centroids = calculate_weighted_centroids(shuffled_data, proteins)

            # Calculate distances with shuffled centroids
            dist = calculate_pairwise_distances(shuffled_centroids)

            # Store distances
            for pair in random_distances:
                try:
                    random_distances[pair].append(dist[pair])
                except KeyError:
                    random_distances[pair].append(0)

        # Step 4: Calculate z-scores
        z_scores = {}
        for (protein1, protein2), observed_distance in observed_distances.items():
            if (protein1, protein2) in random_distances:
                random_dist = np.array(random_distances[(protein1, protein2)])
                mean_random_dist = np.mean(random_dist)
                std_random_dist = np.std(random_dist)

                if std_random_dist == 0:
                    z_scores[(protein1, protein2)] = 0
                else:
                    z_scores[(protein1, protein2)] = (
                        observed_distance - mean_random_dist
                    ) / std_random_dist

        # Create z-score matrix for heatmap
        z_score_matrix = pd.DataFrame(index=proteins, columns=proteins)

        for (protein1, protein2), z in z_scores.items():
            z_score_matrix.loc[protein1, protein2] = z
            z_score_matrix.loc[protein2, protein1] = z

        np.fill_diagonal(z_score_matrix.values, 0)

        # Central widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.figure = plt.figure(figsize=(10, 10))

        row_linkage = linkage(
            z_score_matrix.fillna(0), method="average", metric="correlation"
        )

        g = sns.clustermap(
            z_score_matrix.astype(float),
            figsize=(6, 6),
            cmap="coolwarm",
            vmin=-5,
            vmax=5,
            dendrogram_ratio=(0.1, 0.1),  # Set top dendrogram space to 1%
            cbar_pos=(0.02, 0.1, 0.01, 0.4),
            row_linkage=row_linkage,
            col_linkage=row_linkage,
            xticklabels=True,
            yticklabels=True,
        )

        g.fig.suptitle("Clustered Z-Score Heatmap", y=0.98)
        g.fig.subplots_adjust(top=0.90, left=0.2, right=0.8)

        hm_pos = g.ax_heatmap.get_position()
        cbar_width = 0.02
        cbar_height = 0.4
        hm_center_y = hm_pos.y0 + (hm_pos.height / 2)
        cbar_bottom = hm_center_y - (cbar_height / 2)
        g.ax_cbar.set_position([0.05, cbar_bottom, cbar_width, cbar_height])

        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=6)
        plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=6)
        self.figure = g.fig

        canvas = FigureCanvas(self.figure)
        layout.addWidget(canvas)


def main():
    app = QApplication(sys.argv)

    file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"

    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print("Data file not found. Ensure the path is correct.")
        sys.exit(1)

    data = data.loc[:, data.columns != "N/A"]

    window = ZScoreHeatmapWindow(data, region=(1400, 2100, 1600, 3600))
    window.resize(800, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
