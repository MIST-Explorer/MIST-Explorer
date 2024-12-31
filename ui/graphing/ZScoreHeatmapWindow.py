import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from itertools import combinations

# implement 1

class ZScoreHeatmapWindow(QMainWindow):
    def __init__(self, data, region, parent=None):
        super().__init__(parent)

        font = {
        
        'size'   : 10}

        matplotlib.rc('font', **font)
        matplotlib.rcParams['figure.figsize'] = [4, 4]

                # Define region of interest and filter dataset
        x_min, y_min, x_max, y_max = region
        filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                            (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]
        
        print(filtered_data)

        # Extract proteins and calculate centroids
        protein_columns = filtered_data.columns[3:]  # Assuming proteins start at the 4th column
        proteins = protein_columns


        def calculate_weighted_centroids(data, proteins, signal_threshold=100):
            centroids = {}
            for protein in proteins:
                protein_cells = data[data[protein] >= signal_threshold]
                if len(protein_cells) > 0:
                    weights = protein_cells[protein]
                    avg_x = np.average(protein_cells['Global X'], weights=weights)
                    avg_y = np.average(protein_cells['Global Y'], weights=weights)
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

        for _ in range(num_simulations):
            # Randomly shuffle the cell coordinates
            shuffled_x = np.random.permutation(filtered_data['Global X'].values)
            shuffled_y = np.random.permutation(filtered_data['Global Y'].values)

            # Update centroids with shuffled locations
            shuffled_data = filtered_data.copy()
            shuffled_data['Global X'] = shuffled_x
            shuffled_data['Global Y'] = shuffled_y

            # Recalculate centroids with shuffled data
            shuffled_centroids = calculate_weighted_centroids(shuffled_data, proteins)

            # Calculate distances with shuffled centroids
            dist = calculate_pairwise_distances(shuffled_centroids)

            # Store distances in random_distances
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

                # Handle the case where standard deviation is zero to avoid NaNs
                if std_random_dist == 0:
                    z_scores[(protein1, protein2)] = 0  # Set to neutral z-score for self-pairs
                else:
                    z_scores[(protein1, protein2)] = (observed_distance - mean_random_dist) / std_random_dist

        # Create z-score matrix for heatmap, filling diagonal with neutral z-score
        z_score_matrix = pd.DataFrame(index=proteins, columns=proteins)

        for (protein1, protein2), z in z_scores.items():
            z_score_matrix.loc[protein1, protein2] = z
            z_score_matrix.loc[protein2, protein1] = z  # Ensure symmetry

        np.fill_diagonal(z_score_matrix.values, 0)

        self.setWindowTitle("Clustered Z-Score Heatmap")
        self.resize(1200, 800)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create the Matplotlib figure and canvas for displaying
        self.figure = plt.figure(figsize=(12, 10))
        # self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # Create and plot the clustermap, no ax passed
        row_linkage = linkage(z_score_matrix.fillna(0), method="average", metric="euclidean")
        col_linkage = linkage(z_score_matrix.fillna(0).T, method="average", metric="euclidean")

        # Create the clustered heatmap
        g = sns.clustermap(
            z_score_matrix.astype(float),
            cmap="coolwarm",
            vmin=-5,
            vmax=5,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            xticklabels=True,
            yticklabels=True,
        )

        # Set the title
        g.fig.suptitle("Clustered Z-Score Heatmap")

        # Transfer the clustermap figure to our main figure to embed in PyQt
        self.figure = g.fig

        # Create a canvas to embed the Matplotlib figure into the PyQt6 application
        canvas = FigureCanvas(self.figure)
        layout.addWidget(canvas)





    # def plot_heatmap(self, z_score_matrix):
    #     # Perform hierarchical clustering on rows and columns
        


def main():
    app = QApplication(sys.argv)

    # Load your dataset (replace with actual file path and region)
    file_path = r"/Users/clark/Downloads/cell_data_8_8_Full_Dataset_Biopsy.xlsx"
    data = pd.read_excel(file_path)
    # Filter out columns with the title "N/A"
    data = data.loc[:, data.columns != "N/A"]

    # Create and show the main window
    window = ZScoreHeatmapWindow(data, region=(1400, 2100, 1600, 3600))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
