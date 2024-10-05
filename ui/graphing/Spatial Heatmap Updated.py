import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def SpatialHeatmap():
    # Load your dataset from the Excel file
    file_path = r"C:\Users\meaha\Documents\Grouped Cells Biopsy Data.xlsx"
    data = pd.read_excel(file_path)

    # Define the region of interest
    x_min, y_min = 0, 2100
    x_max, y_max = 1600, 3600

    # Filter the dataset to include only cells within the specified region
    filtered_data = data[(data['Global X'] >= x_min) & (data['Global X'] <= x_max) &
                        (data['Global Y'] >= y_min) & (data['Global Y'] <= y_max)]

    # Extract the list of all proteins dynamically based on the column range
    protein_columns = filtered_data.columns[3:]  # Assuming proteins start at the 4th column
    proteins = protein_columns


    # Function to calculate weighted centroids and average protein expression for each protein
    def calculate_weighted_centroids(data, proteins, signal_threshold=100):
        centroids = {}
        avg_expression = {}
        for protein in proteins:
            # Filter cells expressing the current protein and meeting the signal threshold
            protein_cells = data[data[protein] >= signal_threshold]
            if len(protein_cells) > 0:
                # Calculate the weighted average for the centroids, weighted by the protein expression levels
                weights = protein_cells[protein]
                avg_x = np.average(protein_cells['Global X'], weights=weights)
                avg_y = np.average(protein_cells['Global Y'], weights=weights)
                avg_protein_expression = protein_cells[protein].mean()

                centroids[protein] = (avg_x, avg_y)
                avg_expression[protein] = avg_protein_expression
                print(
                    f"Protein {protein}: {len(protein_cells)} cells found, weighted centroid at ({avg_x}, {avg_y}), average expression: {avg_protein_expression:.2f}")
            else:
                print(f"No cells express {protein} above the signal threshold of {signal_threshold} in this region.")
        return centroids, avg_expression


    # Calculate weighted centroids and average expression for all proteins in the filtered region
    centroids, avg_expression = calculate_weighted_centroids(filtered_data, proteins)

    # Maximum possible distance in the region (e.g., diagonal distance of the region)
    max_possible_distance = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    # Now integrate the calculation of edge lengths between proteins
    edges = []
    for protein1, protein2 in combinations(proteins, 2):
        # Check if both proteins have centroids calculated
        if protein1 in centroids and protein2 in centroids:
            x1, y1 = centroids[protein1]
            x2, y2 = centroids[protein2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        else:
            # If any of the proteins does not have a centroid, set distance to max possible
            distance = max_possible_distance

        edges.append((protein1, protein2, distance))

    # Add the case where the protein is compared to itself (edge distance 0 or max possible if no centroid exists)
    for protein in proteins:
        if protein in centroids:
            edges.append((protein, protein, 0))
        else:
            edges.append((protein, protein, max_possible_distance))

    # Normalize distances for circle size scaling in the heatmap
    if edges:
        distances = np.array([distance for _, _, distance in edges])
        min_distance, max_distance = np.min(distances), np.max(distances)


        # Function to scale the distance to circle size, handling the case of zero distance
        def scale_circle_size(distance, min_size=10, max_size=300):
            if max_distance == min_distance:
                return max_size  # If all distances are zero (same protein), use max size
            normalized_distance = (distance - min_distance) / (max_distance - min_distance)
            circle_size = min_size + (1 - normalized_distance) * (max_size - min_size)
            return circle_size


        # Create a matrix for visualizing protein-protein interactions (edge lengths)
        num_proteins = len(proteins)
        interaction_matrix = np.zeros((num_proteins, num_proteins))

        # Fill the interaction matrix with the calculated distances (as scaled sizes)
        for (protein1, protein2, distance) in edges:
            i = proteins.get_loc(protein1)
            j = proteins.get_loc(protein2)
            circle_size = scale_circle_size(distance)
            interaction_matrix[i, j] = circle_size
            interaction_matrix[j, i] = circle_size  # Mirror the interaction for symmetric matrix

        # Set the max circle size for protein with itself (diagonal values)
        for i in range(num_proteins):
            if proteins[i] in centroids:
                interaction_matrix[i, i] = scale_circle_size(0)  # Max size for the same protein (self-comparison)
            else:
                interaction_matrix[i, i] = scale_circle_size(max_possible_distance)  # Max size if no centroid

        # Calculate the correlation matrix for protein expressions (Spearman) for the filtered region
        correlation_matrix = filtered_data[proteins].corr(method='spearman')

        # Invert both the data and the labels on the y-axis
        interaction_matrix = interaction_matrix[::-1, :]  # Flip the interaction matrix vertically
        correlation_matrix = correlation_matrix.iloc[::-1, :]  # Flip the correlation matrix vertically

        # Visualize the heatmap with inverted y-axis labels and data
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-0.25, vmax=0.25, annot=False,
                        xticklabels=proteins, yticklabels=proteins[::-1])  # Invert y-axis labels

        # Set the font properties for the x and y tick labels
        plt.xticks(rotation=90, fontsize=16, fontweight='bold', fontname='Arial')  # Customize x-axis labels
        plt.yticks(fontsize=16, fontweight='bold', fontname='Arial')  # Customize y-axis labels
        plt.subplots_adjust(bottom=0.20, left=0.20)
        # Add circles proportional to the interaction strengths (edge lengths)
        for i in range(num_proteins):
            for j in range(num_proteins):
                if interaction_matrix[i, j] > 0:  # Only add circles for actual interactions
                    ax.add_patch(
                        plt.Circle((j + 0.5, i + 0.5), radius=interaction_matrix[i, j] / 1000, color='turquoise', fill=True))

        plt.title(f'Protein Correlation Heatmap for Region ({x_min},{y_min}) to ({x_max},{y_max})')
        plt.show()
    else:
        print("No protein pairs found for analysis.")




