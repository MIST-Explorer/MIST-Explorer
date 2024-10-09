import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools

def Neighbor():
    # Load your dataset from the Excel file
    data = pd.read_excel(r'C:\Users\meaha\Desktop\Warped Cell Data\CompleteBiopsyR1&2.xlsx')

    # List of proteins to analyze
    proteins = ['Vimentin', 'CD20', 'CD3', 'CD206', 'CD163']  # Protein list

    # Set the fixed color for the first protein interactions (e.g., Vimentin interactions)
    first_protein = proteins[0]
    fixed_color = 'red'  # Set the fixed color for the first protein's interactions
    other_color = 'blue'  # Set the color for all other interactions


    # Function to generate colors for protein interactions
    def generate_interaction_colors(proteins):
        """Generates a color for each unique pair of proteins. Red for the first protein's interactions, blue for others."""
        interaction_colors = {}

        # Assign the fixed color for interactions involving the first protein
        for pair in itertools.combinations(proteins, 2):
            if first_protein in pair:
                interaction_colors[pair] = fixed_color
            else:
                interaction_colors[pair] = other_color

        return interaction_colors


    # Generate the interaction colors dynamically
    interaction_colors = generate_interaction_colors(proteins)


    # Function to filter cells within a specified region
    def filter_region(data, x1, y1, x2, y2):
        """Filters the dataset to include only cells within the specified region."""
        filtered_data = data[(data['Global X'] >= x1) & (data['Global X'] <= x2) &
                            (data['Global Y'] >= y1) & (data['Global Y'] <= y2)]
        print(f"Number of cells in the selected region: {len(filtered_data)}")
        return filtered_data


    # Function to calculate weighted centroids and average protein expression for each protein
    def calculate_weighted_centroids(data, proteins):
        centroids = {}
        avg_expression = {}
        for protein in proteins:
            # Filter cells expressing the current protein
            protein_cells = data[data[protein] > 100]
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
                print(f"No cells express {protein} in this region.")
        return centroids, avg_expression


    # Function to rescale node sizes based on average protein expression
    def rescale_sizes(avg_expression, min_size, max_size):
        """Rescale the average expression values to fit between min_size and max_size."""
        values = np.array(list(avg_expression.values()))
        min_exp, max_exp = np.min(values), np.max(values)
        # Normalize the expressions to the range [0, 1]
        if max_exp != min_exp:  # Avoid division by zero if all values are the same
            norm_expression = (values - min_exp) / (max_exp - min_exp)
        else:
            norm_expression = np.ones_like(values)  # Set all to 1 if values are identical
        # Scale the normalized values to the range [min_size, max_size]
        scaled_sizes = min_size + norm_expression * (max_size - min_size)
        return dict(zip(avg_expression.keys(), scaled_sizes))


    # Define the region you want to check (you can adjust these values)
    x1, y1 = 0, 0  # Top-left corner
    x2, y2 = 12000, 12000  # Bottom-right corner (adjust these values as needed)

    # Filter the dataset for the region
    filtered_data = filter_region(data, x1, y1, x2, y2)

    # Calculate weighted centroids and average expression for each protein in the filtered region
    centroids, avg_expression = calculate_weighted_centroids(filtered_data, proteins)

    # Define unique colors for each protein node using a colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'purple', 'yellow']  # Colorblind-friendly palette

    # Define minimum and maximum node sizes
    min_size = 12  # Minimum node size
    max_size = 45  # Maximum node size

    # Rescale the node sizes based on average expression, constrained by min_size and max_size
    node_sizes = rescale_sizes(avg_expression, min_size, max_size)

    # Print the centroids and average expressions
    print("\nCentroids and average expression of proteins in the specified region:")
    for protein, coord in centroids.items():
        print(f"{protein}: {coord}, Avg Expression: {avg_expression[protein]:.2f}, Node Size: {node_sizes[protein]:.2f}")

    # Create 3D Scatter plot of the nodes (using X, Y, and Z for positioning)
    node_x = []
    node_y = []
    node_z = []  # Add Z values as well to spread out in 3D space
    node_colors = []  # Assign colors to nodes based on proteins

    # Assign X, Y, Z coordinates for each protein node
    for i, (protein, (x, y)) in enumerate(centroids.items()):
        node_x.append(x)
        node_y.append(y)
        node_z.append(i * 10)  # Use arbitrary Z-coordinates for better separation in 3D
        node_colors.append(colors[i])  # Assign a unique color for each protein

    # Create a list of edges, distances, and corresponding edge colors
    edges = []
    edge_labels = []
    edge_colors = []
    for i, protein1 in enumerate(centroids):
        for j, protein2 in enumerate(list(centroids)[i + 1:]):
            x1, y1 = centroids[protein1]
            x2, y2 = centroids[protein2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Get the color for the interaction (red for first protein, blue for others)
            color = interaction_colors.get((protein1, protein2), interaction_colors.get((protein2, protein1), 'blue'))

            edges.append((protein1, protein2, distance))
            edge_labels.append(f'{distance:.2f}')
            edge_colors.append(color)  # Assign the color for this edge

    # Create the 3D Scatter plot using Plotly
    fig = go.Figure()

    # Add nodes (proteins) with sizes and colors
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=[node_sizes[protein] for protein in proteins],
            color=node_colors,
            opacity=0.9,  # Slight transparency to give depth
            symbol='circle',  # Marker shape as circles
        ),
        text=proteins,  # Display only protein names
        textposition="bottom center",  # Position the label below the node
        textfont=dict(size=12, color="white"),  # Adjust font size and color
        hoverinfo="text"  # Show only node labels on hover
    ))

    # Add edges (lines) connecting the nodes with different colors based on protein interaction
    for (protein1, protein2, distance), label, color in zip(edges, edge_labels, edge_colors):
        # Get the coordinates of the two proteins
        i = proteins.index(protein1)
        j = proteins.index(protein2)

        # Calculate the mid-point of the line to place the label above the center
        mid_x = (node_x[i] + node_x[j]) / 2
        mid_y = (node_y[i] + node_y[j]) / 2
        mid_z = (node_z[i] + node_z[j]) / 2 + 5  # Slightly above the line for better clarity

        # Add the line (edge) representing the actual spatial distance between the proteins
        fig.add_trace(go.Scatter3d(
            x=[node_x[i], node_x[j]],
            y=[node_y[i], node_y[j]],
            z=[node_z[i], node_z[j]],  # Z-coordinates for 3D
            mode='lines',
            line=dict(color=color, width=4),  # Color based on interaction
            hoverinfo='none'  # Disable hover for lines
        ))

        # Add text labels to show distances in the middle of the edges
        # fig.add_trace(go.Scatter3d(
        #     x=[mid_x], y=[mid_y], z=[mid_z],
        #     mode='text',
        #     text=[label],
        #     textposition="middle center",
        #     textfont=dict(size=10, color='white'),  # Adjust font for edge labels
        #     hoverinfo='none'
        # ))

    # Update layout for better visualization
    fig.update_layout(
        title="3D Protein Network",
        scene=dict(
            xaxis=dict(showaxeslabels=False, visible=False),  # Hide X axis labels
            yaxis=dict(showaxeslabels=False, visible=False),  # Hide Y axis labels
            zaxis=dict(showaxeslabels=False, visible=False),  # Hide Z axis labels
            # Black background
            bgcolor='rgba(0, 0, 0, 1)',
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        font=dict(family="Arial, sans-serif", size=14, color="white"),  # Font adjustments
        paper_bgcolor='black',  # Black background for publication
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Adjust the initial view for better framing
        showlegend=False,  # Hide legend
    )

    # Show the plot
    fig.show()

    # Print the distances between the proteins
    print("\nDistances between proteins in the specified region:")
    for (protein1, protein2, distance) in edges:
        print(f"Distance between {protein1} and {protein2}: {distance:.2f}")





