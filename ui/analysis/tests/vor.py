import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generate random points
points = np.random.rand(10, 2)  # 10 points in 2D

# Compute Voronoi diagram
vor = Voronoi(points)

# Get region bounds
regions = {}
for i, region_index in enumerate(vor.point_region):
    vertices = vor.regions[region_index]
    if -1 in vertices:  # Ignore infinite regions
        continue
    regions[tuple(points[i])] = vor.vertices[vertices]

# Print region bounds
for point, vertices in regions.items():
    print(f"Point {point} has region bounds: {vertices}")

# Plot Voronoi diagram
fig, ax = plt.subplots()
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue')

# Plot original points
ax.plot(points[:, 0], points[:, 1], 'ro')

plt.show()