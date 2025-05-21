import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
"""
STL Mesh Slicing and Visualization

This script slices a 3D STL mesh into 2D layers at specified intervals along the Z-axis
and visualizes these slices using Matplotlib. It computes intersections of horizontal
slicing planes with the mesh triangles to generate polygons representing each slice.

Key Functionalities:
- Loads an STL mesh file and extracts triangle vertices.
- Defines slicing planes along the Z-axis at specified intervals (`layer_height`).
- Computes intersections of these slicing planes with mesh triangles to generate 2D polygons.
- Visualizes the generated slices using Matplotlib, where each slice is represented by a 
  colored polygon collection.

Dependencies:
- NumPy: For numerical operations and array manipulation.
- STL (from stl import mesh): For loading and processing STL files.
- Matplotlib: For data visualization, specifically for plotting polygons.

Usage:
1. Replace `"path_to_your_scaled_model.stl"` with the actual path to your STL file.
2. Adjust `layer_height` to control the interval between slicing planes.
3. Run the script to generate and visualize the slices of the STL mesh.
   The visualization displays each slice as a collection of polygons with colors indicating
   different Z-levels based on the viridis colormap.

Note:
- Ensure the STL file path is correctly specified.
- Matplotlib and NumPy libraries need to be installed (`pip install matplotlib numpy`).
"""

def slice_stl(file_path, layer_height):
    # Load STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    facets = stl_mesh.vectors  # Extract triangle vertices
    
    # Determine Z bounds
    z_min = np.min(facets[:, :, 2])
    z_max = np.max(facets[:, :, 2])
    
    # Generate slicing planes
    z_levels = np.arange(z_min, z_max, layer_height)
    
    # Store slices
    slices = []
    for z in z_levels:
        layer_polygons = []
        for triangle in facets:
            intersections = intersect_plane_with_triangle(z, triangle)
            if intersections is not None:
                layer_polygons.append(intersections)
        slices.append((z, layer_polygons))
    
    return slices

def intersect_plane_with_triangle(z, triangle):
    """Find intersections of a horizontal plane with a triangle."""
    edges = [(triangle[i], triangle[(i+1) % 3]) for i in range(3)]
    points = []
    for p1, p2 in edges:
        if (p1[2] <= z and p2[2] >= z) or (p1[2] >= z and p2[2] <= z):
            t = (z - p1[2]) / (p2[2] - p1[2]) if p1[2] != p2[2] else 0
            intersect_point = p1 + t * (p2 - p1)
            points.append(intersect_point[:2])  # Only need X, Y
    if len(points) == 2:
        return points
    return None

def visualise_slices(slices):
    colours = plt.cm.viridis(np.linspace(0, 1, len(slices)))
    fig, ax = plt.subplots(figsize=(8, 8))
    for (z, polygons), colour in zip(slices, colours):
        patches = [Polygon(poly) for poly in polygons]
        p = PatchCollection(patches, facecolor=colour, edgecolor="black", alpha=0.6)
        ax.add_collection(p)
    ax.autoscale()
    plt.show()

# Usage example
file_path = "path_to_your_scaled_model.stl"
layer_height = 1.0  # Example layer height
slices = slice_stl(file_path, layer_height)
visualise_slices(slices)
