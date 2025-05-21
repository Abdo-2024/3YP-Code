import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
3D Slicing and Visualization of STL Mesh using Matplotlib

This script slices a 3D STL mesh into 2D layers at specified intervals along the Z-axis
and visualizes these slices in 3D using Matplotlib. It computes intersections of horizontal
slicing planes with the mesh triangles to generate line segments representing each slice.

Key Functionalities:
- Loads an STL mesh file and extracts triangle vertices.
- Defines slicing planes along the Z-axis at specified intervals (`layer_height`).
- Computes intersections of these slicing planes with mesh triangles to generate 3D line segments.
- Visualizes the 3D slices using Matplotlib, where each slice is represented by line segments
  at different Z-levels, with layer-specific colours from the 'viridis' colormap.

Dependencies:
- NumPy: For numerical operations and array manipulation.
- STL (from stl import mesh): For loading and processing STL files.
- Matplotlib: For 3D data visualization using Axes3D and cm (colormap).

Usage:
1. Replace `"/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array.stl"` with the actual path to your STL file.
2. Adjust `layer_height` to control the interval between slicing planes.
3. Run the script to generate and visualize the 3D slices of the STL mesh using Matplotlib.
   The visualization displays line segments in varying colours along the Z-axis, representing
   intersections between slicing planes and mesh triangles.

Note:
- Ensure the STL file path is correctly specified.
- Matplotlib and NumPy libraries need to be installed (`pip install matplotlib numpy`).
"""


def slice_stl_3d(file_path, layer_height):
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
        layer_lines = []
        for triangle in facets:
            # Skip triangles that cannot intersect this Z-plane
            if np.min(triangle[:, 2]) > z or np.max(triangle[:, 2]) < z:
                continue

            intersections = intersect_plane_with_triangle(z, triangle)
            if intersections is not None:
                layer_lines.append(intersections)
        slices.append((z, layer_lines))
    
    return slices


def intersect_plane_with_triangle(z, triangle):
    """Find intersections of a horizontal plane with a triangle."""
    edges = [(triangle[i], triangle[(i + 1) % 3]) for i in range(3)]
    points = []
    for p1, p2 in edges:
        if (p1[2] <= z and p2[2] >= z) or (p1[2] >= z and p2[2] <= z):
            t = (z - p1[2]) / (p2[2] - p1[2]) if p1[2] != p2[2] else 0
            intersect_point = p1 + t * (p2 - p1)
            points.append(intersect_point)
    if len(points) == 2:
        return points
    return None


def visualise_slices_3d_matplotlib(slices):
    """
    Visualise the 3D slices using matplotlib.
    Args:
        slices (list): List of tuples where each tuple contains a Z-level and line segments at that level.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colour map for layers
    colour_map = cm.get_cmap("viridis", len(slices))

    for i, (z, lines) in enumerate(slices):
        for line in lines:
            x = [line[0][0], line[1][0]]  # X-coordinates
            y = [line[0][1], line[1][1]]  # Y-coordinates
            z_coords = [z, z]  # Fixed Z level

            # Plot the line segment with a layer-specific colour
            ax.plot(x, y, z_coords, color=colour_map(i), linewidth=0.2)

    # Set axis labels
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Z-axis (Layer Height)', fontsize=12)
    ax.set_title("3D Sliced Layers", fontsize=15)

    plt.show()


# Usage example
file_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array.stl"
layer_height = 100  # Example layer height
slices = slice_stl_3d(file_path, layer_height)
visualise_slices_3d_matplotlib(slices)
