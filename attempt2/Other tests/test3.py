import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
3D STL Scaling, Slicing, and Visualization using Matplotlib

This script demonstrates the process of:
1. Scaling an STL file by a given factor.
2. Slicing the scaled STL file into horizontal layers at specified intervals along the Z-axis.
3. Visualizing these 3D slices using Matplotlib, with each slice represented by line segments at different Z-levels,
   using a 'viridis' colormap for layer-specific colours.

Key Functions:
- `scale_stl`: Scales an STL file by a specified factor.
- `slice_stl_3d`: Slices an STL file into horizontal layers.
- `intersect_plane_with_triangle`: Computes intersections of slicing planes with triangle faces of the mesh.
- `visualise_slices_3d_matplotlib`: Visualizes the sliced layers in 3D using Matplotlib.

Dependencies:
- NumPy: For numerical operations and array manipulation.
- STL (from stl import mesh): For loading and processing STL files.
- Matplotlib: For 3D data visualization using Axes3D and cm (colormap).

Usage:
1. Replace `original_file_path` and `scaled_file_path` with your STL file paths.
2. Adjust `scale_factor` to control the scaling factor for the STL model.
3. Set `layer_height` to determine the interval between slicing planes.
4. Run the script to scale the STL file, slice it into layers, and visualize the slices using Matplotlib.

Note:
- Ensure the STL file paths are correctly specified.
- Matplotlib and NumPy libraries need to be installed (`pip install matplotlib numpy`).
"""


def scale_stl(file_path, output_path, scale_factor):
    """
    Scale the STL file by a given factor.
    
    Args:
        file_path (str): Path to the original STL file.
        output_path (str): Path to save the scaled STL file.
        scale_factor (float): Factor to scale the STL model.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Scale the vertices
    stl_mesh.vectors *= scale_factor
    
    # Save the scaled STL file
    stl_mesh.save(output_path)
    print(f"Scaled STL file saved at: {output_path}")


def slice_stl_3d(file_path, layer_height):
    """
    Slice the STL file into horizontal layers.
    
    Args:
        file_path (str): Path to the STL file.
        layer_height (float): Height between slicing planes.
    
    Returns:
        list: A list of slices, where each slice contains Z-level and line segments.
    """
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
    """
    Find intersections of a horizontal plane with a triangle.
    
    Args:
        z (float): Z-level of the slicing plane.
        triangle (numpy.ndarray): Vertices of the triangle.
    
    Returns:
        list: A list of intersection points (if any).
    """
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
original_file_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array.stl"
scaled_file_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl"
scale_factor = 1 / 1000  # Scale down by 1000

# Scale the STL file
scale_stl(original_file_path, scaled_file_path, scale_factor)

# Slice the scaled STL file
layer_height = 0.0006  # Example layer height
slices = slice_stl_3d(scaled_file_path, layer_height)

# Visualise the slices
visualise_slices_3d_matplotlib(slices)
