import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colormaps
"""
3D Scaling, Slicing, and Visualization of STL Mesh using Matplotlib

This script performs three main tasks on an STL mesh file:
1. Scales the STL model by a specified factor (e.g., microns to millimetres).
2. Slices the scaled STL model into horizontal layers based on a given layer height.
3. Visualizes the sliced layers in 3D using Matplotlib.

Key Functionalities:
- Scale an STL file by applying a scaling factor to all vertex coordinates.
- Slice the scaled STL model into horizontal layers using slicing planes.
- Compute intersections between slicing planes and mesh triangles to generate line segments.
- Visualize the 3D slices with layer-specific colours in Matplotlib.

Dependencies:
- NumPy: For numerical operations and array manipulation.
- STL (from stl import mesh): For loading and processing STL files.
- Matplotlib: For 3D data visualization using Axes3D and colormaps.

Usage:
1. Replace `"/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array.stl"` with the actual path to your original STL file.
2. Replace `"/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl"` with the path to save the scaled STL file.
3. Adjust `scale_factor` to specify how much to scale the STL model (e.g., 1 / 1000 for microns to millimetres).
4. Specify `layer_height` in millimetres to control the interval between slicing planes or leave it as None to auto-calculate.
5. Run the script to scale the STL file, slice it into layers, and visualize the 3D slices using Matplotlib.
   The visualization displays line segments at different Z-levels, with each layer represented by a unique colour.

Note:
- Ensure the STL file paths are correctly specified.
- Matplotlib and NumPy libraries need to be installed (`pip install matplotlib numpy`).
- The script assumes a valid STL model with non-zero Z-depth for slicing.
"""

# Function to scale the STL file
def scale_stl(file_path, output_path, scale_factor):
    """
    Scale an STL file by a given factor.
    
    Args:
        file_path (str): Path to the original STL file.
        output_path (str): Path to save the scaled STL file.
        scale_factor (float): Factor to scale the STL model.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Scale the vertices (apply the scale factor to all points)
    stl_mesh.vectors *= scale_factor
    
    # Save the scaled STL file to a new location
    stl_mesh.save(output_path)
    print(f"Scaled STL file saved at: {output_path}")

# Function to slice the STL file into horizontal layers
def slice_stl_3d(file_path, layer_height=None):
    """
    Slice an STL file into horizontal layers based on a specified layer height.
    
    Args:
        file_path (str): Path to the STL file.
        layer_height (float, optional): Height between slicing planes. If None, auto-calculated.

    Returns:
        list: A list of slices, where each slice contains the Z-level and line segments.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    facets = stl_mesh.vectors  # Extract triangle vertices from the mesh

    # Determine the Z-bounds of the model
    z_min = np.min(facets[:, :, 2])
    z_max = np.max(facets[:, :, 2])
    z_range = z_max - z_min

    # Ensure the STL model has a valid Z-range
    if z_range == 0:
        raise ValueError("The STL model appears to have no Z-depth. Check the input file.")

    # Automatically determine layer height if not provided
    if layer_height is None:
        layer_height = z_range / 100.0  # Divide the Z-range into 100 layers by default
        print(f"Automatically determined layer_height: {layer_height:.6f}")

    # Generate slicing planes based on the Z-bounds and layer height
    z_levels = np.arange(z_min, z_max + layer_height, layer_height)

    # Store slices
    slices = []
    for z in z_levels:
        layer_lines = []
        for triangle in facets:
            # Skip triangles that cannot intersect this Z-plane
            if np.min(triangle[:, 2]) > z or np.max(triangle[:, 2]) < z:
                continue

            # Find intersections of the slicing plane with the triangle
            intersections = intersect_plane_with_triangle(z, triangle)
            if intersections is not None:
                layer_lines.append(intersections)
        slices.append((z, layer_lines))  # Append the Z-level and its corresponding line segments
    
    return slices

# Function to compute intersections between a slicing plane and a triangle
def intersect_plane_with_triangle(z, triangle):
    """
    Find the intersection of a horizontal slicing plane with a triangle.
    
    Args:
        z (float): Z-level of the slicing plane.
        triangle (numpy.ndarray): Array containing the 3 vertices of the triangle.

    Returns:
        list: A list of two intersection points (if any).
    """
    # Define the edges of the triangle
    edges = [(triangle[i], triangle[(i + 1) % 3]) for i in range(3)]
    points = []
    for p1, p2 in edges:
        # Check if the edge crosses the slicing plane
        if (p1[2] <= z and p2[2] >= z) or (p1[2] >= z and p2[2] <= z):
            # Compute the intersection point using linear interpolation
            t = (z - p1[2]) / (p2[2] - p1[2]) if p1[2] != p2[2] else 0
            intersect_point = p1 + t * (p2 - p1)
            points.append(intersect_point)
    # Return the intersection points if exactly two are found
    if len(points) == 2:
        return points
    return None

# Function to visualise the slices in 3D using Matplotlib
def visualise_slices_3d_matplotlib(slices):
    """
    Visualise the 3D slices using Matplotlib.
    
    Args:
        slices (list): List of tuples where each tuple contains a Z-level and line segments at that level.
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colour map to assign different colours to each layer
    colour_map = colormaps.get_cmap("viridis", len(slices))  # Updated for Matplotlib 3.7+


    for i, (z, lines) in enumerate(slices):
        for line in lines:
            x = [line[0][0], line[1][0]]  # X-coordinates of the line
            y = [line[0][1], line[1][1]]  # Y-coordinates of the line
            z_coords = [z, z]  # Fixed Z level

            # Plot the line segment with a layer-specific colour
            ax.plot(x, y, z_coords, color=colour_map(i), linewidth=0.2)

    # Set axis labels and title
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Z-axis (Layer Height)', fontsize=12)
    ax.set_title("3D Sliced Layers", fontsize=15)

    # Show the plot
    plt.show()

# Main script: Scale, slice, and visualise the STL file
if __name__ == "__main__":
    # Path to the original STL file
    original_file_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array.stl"

    # Path to save the scaled STL file
    scaled_file_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl"

    # Scale factor: Reduce size by 1000 (e.g., microns to millimetres)
    scale_factor = 1 / 1000

    # Scale the STL file
    scale_stl(original_file_path, scaled_file_path, scale_factor)

    # Slice the scaled STL file
    # Optionally, specify a layer height (e.g., 0.001 for 1 micron); otherwise, it will auto-calculate
    layer_height = 0.001  # 1 micron in millimetres
    slices = slice_stl_3d(scaled_file_path, layer_height)

    # Visualise the slices in 3D
    visualise_slices_3d_matplotlib(slices)
