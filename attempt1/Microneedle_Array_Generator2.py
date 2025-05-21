"""
Microneedle Array Generator with PyVista Visualisation

This script generates a 3D visualisation of a 10x10 grid of hollow microneedles 
on a square base platform using PyVista. Each microneedle is modelled as a 
hollow cone formed by subtracting a smaller inner cone from a larger outer one. 

Key Features:
- Parametric control over needle height, base/tip diameters, and grid size
- Hollow design to simulate real-world drug delivery applications
- Triangulated mesh structures for accurate Boolean operations
- Visualisation using a height-based colour map for inspection and analysis

Dependencies:
- NumPy
- PyVista (https://docs.pyvista.org)

Output:
- An interactive 3D rendering of the microneedle array using the Viridis colour map
"""

import numpy as np
import pyvista as pv

# Microneedle design parameters
height = 500  # Height of each needle in microns (um)
base_diameter = 175  # Diameter of the needle base (um)
tip_diameter = 5  # Diameter of the inner tip opening (um)
tip_thickness = 10  # Thickness of the square base platform (um)
grid_size = 10  # 10x10 microneedle array
spacing = 3600 / grid_size  # Distance between each needle centre on the grid (um)

# Create the base platform using a cube and triangulate for mesh operations
base = pv.Cube(bounds=(0, 3600, 0, 3600, 0, tip_thickness)).triangulate()

# Function to create a single hollow conical microneedle
def create_hollow_cone(height, base_d, tip_d, position):
    # Outer cone defining the full external shape
    outer_cone = pv.Cone(
        center=(0, 0, height / 2),
        direction=(0, 0, 1),
        height=height,
        radius=base_d / 2
    ).triangulate()

    # Inner cone used to subtract from the outer to make it hollow
    inner_cone = pv.Cone(
        center=(0, 0, height / 2),
        direction=(0, 0, 1),
        height=height,
        radius=tip_d / 2
    ).triangulate()

    # Boolean subtraction to create hollow structure
    hollow_cone = outer_cone.boolean_difference(inner_cone)
    # Move the hollow cone to its position on the grid
    hollow_cone.translate(position)
    return hollow_cone

# Generate the full microneedle array
microneedles = []
for i in range(grid_size):
    for j in range(grid_size):
        # Calculate position of each needle on the grid
        pos_x = i * spacing + spacing / 2
        pos_y = j * spacing + spacing / 2
        position = (pos_x, pos_y, tip_thickness)
        # Create and store the hollow needle
        needle = create_hollow_cone(height, base_diameter, tip_diameter, position)
        microneedles.append(needle)

# Combine the base with all needles into a single structure
structure = base
for needle in microneedles:
    needle = needle.triangulate()  # Ensure mesh is triangulated before Boolean union
    structure = structure.boolean_union(needle)

# Visualise the structure using a colour map based on height (Z-coordinate)
plotter = pv.Plotter()
structure['Height'] = structure.points[:, 2]  # Add Z-coordinates as scalar field
plotter.add_mesh(structure, scalars='Height', cmap='viridis')  # Apply Viridis colormap
plotter.show()

