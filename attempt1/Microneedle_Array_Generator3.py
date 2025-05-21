"""
Microneedle Array Generator using SolidPython

This script generates a 3D model of a microneedle array using parametric design.
Each microneedle is modelled as a hollow cone with a specified wall thickness.
The array is placed on a square base, and visualised in 10 colour-coded vertical layers 
for better clarity and inspection.

Key Features:
- Parametric control of microneedle dimensions, spacing, and array size
- Hollow interior for simulating drug delivery channels
- Colour-coded layers for clear visual separation of needle height segments
- Exports the final model as a .scad file for use with OpenSCAD or 3D printing

Dependencies:
- SolidPython (https://github.com/SolidCode/SolidPython)

Output:
- A file named 'microneedle_model.scad' containing the full model
"""

from solid import *
from solid.utils import *

# Microneedle design parameters
needle_height = 500  # Height of each microneedle in microns
base_diameter = 175  # Diameter of the base of each microneedle in microns
base_radius = base_diameter / 2
wall_thickness = 10  # Thickness of the microneedle wall in microns
inner_tip_diameter = 5  # Diameter of the inner tip of each microneedle in microns
inner_tip_radius = inner_tip_diameter / 2
base_thickness = 10  # Thickness of the square base in microns

# Base dimensions
base_size = 3600  # Size of the square base (3.6 mm) in microns
array_size = 10  # Number of microneedles per side in the array
inter_spacing = 200  # Spacing between microneedles in microns

# Layer properties
layer_height = needle_height // 10  # Height of each layer for visualisation


def create_hollow_cone():
    """Creates a single hollow microneedle."""
    # Outer cone representing the microneedle shape
    outer_cone = cone(h=needle_height, r1=base_radius, r2=inner_tip_radius)
    # Inner cone subtracted from the outer cone to create the hollow structure
    inner_cone = cone(h=needle_height, r1=base_radius - wall_thickness, r2=0)
    return difference()(outer_cone, inner_cone)


def create_needle_array():
    """Creates the full array of hollow microneedles."""
    needle = create_hollow_cone()
    needles = []
    # Generate microneedle array in a grid pattern
    for i in range(array_size):
        for j in range(array_size):
            x_offset = i * (base_diameter + inter_spacing)
            y_offset = j * (base_diameter + inter_spacing)
            # Translate each microneedle to its position in the array
            needles.append(translate([x_offset, y_offset, base_thickness])(needle))
    return union()(*needles)


def create_coloured_layers():
    """Creates layers with different colours for visualisation."""
    layers = []
    colours = ["blue", "orange", "green", "red", "purple", "yellow", "cyan", "pink", "grey", "white"]
    # Create multiple layers with different colours
    for i in range(10):
        # Translate each layer vertically to stack them
        layer = translate([0, 0, i * layer_height])(
            color(colours[i % len(colours)])(create_needle_array())
        )
        layers.append(layer)
    return union()(*layers)


def create_square_base():
    """Creates the square base to hold the microneedle array."""
    return cube([base_size, base_size, base_thickness], center=False)


def create_full_model():
    """Combines the base and microneedles into a full 3D model."""
    base = create_square_base()
    needles = create_coloured_layers()
    # Combine base and microneedle layers into a single 3D model
    return union()(base, needles)

if __name__ == "__main__":
    # Generate the full 3D model
    model = create_full_model()
    # Export the model to an OpenSCAD file
    scad_render_to_file(model, "microneedle_model.scad")
    print("3D model has been saved as microneedle_model.scad")

