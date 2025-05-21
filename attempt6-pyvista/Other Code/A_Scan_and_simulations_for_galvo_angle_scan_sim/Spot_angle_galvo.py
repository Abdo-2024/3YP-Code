import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""
This script uses Matplotlib to draw a 3.6 × 3.6 mm stage and plot sample galvo mirror deflection positions. It maps normalized deflection values (–0.5 to 0.5) to millimetre coordinates on the stage, draws small red circles of 7 µm diameter at each mapped position, and labels them with their deflection values, all on an equal‐aspect grid.
"""

# Stage parameters: 3.6 x 3.6 mm stage (centered at 0)
width_mm = 3.6
height_mm = 3.6

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-width_mm/2, width_mm/2)
ax.set_ylim(-height_mm/2, height_mm/2)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Stage with Sample Galvo Mirror Deflection Positions")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')  # Ensure circles are not elliptical

# Define sample deflections (normalized, where -0.5 maps to -1.8 mm and 0.5 maps to 1.8 mm)
sample_deflections = [
    (-0.5, -0.5),   # bottom-left corner
    ( 0.5, -0.5),   # bottom-right corner
    (-0.5,  0.5),   # top-left corner
    ( 0.5,  0.5),   # top-right corner
    ( 0.0,  0.0),   # center
    ( 0.25, 0.25),  # additional sample
    (-0.25, -0.25)  # additional sample
]

# Mapping: position (mm) = deflection * stage width
for d_x, d_y in sample_deflections:
    pos_x_mm = d_x * width_mm
    pos_y_mm = d_y * height_mm
    # Create a circle patch with a 7 µm diameter (7 µm = 0.007 mm; radius = 0.0035 mm)
    dot_radius_mm = 0.007 / 2
    circle_patch = patches.Circle((pos_x_mm, pos_y_mm), radius=dot_radius_mm, color='red', fill=True)
    ax.add_patch(circle_patch)
    # Label the point with its deflection values
    ax.text(pos_x_mm, pos_y_mm, f" {d_x}, {d_y}", color='red', fontsize=10,
            verticalalignment='bottom', horizontalalignment='left')

plt.show()
