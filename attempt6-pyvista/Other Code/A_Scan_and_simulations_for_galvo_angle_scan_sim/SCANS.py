#!/usr/bin/env python
"""
Improved OCT simulation code:
• Generates a 2D full-array intensity map.
• Builds a 3D sample from a base block with cones.
• Extracts:
   - An A-scan (vertical profile through a 7 µm cross-section),
   - A B-scan (vertical slice obtained by sampling vertical lines with 7 µm lateral spacing over 3.6 mm),
   - A C-scan (an en-face slice; interpolated onto a 3.6 mm × 3.6 mm grid at 7 µm resolution).
• Displays all four plots in one figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import griddata
from tqdm import tqdm  # For progress bars

# ------------------------------
# 2D functions (from Intensity.py)
# ------------------------------
def circle_intensity(r, r_inner, r_outer):
    """
    Compute a piecewise intensity:
      - 0 for r <= r_inner,
      - Linear ramp for r between r_inner and r_outer,
      - 1 for r >= r_outer.
    """
    I = np.ones_like(r)
    I[r <= r_inner] = 0.0
    between = (r > r_inner) & (r < r_outer)
    I[between] = (r[between] - r_inner) / (r_outer - r_inner)
    return I

def create_pattern(width_mm=3.6, height_mm=3.6, r_inner_um=7.5, r_outer_um=54.5,
                   resolution_um=7.0, grid_size=10):
    """
    Create a 2D intensity map (full array) composed of many circles.
    Returns:
      xx, yy: coordinate grids (in µm),
      image: 2D intensity array,
      circle_centers: list of (x, y) circle center positions.
    """
    width_um = width_mm * 1000.0   # 3600 µm
    height_um = height_mm * 1000.0   # 3600 µm
    nx = int(width_um / resolution_um)
    ny = int(height_um / resolution_um)
    x_vals = np.linspace(-width_um/2, width_um/2, nx)
    y_vals = np.linspace(-height_um/2, height_um/2, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    
    # Vectorized computation of image
    image = np.ones((ny, nx))
    
    # Calculate circle centers
    margin = 200.0  # µm margin
    x_centers = np.linspace(-width_um/2 + margin, width_um/2 - margin, grid_size)
    y_centers = np.linspace(-height_um/2 + margin, height_um/2 - margin, grid_size)
    circle_centers = [(x, y) for x in x_centers for y in y_centers]
    
    # Vectorized calculation for all circles at once
    for (cx, cy) in circle_centers:
        rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        circle_vals = circle_intensity(rr, r_inner_um, r_outer_um)
        image = np.minimum(image, circle_vals)
        
    return xx, yy, image, circle_centers

# ------------------------------
# 3D functions (using PyVista more efficiently)
# ------------------------------
def create_3d_sample(width_mm=3.6, height_mm=3.6, resolution_um=7.0, grid_size=10):
    """
    Build a 3D sample by creating a base block and adding cones at grid positions.
    Uses PyVista's built-in primitives for better performance.
    
    Returns:
      sample_mesh: the merged pyvista mesh,
      centers: list of (x, y) positions of the cones.
    """
    width_um = width_mm * 1000.0  # 3600 µm
    height_um = height_mm * 1000.0  # 3600 µm
    thickness_um = 550.0
    
    # Create the base block with PyVista's built-in cube
    base = pv.Cube(
        center=(0, 0, thickness_um / 2.0),
        x_length=width_um, 
        y_length=height_um, 
        z_length=thickness_um
    )
    base["scalars"] = np.full(base.n_points, 0.8)
    
    # Calculate cone positions on a grid
    margin = 200.0
    x_positions = np.linspace(-width_um/2 + margin, width_um/2 - margin, grid_size)
    y_positions = np.linspace(-height_um/2 + margin, height_um/2 - margin, grid_size)
    centers = [(x, y) for x in x_positions for y in y_positions]
    
    # Initialize sample mesh with the base
    sample_mesh = base
    
    # Cone parameters (in µm)
    needle_height = 450.0
    bottom_radius = 109.0   # outer radius (white edge)
    top_radius = 15.0       # inner radius (black center)
    
    # Create a template cone once with PyVista's built-in cone primitive
    # The template will be transformed for each position
    resolution = 60
    template_cone = pv.Cone(
        center=(0, 0, needle_height/2),
        direction=(0, 0, 1),
        height=needle_height,
        radius=bottom_radius,
        resolution=resolution,
        capping=True
    )
    
    # Add scalar values based on radial distance
    cone_points = template_cone.points
    r = np.sqrt(cone_points[:, 0]**2 + cone_points[:, 1]**2)
    template_cone["scalars"] = circle_intensity(r, top_radius, bottom_radius)
    
    # Add cones to the base at each center position
    print("Building 3D sample...")
    for (cx, cy) in tqdm(centers, desc="Adding cones"):
        # Clone the template cone
        cone = template_cone.copy()
        # Translate to the correct position
        cone.translate((cx, cy, thickness_um))
        # Merge with the sample mesh
        sample_mesh = sample_mesh.merge(cone)
    
    return sample_mesh, centers

# ------------------------------
# Sampling and scan generation functions
# ------------------------------
def generate_a_scan(sample_mesh, x, y, z_start=550, z_end=1050, resolution_um=7.0):
    """Generate an A-scan at the specified (x,y) position."""
    n_z = int((z_end - z_start) / resolution_um) + 1
    point_a = [x, y, z_start]
    point_b = [x, y, z_end]
    
    # Create points along the line manually for compatibility with older PyVista versions
    t_vals = np.linspace(0, 1, n_z)
    line_points = np.array([
        point_a[0] + t * (point_b[0] - point_a[0]),
        point_a[1] + t * (point_b[1] - point_a[1]),
        point_a[2] + t * (point_b[2] - point_a[2])
    ]).T
    
    # Sample the points
    a_scan_values = []
    a_scan_z = line_points[:, 2]
    
    for point in line_points:
        # Use probe instead of sample_over_line
        value = sample_mesh.probe(point)["scalars"]
        a_scan_values.append(value)
    
    return a_scan_z, np.array(a_scan_values)

def generate_b_scan(sample_mesh, x, width_um=3600.0, z_start=550, z_end=1050, resolution_um=7.0):
    """Generate a B-scan at the specified x position."""
    n_z = int((z_end - z_start) / resolution_um) + 1
    y_vals = np.linspace(-width_um/2, width_um/2, int(width_um/resolution_um))
    n_y = len(y_vals)
    b_scan = np.zeros((n_z, n_y))
    z_vals = np.linspace(z_start, z_end, n_z)
    
    print("Generating B-scan...")
    for i, y in tqdm(enumerate(y_vals), total=n_y, desc="Sampling lines"):
        for j, z in enumerate(z_vals):
            # Sample each point individually
            point = [x, y, z]
            try:
                value = sample_mesh.probe(point)["scalars"]
                b_scan[j, i] = value
            except:
                b_scan[j, i] = 0.8  # Default background value if probing fails
    
    return y_vals, z_vals, b_scan

def generate_c_scan(sample_mesh, slice_z=800, width_um=3600.0, resolution_um=7.0):
    """Generate a C-scan (en-face slice) at the specified z height."""
    # Create a slice through the sample at the specified z
    print("Generating C-scan...")
    c_slice = sample_mesh.slice(normal=[0, 0, 1], origin=(0, 0, slice_z))
    pts = c_slice.points
    scalars = c_slice["scalars"]
    
    # Interpolate onto a regular grid
    nx = ny = int(width_um / resolution_um)
    x_vals_grid = np.linspace(-width_um/2, width_um/2, nx)
    y_vals_grid = np.linspace(-width_um/2, width_um/2, ny)
    X, Y = np.meshgrid(x_vals_grid, y_vals_grid)
    
    c_scan_grid = griddata(pts[:, :2], scalars, (X, Y), method='linear', fill_value=0.8)
    return x_vals_grid, y_vals_grid, c_scan_grid

# ------------------------------
# Main routine: generate scans and plot
# ------------------------------
def main():
    # Parameters
    width_mm = 3.6
    height_mm = 3.6
    resolution_um = 7.0
    grid_size = 10
    
    # Generate the 2D intensity pattern
    print("Generating 2D intensity pattern...")
    xx, yy, image, circle_centers = create_pattern(
        width_mm=width_mm, 
        height_mm=height_mm, 
        resolution_um=resolution_um,
        grid_size=grid_size
    )
    
    # Generate the 3D sample mesh using PyVista's built-in primitives
    sample_mesh, centers = create_3d_sample(
        width_mm=width_mm,
        height_mm=height_mm,
        resolution_um=resolution_um,
        grid_size=grid_size
    )
    
    # Choose a point from a central cone for the A-scan and B-scan
    central_cone = centers[len(centers) // 2]
    a_scan_x = central_cone[0] + 20  # offset by 20 µm
    a_scan_y = central_cone[1]
    
    # Generate A-scan
    print("Generating A-scan...")
    a_scan_z, a_scan_intensity = generate_a_scan(
        sample_mesh, 
        a_scan_x, 
        a_scan_y,
        resolution_um=resolution_um
    )
    
    # Generate B-scan
    y_vals, z_vals, b_scan = generate_b_scan(
        sample_mesh, 
        a_scan_x,
        width_um=width_mm*1000.0,
        resolution_um=resolution_um
    )
    
    # Generate C-scan
    x_vals_c, y_vals_c, c_scan_grid = generate_c_scan(
        sample_mesh,
        slice_z=800,
        width_um=width_mm*1000.0,
        resolution_um=resolution_um
    )
    
    # ----- Plotting -----
    print("Creating plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Full array intensity (2D pattern)
    im0 = axs[0, 0].imshow(image, extent=[xx[0, 0], xx[0, -1], yy[0, 0], yy[-1, 0]],
                          origin='lower', cmap='gray')
    axs[0, 0].set_title("Full Array Intensity (2D Pattern)")
    axs[0, 0].set_xlabel("x (µm)")
    axs[0, 0].set_ylabel("y (µm)")
    fig.colorbar(im0, ax=axs[0, 0])
    
    # Plot 2: A-scan (vertical profile)
    axs[0, 1].plot(a_scan_intensity, a_scan_z, marker='o', markersize=3)
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_title("A-scan (Vertical Profile)")
    axs[0, 1].set_xlabel("Intensity")
    axs[0, 1].set_ylabel("Depth (µm)")
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: B-scan (vertical slice)
    width_um = width_mm * 1000.0
    im2 = axs[1, 0].imshow(b_scan, extent=[-width_um/2, width_um/2, z_vals[-1], z_vals[0]],
                          aspect='auto', cmap='gray')
    axs[1, 0].set_title("B-scan (Vertical Slice)")
    axs[1, 0].set_xlabel("y (µm)")
    axs[1, 0].set_ylabel("Depth (µm)")
    fig.colorbar(im2, ax=axs[1, 0])
    
    # Plot 4: C-scan (en-face horizontal slice at z = 800 µm)
    im3 = axs[1, 1].imshow(c_scan_grid, extent=[-width_um/2, width_um/2, -width_um/2, width_um/2],
                          origin='lower', cmap='gray')
    axs[1, 1].set_title("C-scan (En-face Slice at z = 800 µm)")
    axs[1, 1].set_xlabel("x (µm)")
    axs[1, 1].set_ylabel("y (µm)")
    fig.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.savefig('oct_simulation_results.png', dpi=300)
    print("Plot saved as 'oct_simulation_results.png'")
    plt.show()

if __name__ == "__main__":
    main()