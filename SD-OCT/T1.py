import numpy as np
import matplotlib.pyplot as plt
"""
This script constructs a 3D scaled reflectivity and refractive‐index model for OCT simulation on a 310×310×500 mm grid. It assigns a uniform sample refractive index (1.52), sets high reflectivity (1.0) in the bottom 40% (stage and base), and adds four conical reflectivity features in the top 60% by computing axial and lateral decay around specified cone centres. The reflectivity map is clipped to [0,1], and a central axial slice of reflectivity versus depth is plotted. Comments explain how to rescale back to the original micrometre dimensions.
"""

# ----- Parameters (all lengths in mm in the scaled model) -----
# Overall volume dimensions
lat_size = 310    # lateral size in x and y (mm)
axial_size = 500  # axial size in z (mm)

# Grid resolution (choose resolution to capture the system resolution; here, scaled resolution)
dx = 7    # corresponds to 7 um * scaling factor (e.g. if scale=1000, then 7 um -> 7 mm)
dz = 10   # corresponds to 10 um * scaling factor
x = np.arange(0, lat_size+dx, dx)
y = np.arange(0, lat_size+dx, dx)
z = np.arange(0, axial_size+dz, dz)

Nx, Ny, Nz = len(x), len(y), len(z)
print("Grid dimensions: ", Nx, Ny, Nz)

# Create 3D coordinate arrays
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape: (Nx,Ny,Nz)

# ----- Create refractive index map -----
# Outside the sample, assume air (n=1.0). Inside the sample, assign n=1.52.
# Here, we assume the entire defined volume is our sample.
n_air = 1.0
n_sample = 1.52

n_map = n_air * np.ones((Nx, Ny, Nz))
# In our case, the sample fills the entire grid:
n_map[:, :, :] = n_sample

# ----- Create a reflectivity map -----
# We will define three axial regions (stage, base, cone region)
# For simplicity, assume:
#   Stage: 0 <= z < 0.1 * axial_size (bottom 10%)
#   Base: 0.1 * axial_size <= z < 0.4 * axial_size (next 30%)
#   Cone region: 0.4 * axial_size <= z <= axial_size (remaining 60%)

z_stage_max = 0.1 * axial_size   # stage thickness
z_base_max  = 0.4 * axial_size   # stage + base

# Create an empty reflectivity map; here we simply set a background value.
# For example, let the base reflectivity value be 0.8 and the cones have a lower reflectivity.
R_map = np.zeros((Nx, Ny, Nz))

# Assign reflectivity values for stage and base: highest reflectivity (e.g. 1.0)
for k, z_val in enumerate(z):
    if z_val < z_stage_max:
        R_map[:, :, k] = 1.0  # Stage region: highest reflectivity
    elif z_val < z_base_max:
        R_map[:, :, k] = 1.0  # Base region: also high reflectivity
    else:
        R_map[:, :, k] = 0.0  # Initialize cone region to zero; will add cones below

# ----- Define Four Cones on the Base -----
# For simplicity, assume the cones are located on the top surface of the base.
# Let the top of the base be at z = z_base_max.
# Place cones at four positions (e.g. near the center of each quadrant of the lateral area).
cone_centers = [
    (lat_size/4, lat_size/4),
    (3*lat_size/4, lat_size/4),
    (lat_size/4, 3*lat_size/4),
    (3*lat_size/4, 3*lat_size/4)
]
# Cone parameters (in mm, in the scaled model)
cone_height = axial_size - z_base_max  # cones extend from z_base_max to top (axial_size)
cone_max_reflectivity = 0.2  # reflectivity at the cone tip (lower than base)
# We use a simple linear gradient for reflectivity in the cone: from high at the base to low at the tip.
# Also, assume a lateral decay: points closer to the cone center have higher reflectivity.

# Loop over cone region voxels and add the cone reflectivity:
for (x0, y0) in cone_centers:
    # For each voxel in the cone region:
    for k, z_val in enumerate(z):
        if z_val >= z_base_max:
            # Relative axial position in the cone (0 at base, 1 at tip)
            rel_z = (z_val - z_base_max) / cone_height
            # Lateral distance from the cone center:
            rad = np.sqrt((X[:, :, k] - x0)**2 + (Y[:, :, k] - y0)**2)
            # Define a lateral "radius" for the cone effect (e.g., cone base radius)
            cone_radius = lat_size / 8  # arbitrary; adjust as needed
            # Compute a lateral decay: use an exponential or linear function
            lateral_decay = np.maximum(0, 1 - rad/cone_radius)
            # Compute cone reflectivity as a product of lateral and axial decay
            # For example, higher reflectivity at base (rel_z = 0) and lower at tip (rel_z = 1)
            cone_reflectivity = cone_max_reflectivity * lateral_decay * (1 - rel_z)
            # Add the cone contribution (you could also choose to replace the base reflectivity in the cone region)
            R_map[:, :, k] += cone_reflectivity

# Optionally, clip the reflectivity to a maximum value (e.g. 1.0)
R_map = np.clip(R_map, 0, 1.0)

# ----- Visualization -----
# Let’s look at a cross-sectional slice (for example, at the center of x and y) along the z direction.
center_x = Nx // 2
center_y = Ny // 2
plt.figure(figsize=(8, 4))
plt.plot(z, R_map[center_x, center_y, :], '-o')
plt.xlabel('Axial position (mm)')
plt.ylabel('Reflectivity')
plt.title('Reflectivity vs Axial Position at Center (x,y)')
plt.grid(True)
plt.show()

# ----- Notes on Scaling -----
# In your final simulation, remember that while you are working with the scaled model
# (310 mm lateral, 500 mm axial), the optical calculations (e.g., phase, resolution)
# must be converted back to the original dimensions (3.6 mm × 3.6 mm × 0.5 mm)
# by applying the appropriate scaling factors.
#
# For example, if the lateral scaling factor is S_lat and the axial scaling factor is S_ax,
# then a distance d (in the scaled model) corresponds to d/S in the original units.
#
# This code sets up a basic 3D grid with:
#   - A refractive index map 'n_map' (all sample region set to 1.52)
#   - A reflectivity map 'R_map' with high reflectivity in the stage (bottom 10%) and base (next 30%)
#     and with four cones added in the cone region (top 60%) that have lower reflectivity near their tips.
#
# You can refine these functions (e.g., the cone profiles) based on further CAD or optical design data.
# For example, you could add more complex cone shapes, different reflectivity profiles, or other optical effects.
