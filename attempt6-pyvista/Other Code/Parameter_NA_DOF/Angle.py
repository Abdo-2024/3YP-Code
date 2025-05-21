import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib import cm, colors
from matplotlib.offsetbox import AnchoredText

# --- Parameters ---
# Wavelength (use the optimal wavelength from your system, e.g., 792.3 nm)
wavelength = 792.3e-9       # in meters
# Beam radius incident on the axicon
R = 5.25e-3                    # in meters (adjust as needed)
# Target Depth of Focus (DOF)
target_DOF = 600e-6         # in meters (550 µm)

# --- Define axicon angle range ---
# Here, angles are in radians. Adjust the range if needed.
angle_min = 0.01             # radians
angle_max = 1.5             # radians
num_angles = 10000
angles = np.linspace(angle_min, angle_max, num_angles)

# --- Compute Lateral Resolution and DOF for each angle ---
# Lateral resolution (approximate FWHM-like metric) in meters:
# r0 ≈ 2.405/(k sinθ) where k = 2π/λ; and 2.405/(2π) ≈ 0.3827
lat_res = (0.3827 * wavelength) / np.sin(angles)   # in meters
lat_res_um = lat_res * 1e6                           # in micrometers

# Depth of Focus (DOF) in meters:
DOF = R / np.tan(angles)                             # in meters
DOF_um = DOF * 1e6                                   # in micrometers

# --- Find the optimal angle that minimizes lateral resolution
# while meeting the DOF constraint (DOF >= target_DOF)
mask = DOF_um >= target_DOF*1e6 if False else (DOF_um >= target_DOF*1)  
# The above line is a trick; here target_DOF is already in meters so:
mask = DOF_um >= (target_DOF * 1e6)  # This is not needed; instead use target_DOF in µm

# Actually, since we converted DOF to µm and target_DOF to µm:
target_DOF_um = target_DOF * 1e6  # convert target DOF to µm

mask = DOF_um >= target_DOF_um
if np.any(mask):
    # Set lateral resolution to infinity for points not meeting the DOF constraint.
    lat_res_valid = np.where(mask, lat_res_um, np.inf)
    # Find the index where lateral resolution is minimal among the valid angles.
    min_index = np.argmin(lat_res_valid)
    optimal_angle = angles[min_index]            # in radians
    optimal_lat_res = lat_res_um[min_index]        # in µm
    optimal_DOF = DOF_um[min_index]                # in µm
else:
    print("No angle satisfies the DOF constraint.")
    optimal_angle = None
    optimal_lat_res = None
    optimal_DOF = None

print("Optimal configuration:")
if optimal_angle is not None:
    print(f"  Axicon Angle: {np.degrees(optimal_angle):.2f}° ({optimal_angle:.4f} rad)")
    print(f"  Lateral Resolution: {optimal_lat_res:.2f} µm")
    print(f"  DOF: {optimal_DOF:.2f} µm (Target was {target_DOF_um:.2f} µm)")
else:
    print("No valid configuration found.")

# --- 3D Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# For visualization, we can plot the curve in 3D.
# x-axis: Axicon Angle (in degrees), y-axis: Lateral Resolution (µm), z-axis: DOF (µm)
angles_deg = np.degrees(angles)

# Create a scatter plot where color maps to the axicon angle
norm = colors.Normalize(vmin=angle_min, vmax=angle_max)
sc = ax.scatter(angles_deg, lat_res_um, DOF_um, c=angles, cmap='viridis', s=10)

ax.set_xlabel('Axicon Angle (°)')
ax.set_ylabel('Lateral Resolution (µm)')
ax.set_zlabel('DOF (µm)')
ax.set_title('Bessel Beam: Lateral Resolution & DOF vs. Axicon Angle')

# Highlight the optimal point in red if found.
if optimal_angle is not None:
    ax.scatter(np.degrees(optimal_angle), optimal_lat_res, optimal_DOF, 
               color='r', s=100, label='Optimal Configuration')
    opt_text = (f"Optimal Configuration:\n"
                f"Angle = {np.degrees(optimal_angle):.2f}°\n"
                f"Lateral Res = {optimal_lat_res:.2f} µm\n"
                f"DOF = {optimal_DOF:.2f} µm")
    anchored_text = AnchoredText(opt_text, loc='upper left', prop=dict(size=10), frameon=True)
    ax.add_artist(anchored_text)
    ax.legend()

# Add a colorbar showing the mapping from angle (radians) to color.
mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
mappable.set_array(angles)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Axicon Angle (rad)')

plt.show()
