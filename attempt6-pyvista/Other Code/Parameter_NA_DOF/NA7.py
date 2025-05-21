import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from matplotlib import cm, colors
from matplotlib.offsetbox import AnchoredText

# --- Parameters ---
# Wavelength range: from 780 nm to 820 nm
lambda0_min = 780e-9      # in meters
lambda0_max = 820e-9      # in meters
num_lambda = 1000         # number of wavelength points
lambda0_values = np.linspace(lambda0_min, lambda0_max, num_lambda)

# Refractive index (assumed constant)
n = 1.52

# Define NA range
NA_min = 0.01
NA_max = 0.5
num_NA = 1000             # number of NA points
NA_values = np.linspace(NA_min, NA_max, num_NA)

# --- Target DOF ---
target_DOF = 50 # desired Depth of Focus in micrometers

# --- Create a meshgrid for NA and λ₀ ---
# 'NA_grid' will vary along the x-axis and 'L_grid' represents λ₀.
NA_grid, L_grid = np.meshgrid(NA_values, lambda0_values, indexing='xy')

# --- Compute Lateral Resolution and DOF for each (λ₀, NA) combination ---
# Using the new equations:
# Lateral resolution (δx) in meters: δx = (2 ln2 * λ₀) / (π * NA)
lat_res = (2 * np.log(2) * L_grid) / (np.pi * NA_grid)  # in meters

# Depth of focus (DOF) in meters: b = (n * λ₀) / (2π * NA²)
DOF = (n * L_grid) / (2 * np.pi * NA_grid**2)           # in meters

# Convert both to micrometers (1 m = 1e6 µm)
lat_res_um = lat_res * 1e6
DOF_um = DOF * 1e6

# --- Find the optimal (λ₀, NA) that meets the target DOF (DOF >= target_DOF)
# and minimizes lateral resolution (i.e. the smallest lateral resolution among valid points)
mask = DOF_um >= target_DOF
if np.any(mask):
    # For points not meeting the DOF constraint, set lateral resolution to infinity
    lat_res_valid = np.where(mask, lat_res_um, np.inf)
    # Find the index where lateral resolution is minimal among the valid points
    min_index = np.argmin(lat_res_valid)
    idx_lambda, idx_NA = np.unravel_index(min_index, lat_res_valid.shape)
    optimal_lambda = lambda0_values[idx_lambda]  # in meters
    optimal_lambda_nm = optimal_lambda * 1e9       # in nm
    optimal_NA = NA_values[idx_NA]
    optimal_lat_res = lat_res_um[idx_lambda, idx_NA]
    optimal_DOF = DOF_um[idx_lambda, idx_NA]
else:
    print("No (λ₀, NA) combination satisfies the DOF constraint.")
    optimal_lambda_nm = None
    optimal_NA = None
    optimal_lat_res = None
    optimal_DOF = None

print("Optimal configuration:")
print(f"  Wavelength: {optimal_lambda_nm:.1f} nm")
print(f"  NA: {optimal_NA:.4f}")
print(f"  Lateral Resolution: {optimal_lat_res:.2f} µm")
print(f"  DOF: {optimal_DOF:.2f} µm (Target was {target_DOF} µm)")

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Map the wavelength (L_grid, in meters) to a color; convert to nm.
wavelength_nm = L_grid * 1e9
# Normalize wavelength to the range [780,820] nm
norm = colors.Normalize(vmin=lambda0_min*1e9, vmax=lambda0_max*1e9)
# Compute facecolors using the viridis colormap.
facecolors = cm.viridis(norm(wavelength_nm))

# Plot the surface with facecolors representing wavelength; disable shading.
surf = ax.plot_surface(NA_grid, lat_res_um, DOF_um, facecolors=facecolors,
                       edgecolor='none', alpha=1, shade=False)

ax.set_xlabel('Numerical Aperture (NA)')
ax.set_ylabel('Lateral Resolution (µm)')
ax.set_zlabel('DOF (µm)')
ax.set_title('Surface: NA vs. Lateral Resolution vs. DOF\n(Varying λ₀ from 780 nm to 820 nm)')

# Overlay the optimal point as a red marker if it exists.
if optimal_lambda_nm is not None:
    ax.scatter(optimal_NA, optimal_lat_res, optimal_DOF, color='r', s=100, label='Optimal Configuration')
    # Instead of annotating on the point, add an anchored text box (like a legend) in the upper left.
    opt_text = (f"Optimal Configuration:\n"
                f"λ = {optimal_lambda_nm:.1f} nm\n"
                f"NA = {optimal_NA:.3f}\n"
                f"Lateral Res = {optimal_lat_res:.1f} µm\n"
                f"DOF = {optimal_DOF:.1f} µm")
    anchored_text = AnchoredText(opt_text, loc='upper left', prop=dict(size=10), frameon=True)
    ax.add_artist(anchored_text)
    ax.legend()

# Create a colorbar mapping the colormap to wavelength (nm)
mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(wavelength_nm)  # Associate wavelength data for the colorbar
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Wavelength (nm)')

plt.show()
