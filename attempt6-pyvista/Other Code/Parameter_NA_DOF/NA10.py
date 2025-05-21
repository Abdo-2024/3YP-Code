import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from matplotlib import cm, colors
from matplotlib.offsetbox import AnchoredText

# --- Parameters ---
# Wavelength range: from 780 nm to 820 nm (in meters)
lambda0_min = 780e-9
lambda0_max = 820e-9
num_lambda = 40         # number of wavelength points
lambda0_values = np.linspace(lambda0_min, lambda0_max, num_lambda)

# Define refractive indices:
n_air    = 1.0       # Environment in which the lens operates (air)
# For the new objective lens, the object-side NA is specified as 0.06 (in air)
NA_object_fixed = 0.04  
# We assume the sample has a refractive index:
n_sample = 1.52

# --- Effective NA Calculation ---
# When light goes from air (where NA_object is defined) into the sample,
# its effective NA increases:
#     NA_sample = NA_object * (n_sample / n_air)
# We simulate a narrow range around the fixed object NA (0.05 to 0.07)
num_NA = 1000
NA_values = np.linspace(0.03, 0.07, num_NA)  # object-side NA values (in air)
# Compute effective NA in the sample:
NA_sample_values = NA_values * (n_sample / n_air)  # Multiply by 1.52

# --- Target Depth of Focus (DOF) ---
target_DOF = 50  # desired DOF in micrometers

# --- Create a meshgrid for effective NA (in sample) and wavelength λ₀ ---
NA_grid, L_grid = np.meshgrid(NA_sample_values, lambda0_values, indexing='xy')

# --- Compute Lateral Resolution and DOF for each (λ₀, NA_sample) combination ---
# Lateral resolution (δx, in meters) is given by:
#    δx = (2 ln2 * λ₀) / (π * NA_sample)
lat_res = (2 * np.log(2) * L_grid) / (np.pi * NA_grid)  # in meters

# Depth of focus (DOF, in meters) is given by:
#    DOF = (n_sample * λ₀) / (2π * NA_sample²)
DOF = (n_sample * L_grid) / (2 * np.pi * NA_grid**2)      # in meters

# Convert both to micrometers (1 m = 1e6 µm)
lat_res_um = lat_res * 1e6
DOF_um = DOF * 1e6

# --- Find the optimal (λ₀, NA) combination ---
# We want DOF >= target and minimal lateral resolution.
mask = DOF_um >= target_DOF
if np.any(mask):
    lat_res_valid = np.where(mask, lat_res_um, np.inf)
    min_index = np.argmin(lat_res_valid)
    idx_lambda, idx_NA = np.unravel_index(min_index, lat_res_valid.shape)
    optimal_lambda = lambda0_values[idx_lambda]      # in meters
    optimal_lambda_nm = optimal_lambda * 1e9           # in nm
    # The optimal effective NA in the sample is from our grid:
    optimal_NA_sample = NA_grid[idx_lambda, idx_NA]
    # Convert back to object NA (as defined in air):
    optimal_NA_object = optimal_NA_sample / n_sample
    optimal_lat_res = lat_res_um[idx_lambda, idx_NA]
    optimal_DOF = DOF_um[idx_lambda, idx_NA]
else:
    print("No valid (λ₀, NA) combination satisfies the DOF constraint.")
    optimal_lambda_nm = None
    optimal_NA_object = None
    optimal_lat_res = None
    optimal_DOF = None

print("Optimal configuration:")
print(f"  Wavelength: {optimal_lambda_nm:.1f} nm")
print(f"  Object NA: {optimal_NA_object:.4f}")
print(f"  Lateral Resolution: {optimal_lat_res:.2f} µm")
print(f"  DOF: {optimal_DOF:.2f} µm (Target was {target_DOF} µm)")

# --- Plot settings for a bolder appearance ---
plt.rcParams.update({'font.size': 10, 'lines.linewidth': 2})
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Map the wavelength (L_grid, in meters) to a color; convert to nm.
wavelength_nm = L_grid * 1e9
norm = colors.Normalize(vmin=lambda0_min*1e9, vmax=lambda0_max*1e9)
facecolors = cm.viridis(norm(wavelength_nm))

# Plot the surface with a dense mesh and smooth shading.
surf = ax.plot_surface(NA_grid, lat_res_um, DOF_um,
#                       rstride=1, cstride=1,
                       facecolors=facecolors,
                       edgecolor='none', alpha=1, shade=True)

ax.set_xlabel('Effective NA in Sample', fontsize=10, labelpad=10)
ax.set_ylabel('Lateral Resolution (µm)', fontsize=10, labelpad=10)
ax.set_zlabel('DOF (µm)', fontsize=10, labelpad=10)
ax.set_title('Surface: Effective NA vs. Lateral Resolution vs. DOF\n(Varying λ₀ from 780 nm to 820 nm)', fontsize=10, pad=20)

if optimal_lambda_nm is not None:
    ax.scatter(optimal_NA_sample, optimal_lat_res, optimal_DOF, color='r', s=150,
               depthshade=True, label='Optimal Configuration')
    opt_text = (f"Optimal Configuration:\n"
                f"λ = {optimal_lambda_nm:.1f} nm\n"
                f"Object NA = {optimal_NA_object:.3f}\n"
                f"Lateral Res = {optimal_lat_res:.1f} µm\n"
                f"DOF = {optimal_DOF:.1f} µm")
    anchored_text = AnchoredText(opt_text, loc='upper left', prop=dict(size=10), frameon=True)
    ax.add_artist(anchored_text)
    ax.legend(fontsize=10)

mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(wavelength_nm)
cb = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Wavelength (nm)')
cb.ax.tick_params(labelsize=10)

plt.show()
