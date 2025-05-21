import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
lambda0 = 800e-9        # central wavelength in meters (800 nm)
n = 1.48                # refractive index
DOF_target = 10000        # desired DOF in micrometers

# Define NA range (avoid values too close to zero)
NA_min = 0.01
NA_max = 0.6
num_NA = 500            # number of NA points
NA_values = np.linspace(NA_min, NA_max, num_NA)

# Define k range (weighting factors) using a logarithmic scale
num_k = 100
k_values = np.logspace(-6, 3, num_k)

# --- Calculate lateral resolution and DOF ---
# Lateral resolution: delta_x = 0.61 * lambda0 / NA  (in meters)
# DOF: DOF = n * lambda0 / NA^2   (in meters)
lateral_resolution = 0.61 * lambda0 / NA_values  # in meters
DOF = n * lambda0 / (NA_values ** 2)               # in meters

# Convert to micrometers for plotting
lateral_resolution_um = lateral_resolution * 1e6
DOF_um = DOF * 1e6

# --- Create a meshgrid for k and NA ---
# k on the vertical axis and NA on the horizontal axis
K, NA_grid = np.meshgrid(k_values, NA_values, indexing='ij')

# Compute the cost for each combination:
# cost = lateral_resolution (µm) + k * (DOF_target - DOF (µm))^2
# Note: lateral_resolution_um and DOF_um are 1D arrays over NA; they are broadcast to the grid.
cost = lateral_resolution_um[np.newaxis, :] + K * (DOF_target - DOF_um[np.newaxis, :])**2

# --- For each k (each row), find the NA value that minimizes the cost ---
optimal_NA = np.zeros(num_k)
optimal_lat_res = np.zeros(num_k)
optimal_DOF = np.zeros(num_k)
optimal_cost = np.zeros(num_k)

for i in range(num_k):
    idx_min = np.argmin(cost[i, :])
    optimal_NA[i] = NA_values[idx_min]
    optimal_lat_res[i] = lateral_resolution_um[idx_min]
    optimal_DOF[i] = DOF_um[idx_min]
    optimal_cost[i] = cost[i, idx_min]

# --- Plotting ---
fig, (ax_heat, ax_curve) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Heatmap of cost vs NA and k
heat = ax_heat.pcolormesh(NA_values, k_values, cost, shading='auto', cmap='viridis')
ax_heat.set_xlabel('Numerical Aperture (NA)')
ax_heat.set_ylabel('Weighting Factor k')
ax_heat.set_title('Heatmap of Cost Function')
cbar = plt.colorbar(heat, ax=ax_heat)
cbar.set_label('Cost (a.u.)')
# Overlay the optimal NA curve for each k
ax_heat.plot(optimal_NA, k_values, 'r-', linewidth=2, label='Optimal NA')
ax_heat.legend()

# Subplot 2: Optimal NA, Lateral Resolution, and DOF vs k
ax_curve.set_xscale('log')
ax_curve.plot(k_values, optimal_NA, 'r-', label='Optimal NA')
ax_curve.set_xlabel('Weighting Factor k (log scale)')
ax_curve.set_ylabel('Optimal NA', color='r')
ax_curve.tick_params(axis='y', labelcolor='r')
ax_curve.set_title('Optimal Values vs. k')

# Create a twin y-axis to plot lateral resolution and DOF
ax_curve2 = ax_curve.twinx()
ax_curve2.plot(k_values, optimal_lat_res, 'b--', label='Lateral Res (µm)')
ax_curve2.plot(k_values, optimal_DOF, 'g--', label='DOF (µm)')
ax_curve2.set_ylabel('Lateral Resolution & DOF (µm)')
ax_curve2.tick_params(axis='y')
# Combine legends from both axes
lines_1, labels_1 = ax_curve.get_legend_handles_labels()
lines_2, labels_2 = ax_curve2.get_legend_handles_labels()
ax_curve2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.suptitle('Trade-off Optimization: Varying k to Balance Lateral Resolution and DOF', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- (Optional) Print a summary of optimal values ---
print("k\tOptimal NA\tLateral Res (µm)\tDOF (µm)\tCost")
for k_val, na_opt, lat_res, dof_val, c_val in zip(k_values, optimal_NA, optimal_lat_res, optimal_DOF, optimal_cost):
    print(f"{k_val:.1e}\t{na_opt:.3f}\t\t{lat_res:.2f}\t\t{dof_val:.2f}\t{c_val:.2f}")
