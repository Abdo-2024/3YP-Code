import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
lambda0 = 800e-9        # central wavelength in meters (800 nm)
n = 1.48                # refractive index
DOF_target = 500        # desired DOF in micrometers (target, used in cost)
# For plotting, we want to show DOF values beyond 500 up to 1000 or so

# Define NA range (avoid values too close to zero)
NA_min = 0.01
NA_max = 0.2
num_NA = 500            # number of NA points
NA_values = np.linspace(NA_min, NA_max, num_NA)

# Define k range (weighting factors) using a logarithmic scale
num_k = 1000
k_values = np.logspace(-6, -1, num_k)

# --- Calculate lateral resolution and DOF for each NA ---
# Lateral resolution: delta_x = 0.61 * lambda0 / NA   (in meters)
# DOF: DOF = n * lambda0 / NA^2   (in meters)
lateral_resolution = 0.61 * lambda0 / NA_values  # in meters
DOF = n * lambda0 / (NA_values ** 2)               # in meters

# Convert to micrometers (1 m = 1e6 µm)
lateral_resolution_um = lateral_resolution * 1e6
DOF_um = DOF * 1e6

# --- Create a meshgrid for k and NA ---
K, NA_grid = np.meshgrid(k_values, NA_values, indexing='ij')

# --- Compute the cost function ---
# cost = lateral resolution (µm) + k * (DOF_target - DOF (µm))^2
cost = lateral_resolution_um[np.newaxis, :] + K * (DOF_target - DOF_um[np.newaxis, :])**2

# --- For each k, find the NA that minimizes the cost ---
optimal_NA_vs_k = np.zeros(num_k)
optimal_lat_res_vs_k = np.zeros(num_k)
optimal_DOF_vs_k = np.zeros(num_k)
optimal_cost_vs_k = np.zeros(num_k)

for i in range(num_k):
    idx = np.argmin(cost[i, :])
    optimal_NA_vs_k[i] = NA_values[idx]
    optimal_lat_res_vs_k[i] = lateral_resolution_um[idx]
    optimal_DOF_vs_k[i] = DOF_um[idx]
    optimal_cost_vs_k[i] = cost[i, idx]

# --- Plotting: Lateral Resolution, DOF, and Optimal NA vs. k ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary y-axis: Lateral Resolution and DOF (both in µm)
ax1.semilogx(k_values, optimal_lat_res_vs_k, 'b--', label='Lateral Resolution (µm)')
ax1.semilogx(k_values, optimal_DOF_vs_k, 'g--', label='DOF (µm)')
ax1.set_xlabel('Weighting Factor k (log scale)')
ax1.set_ylabel('Resolution / DOF (µm)')
ax1.set_title('Optimal Values vs. k')
ax1.grid(True)
ax1.set_ylim(0, 1200)  # extend DOF axis so DOF values up to 1000 µm are visible

# Secondary y-axis: Optimal NA
ax2 = ax1.twinx()
ax2.semilogx(k_values, optimal_NA_vs_k, 'r-', linewidth=2, label='Optimal NA')
ax2.set_ylabel('Optimal NA')
ax2.set_ylim(NA_min, NA_max)

# Combine legends from both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# --- (Optional) Print summary of optimal values ---
print("k\tOptimal NA\tLateral Res (µm)\tDOF (µm)\tCost")
for k_val, na_opt, lat_res, dof_val, c_val in zip(k_values, optimal_NA_vs_k, optimal_lat_res_vs_k, optimal_DOF_vs_k, optimal_cost_vs_k):
    print(f"{k_val:.1e}\t{na_opt:.3f}\t\t{lat_res:.2f}\t\t{dof_val:.2f}\t{c_val:.2f}")
