import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# --- Parameters ---
lambda0 = 800e-9        # central wavelength in meters (800 nm)
n = 1.48                # refractive index
DOF_target = 500        # desired DOF in micrometers

# Define NA range and k range
NA_min = 0.01           # avoid zero to prevent division by zero
NA_max = 0.3
num_NA = 500            # number of NA points
NA_values = np.linspace(NA_min, NA_max, num_NA)

num_k = 1000
# Use logarithmically spaced k values (adjust range as needed)
k_values = np.logspace(-6, -3, num_k)

# --- Calculate lateral resolution and DOF for each NA ---
# Lateral resolution (delta_x) in meters: 0.61 * lambda0 / NA
# DOF in meters: n * lambda0 / NA^2
lateral_resolution = 0.61 * lambda0 / NA_values  # in meters
DOF = n * lambda0 / (NA_values**2)                # in meters

# Convert both to micrometers (1 m = 1e6 µm)
lateral_resolution_um = lateral_resolution * 1e6
DOF_um = DOF * 1e6

# --- Create a meshgrid for k and NA ---
# k will be along the vertical axis, NA along horizontal.
K, NA_grid = np.meshgrid(k_values, NA_values, indexing='ij')

# --- Compute the cost function over the grid ---
# cost = lateral_resolution (µm) + k * (DOF_target - DOF (µm))^2
# Note: lateral_resolution_um and DOF_um are 1D arrays (for NA) and are broadcast.
cost = lateral_resolution_um[np.newaxis, :] + K * (DOF_target - DOF_um[np.newaxis, :])**2

# --- Find the optimal (minimum cost) point over the grid ---
opt_index = np.argmin(cost)
opt_k_idx, opt_NA_idx = np.unravel_index(opt_index, cost.shape)
optimal_k = k_values[opt_k_idx]
optimal_NA = NA_values[opt_NA_idx]
optimal_cost = cost[opt_k_idx, opt_NA_idx]
optimal_lat_res = lateral_resolution_um[opt_NA_idx]
optimal_DOF = DOF_um[opt_NA_idx]

print(f"Optimal k: {optimal_k:.2e}")
print(f"Optimal NA: {optimal_NA:.4f}")
print(f"Optimal cost: {optimal_cost:.4f}")
print(f"Lateral Resolution at optimum: {optimal_lat_res:.2f} µm")
print(f"DOF at optimum: {optimal_DOF:.2f} µm")

# --- 3D Plot: Surface of the cost function ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface; NA on x-axis, k on y-axis, cost on z-axis
surf = ax.plot_surface(NA_grid, K, cost, cmap=cm.viridis, edgecolor='none', alpha=0.8)
ax.set_xlabel('Numerical Aperture (NA)')
ax.set_ylabel('Weighting Factor k')
ax.set_zlabel('Cost (a.u.)')
ax.set_title('3D Surface of Cost vs. NA and k')

# Mark the optimal point
ax.scatter(optimal_NA, optimal_k, optimal_cost, color='r', s=100, label='Optimal Point')
ax.legend()

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# --- (Optional) 3D scatter plot of optimal NA vs. k ---
# For each k, find the NA that minimizes the cost.
optimal_NA_vs_k = np.zeros(num_k)
optimal_cost_vs_k = np.zeros(num_k)
optimal_lat_res_vs_k = np.zeros(num_k)
optimal_DOF_vs_k = np.zeros(num_k)
for i in range(num_k):
    idx = np.argmin(cost[i, :])
    optimal_NA_vs_k[i] = NA_values[idx]
    optimal_cost_vs_k[i] = cost[i, idx]
    optimal_lat_res_vs_k[i] = lateral_resolution_um[idx]
    optimal_DOF_vs_k[i] = DOF_um[idx]

plt.figure(figsize=(10,6))
plt.semilogx(k_values, optimal_NA_vs_k, 'r-', label='Optimal NA vs k')
plt.xlabel('Weighting Factor k (log scale)')
plt.ylabel('Optimal NA')
plt.title('Optimal NA as a Function of k')
plt.grid(True)
plt.legend()
plt.show()

# You can also plot how the corresponding lateral resolution and DOF vary with k:
plt.figure(figsize=(10,6))
plt.semilogx(k_values, optimal_lat_res_vs_k, 'b--', label='Lateral Resolution (µm)')
plt.semilogx(k_values, optimal_DOF_vs_k, 'g--', label='DOF (µm)')
plt.xlabel('Weighting Factor k (log scale)')
plt.ylabel('Value (µm)')
plt.title('Optimal Lateral Resolution and DOF vs. k')
plt.grid(True)
plt.legend()
plt.show()
