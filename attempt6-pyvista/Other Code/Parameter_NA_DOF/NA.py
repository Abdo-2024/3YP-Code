import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda0 = 800e-9      # central wavelength in meters (800 nm)
n = 1.48              # refractive index
DOF_target = 500      # desired DOF in micrometers
k = 0.0144            # weight factor for cost function

# Define NA range (avoid too close to zero to have meaningful values)
NA_min = 0.01
NA_max = 0.6
num_points = 10000

# Generate NA values
NA_values = np.linspace(NA_min, NA_max, num_points)

# Calculate lateral resolution (delta_x) and DOF using the given formulas
# delta_x = 0.61 * lambda0 / NA      (in meters)
# DOF = n * lambda0 / NA^2            (in meters)
lateral_resolution = 0.61 * lambda0 / NA_values  # in meters
DOF = n * lambda0 / (NA_values**2)               # in meters

# Convert to micrometers
lateral_resolution_um = lateral_resolution * 1e6
DOF_um = DOF * 1e6

# Define the cost function:
# We want lateral resolution to be as small as possible and DOF to be close to DOF_target.
# Here, cost = lateral_resolution (µm) + k * (DOF_target - DOF (µm))^2
cost = lateral_resolution_um + k * (DOF_target - DOF_um)**2

# Find the NA value that minimizes the cost
optimal_index = np.argmin(cost)
optimal_NA = NA_values[optimal_index]
optimal_lat_res = lateral_resolution_um[optimal_index]
optimal_DOF = DOF_um[optimal_index]
optimal_cost = cost[optimal_index]

print(f"Optimal NA: {optimal_NA:.4f}")
print(f"Lateral Resolution at optimal NA: {optimal_lat_res:.2f} µm")
print(f"DOF at optimal NA: {optimal_DOF:.2f} µm")
print(f"Cost function value at optimal NA: {optimal_cost:.2f}")

# Plotting in subplots (three panels side by side)
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# Subplot 1: Lateral Resolution vs. NA
axs[0].plot(NA_values, lateral_resolution_um, color='tab:blue')
axs[0].set_title('Lateral Resolution vs. NA')
axs[0].set_xlabel('NA')
axs[0].set_ylabel('Lateral Resolution (µm)')
axs[0].grid(True)
axs[0].axvline(optimal_NA, color='r', linestyle='--', label=f'Optimal NA = {optimal_NA:.3f}')
axs[0].legend()

# Subplot 2: DOF vs. NA
axs[1].plot(NA_values, DOF_um, color='tab:red')
axs[1].set_title('DOF vs. NA')
axs[1].set_xlabel('NA')
axs[1].set_ylabel('DOF (µm)')
axs[1].grid(True)
axs[1].axvline(optimal_NA, color='r', linestyle='--', label=f'Optimal NA = {optimal_NA:.3f}')
axs[1].legend()

# Subplot 3: Cost Function vs. NA
axs[2].plot(NA_values, cost, color='tab:green')
axs[2].set_title('Cost Function vs. NA')
axs[2].set_xlabel('NA')
axs[2].set_ylabel('Cost (a.u.)')
axs[2].grid(True)
axs[2].axvline(optimal_NA, color='r', linestyle='--', label=f'Optimal NA = {optimal_NA:.3f}')
axs[2].legend()

plt.suptitle('Trade-off Optimization between Lateral Resolution and DOF', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
