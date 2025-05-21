import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda0 = 800e-9      # central wavelength in meters (800 nm)
n = 1.48              # refractive index
DOF_target = 500      # desired DOF in micrometers

# Define NA range (avoid too close to zero to get meaningful values)
NA_min = 0.01
NA_max = 0.6
num_points = 10000
NA_values = np.linspace(NA_min, NA_max, num_points)

# Calculate lateral resolution (delta_x) and DOF using the given formulas
lateral_resolution = 0.61 * lambda0 / NA_values  # in meters
DOF = n * lambda0 / (NA_values**2)                 # in meters

# Convert to micrometers
lateral_resolution_um = lateral_resolution * 1e6
DOF_um = DOF * 1e6

# List of k values to examine
k_values = [0.001, 0.005, 0.01, 0.02, 0.05]

plt.figure(figsize=(10, 6))

# Loop over each k and compute the cost function
for k in k_values:
    cost = lateral_resolution_um + k * (DOF_target - DOF_um)**2
    plt.plot(NA_values, cost, label=f'k = {k}')

plt.xlabel('Numerical Aperture (NA)')
plt.ylabel('Cost (a.u.)')
plt.title('Cost Function vs. NA for Various k Values')
plt.legend()
plt.grid(True)
plt.show()
