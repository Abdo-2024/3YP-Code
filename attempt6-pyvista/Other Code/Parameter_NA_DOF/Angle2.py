import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (from your laser data) ---
wavelength = 800e-9    # 800 nm, in meters
R = 5e-3               # 5 mm, in meters

# Target DOF in micrometers:
target_DOF_um = 550  # e.g., 550 µm
target_DOF_m = target_DOF_um * 1e-6

# --- Axicon angle range (in radians) ---
angle_min = 0.01   # ~0.57°
angle_max = 1.0    # ~57.3°
num_points = 1000
theta = np.linspace(angle_min, angle_max, num_points)

# --- Bessel beam formulas ---
# Lateral resolution (approx. FWHM of central lobe), in meters
lat_res = 0.3827 * wavelength / np.sin(theta)

# Depth of focus (non-diffracting range), in meters
dof = R / np.tan(theta)

# Convert to micrometers for plotting
lat_res_um = lat_res * 1e6
dof_um = dof * 1e6

# --- Find the axicon angle that meets the DOF constraint and minimizes lateral res ---
mask = dof >= target_DOF_m  # those angles that give DOF >= 550 µm
if np.any(mask):
    lat_res_valid = np.where(mask, lat_res_um, np.inf)
    min_index = np.argmin(lat_res_valid)
    optimal_angle = theta[min_index]
    optimal_lat_res = lat_res_um[min_index]
    optimal_dof = dof_um[min_index]
    print("Optimal Axicon Angle (radians):", optimal_angle)
    print("Optimal Axicon Angle (degrees):", np.degrees(optimal_angle))
    print("Resulting Lateral Res (µm):", optimal_lat_res)
    print("Resulting DOF (µm):", optimal_dof)
else:
    print("No angle produces the required DOF.")

# --- Quick Plots ---
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(np.degrees(theta), lat_res_um, label='Lat. Res (µm)')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Axicon Angle (deg)')
plt.ylabel('Lateral Resolution (µm)')
plt.title('Lateral Resolution vs. Angle')

plt.subplot(1,2,2)
plt.plot(np.degrees(theta), dof_um, label='DOF (µm)')
plt.axhline(y=target_DOF_um, color='r', linestyle='--', label='Target DOF')
plt.xlabel('Axicon Angle (deg)')
plt.ylabel('Depth of Focus (µm)')
plt.title('DOF vs. Angle')
plt.legend()

plt.tight_layout()
plt.show()
