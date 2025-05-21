import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Define Parameters
# -----------------------
# Source parameters
lambda_center = 787.3e-9       # center wavelength in meters (787.3 nm)
bandwidth = 40e-9              # bandwidth in meters (40 nm)
num_wavelengths = 1000         # number of wavelengths to sample
wavelengths = np.linspace(lambda_center - bandwidth/2,
                          lambda_center + bandwidth/2,
                          num_wavelengths)  # in meters

# Grating parameters
groove_density = 1200          # lines per mm
d = 1e-3 / groove_density      # grating period in meters (convert lines/mm to m)
m = 1                        # diffraction order

# Incident angle in radians (e.g., 30°)
theta_i_deg = 30.0
theta_i = np.deg2rad(theta_i_deg)

# Imaging lens focal length (in meters)
f_imaging = 100e-3  # 100 mm

# Detector properties
detector_width = 14.34e-3  # detector active width in meters (14.34 mm)

# -----------------------
# Compute Diffraction Angle for each wavelength
# -----------------------
# Grating equation: d*(sin(theta_i) + sin(theta_d)) = m*lambda
# Solve for sin(theta_d):
sin_theta_d = m * wavelengths / d - np.sin(theta_i)

# Clip sin_theta_d to the valid range [-1, 1] to avoid errors with arcsin
sin_theta_d = np.clip(sin_theta_d, -1, 1)
theta_d = np.arcsin(sin_theta_d)  # diffraction angle in radians

# -----------------------
# Compute Linear Position on Detector
# -----------------------
# Using an imaging lens, the linear position is: x = f * tan(theta_d)
x_positions = f_imaging * np.tan(theta_d)  # in meters

# -----------------------
# Plot Results
# -----------------------
plt.figure(figsize=(10, 6))
plt.plot(wavelengths*1e9, x_positions*1e3, 'b-', linewidth=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Position on Detector (mm)")
plt.title("Simulated Spectrum Dispersion on Detector")
plt.grid(True)

# Mark the span on the detector
span = np.max(x_positions) - np.min(x_positions)
plt.annotate(f"Span = {span*1e3:.2f} mm", xy=(lambda_center*1e9, np.mean(x_positions)*1e3),
             xytext=(lambda_center*1e9+10, np.mean(x_positions)*1e3+1),
             arrowprops=dict(facecolor='black', arrowstyle="->"))

plt.show()

# Check if the spectral spread fits on the detector
print(f"Spectral span on detector: {span*1e3:.2f} mm")
print(f"Detector active width: {detector_width*1e3:.2f} mm")
if span < detector_width:
    print("The dispersed spectrum fits on the detector.")
else:
    print("Warning: The dispersed spectrum exceeds the detector width!")
