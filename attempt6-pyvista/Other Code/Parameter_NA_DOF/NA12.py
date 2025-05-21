import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib.offsetbox import AnchoredText

# --- Set font and plot defaults ---
plt.rcParams["font.family"] = "Nimbus Sans"
plt.rcParams.update({'font.size': 10, 'lines.linewidth': 2})

# --- Parameters ---
# Fixed central wavelength and bandwidth (for context)
L = 800e-9          # 800 nm in meters
bandwidth = 40e-9   # 40 nm bandwidth (not used in calculations)

# Refractive indices
n_air = 1.0         # Object NA is defined in air
n_sample = 1.52     # Sample refractive index

# Define object NA range (in air) and compute effective NA in sample.
# Expanded range from 0.01 to 0.08 to include both target configurations.
num_NA = 1000
NA_object_values = np.linspace(0.01, 0.08, num_NA)
NA_sample_values = NA_object_values * (n_sample / n_air)  # effective NA in sample

# --- Compute Lateral Resolution and Depth of Focus (DOF) ---
# Lateral resolution: δx = (2 ln2 * L) / (π * NA_sample) [in meters]
lat_res = (2 * np.log(2) * L) / (np.pi * NA_sample_values)
lat_res_um = lat_res * 1e6  # convert to µm

# Depth of Focus: DOF = (n_sample * L) / (2π * NA_sample²) [in meters]
DOF = (n_sample * L) / (2 * np.pi * NA_sample_values**2)
DOF_um = DOF * 1e6  # convert to µm

# --- Compute target configurations ---
# For a given DOF target, the effective NA is:
#   NA_sample_target = sqrt((n_sample * L) / (2π * DOF_target))
def compute_configuration(DOF_target):
    NA_sample_target = np.sqrt((n_sample * L) / (2 * np.pi * (DOF_target * 1e-6)))
    NA_object_target = NA_sample_target / n_sample
    lat_res_target = (2 * np.log(2) * L) / (np.pi * NA_sample_target) * 1e6
    return NA_object_target, NA_sample_target, lat_res_target

# Target configurations
DOF_target1 = 100  # µm (red marker)
DOF_target2 = 500  # µm (gray marker)

objNA_target1, NA_sample_target1, lat_res_target1 = compute_configuration(DOF_target1)
objNA_target2, NA_sample_target2, lat_res_target2 = compute_configuration(DOF_target2)

# --- Plotting ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the relation curve over the full effective NA range in the fixed color.
ax.plot(NA_sample_values, lat_res_um, DOF_um, color="#414f73ff", linewidth=2)

# Set axis labels and title with specified font sizes.
ax.set_xlabel('Effective NA in Sample', fontsize=10, labelpad=10)
ax.set_ylabel('Lateral Resolution (µm)', fontsize=10, labelpad=10)
ax.set_zlabel('DOF (µm)', fontsize=10, labelpad=10)
ax.set_title('Effective NA vs. Lateral Resolution vs. DOF at 800 nm',
             fontsize=11, fontweight='bold', pad=20)

# Set tick label font size to 9.
ax.tick_params(axis='both', which='major', labelsize=9)

# Optionally set axis limits so the entire curve and markers are clearly visible.
ax.set_xlim(NA_sample_values[0], NA_sample_values[-1])
ax.set_ylim(np.min(lat_res_um)*0.95, np.max(lat_res_um)*1.05)
ax.set_zlim(np.min(DOF_um)*0.95, np.max(DOF_um)*1.05)

# Make the background grid thinner.
ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax.zaxis._axinfo["grid"]['linewidth'] = 0.5

# Plot markers for both target configurations.
# Marker for DOF = 100 µm (red)
ax.scatter(NA_sample_target1, lat_res_target1, DOF_target1,
           color='red', s=150, depthshade=True, label='DOF = 100 µm')
# Marker for DOF = 500 µm (gray)
ax.scatter(NA_sample_target2, lat_res_target2, DOF_target2,
           color='gray', s=150, depthshade=True, label='DOF = 500 µm')

# Create an anchored text box showing both configurations.
config_text = (f"Configurations:\n"
               f"DOF = 100 µm:\n"
               f"  Object NA = {objNA_target1:.3f}\n"
               f"  Lateral Res = {lat_res_target1:.1f} µm\n\n"
               f"DOF = 500 µm:\n"
               f"  Object NA = {objNA_target2:.3f}\n"
               f"  Lateral Res = {lat_res_target2:.1f} µm")
anchored_text = AnchoredText(config_text, loc='upper left', prop=dict(size=10), frameon=True)
ax.add_artist(anchored_text)

# Add legend for the markers.
ax.legend(fontsize=10)

# Save the plot as an SVG file.
plt.savefig("plot.svg", format="svg")
plt.show()
