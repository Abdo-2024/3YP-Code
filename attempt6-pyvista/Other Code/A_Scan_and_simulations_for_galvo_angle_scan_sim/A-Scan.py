import numpy as np
import matplotlib.pyplot as plt
"""
This script simulates spectral‐domain optical coherence interferometry by:
1. Defining a Gaussian source spectrum over a wavenumber range.
2. Generating interferograms for two reflectors at 50 μm (10% reflectivity) and 300 μm (5% reflectivity), and their combined signal including interference cross‐terms.
3. Subtracting the DC envelope, performing an FFT to compute the depth‐resolved A‐scan, and scaling the depth axis to micrometres.
4. Plotting: (a) single‐reflector interferogram over the envelope, (b) second reflector, (c) combined signal, and (d) the A‐scan amplitude with annotated peaks.
5. Saving the composite figure as an SVG file.
"""

# -----------------------------------------------------------
# 1. Define parameters
# -----------------------------------------------------------
k_min = 1.0  # [1/μm], arbitrary
k_max = 5.0  # [1/μm], arbitrary
num_points = 1000
k = np.linspace(k_min, k_max, num_points)

# Define a Gaussian envelope for the source spectrum:
k_center = 3
sigma = 0.8
envelope = np.exp(-0.5 * ((k - k_center)/sigma)**2)

# Reflector parameters:
Rr = 1.0
R1 = 0.10  # 10%
z1 = 50.0  # [μm]
R2 = 0.05  # 5%
z2 = 300.0 # [μm]

A1 = np.sqrt(Rr * R1)
A2 = np.sqrt(Rr * R2)

phi1 = 1/4 * z1 * k
phi2 = 2 * z2 * k

# -----------------------------------------------------------
# 2. Construct the individual interferograms
# -----------------------------------------------------------
interferogram_a = envelope * (1 + 2 * A1 * np.cos(phi1))
interferogram_b = envelope * (1 + 2 * A2 * np.cos(phi2))
interferogram_c = envelope * (
    1
    + 2 * A1 * np.cos(phi1)
    + 2 * A2 * np.cos(phi2)
    + 2 * A1 * A2 * np.cos(phi1 - phi2)
)

# -----------------------------------------------------------
# 3. Compute the A-scan via Fourier transform
# -----------------------------------------------------------
# Remove DC (the envelope) to highlight interference fringes
interferogram_c_centered = interferogram_c - envelope*2

# FFT computation
spectrum_c = np.fft.fft(interferogram_c_centered)
freq_axis = np.fft.fftfreq(num_points, d=(k[1]-k[0]))

# Set the scaling factor so that the depth axis extends roughly to 600 μm.
scale_factor = 500 * 2 * (k[1]-k[0])
z_scale = np.abs(freq_axis) * scale_factor

ascan = np.abs(spectrum_c)
half = num_points // 2
z_plot = z_scale[:half]
ascan_plot = ascan[:half]

# -----------------------------------------------------------
# 4. Plotting the results with customizations
# -----------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 7))

# Custom color as requested
custom_color = '#414f73ff'
# Use a decimation factor of 5 for smoother appearance and add markers to show curvature
decimation = 1

# (a) Single reflector at 50 μm
axs[0,0].plot(k[::decimation], interferogram_a[::decimation],
              color=custom_color, markersize=3, linestyle='-',
              label='Interferogram')
axs[0,0].plot(k, envelope, 'k--', label='Envelope')
axs[0,0].set_xlabel('wavenumber k (1/μm)')
axs[0,0].set_ylabel('Intensity')
axs[0,0].set_title('(a) Reflector at 50 μm, R=10%')
axs[0,0].legend(loc='upper left')

# (b) Single reflector at 300 μm
axs[0,1].plot(k[::decimation], interferogram_b[::decimation],
              color=custom_color, markersize=3, linestyle='-',
              label='Interferogram')
axs[0,1].plot(k, envelope, 'k--', label='Envelope')
axs[0,1].set_xlabel('wavenumber k (1/μm)')
axs[0,1].set_ylabel('Intensity')
axs[0,1].set_title('(b) Reflector at 300 μm, R=5%')
axs[0,1].legend(loc='upper left')

# (c) Two reflectors
axs[1,0].plot(k[::decimation], interferogram_c[::decimation],
              color=custom_color, markersize=3, linestyle='-',
              label='Interferogram')
axs[1,0].plot(k, envelope, 'k--', label='Envelope')
axs[1,0].set_xlabel('wavenumber k (1/μm)')
axs[1,0].set_ylabel('Intensity')
axs[1,0].set_title('(c) Signal of both reflectors')
axs[1,0].legend(loc='upper left')

# (d) A-scan plot
axs[1,1].plot(z_plot, ascan_plot, color=custom_color)
axs[1,1].set_xlabel('z-depth [μm]')
axs[1,1].set_ylabel('Reflectivity')
axs[1,1].set_title('(d) A-scan of 2 reflectors')

# Annotate the peaks (rough estimation)
peak_indices = np.argpartition(ascan_plot, -2)[-2:]
peak_indices = peak_indices[np.argsort(ascan_plot[peak_indices])]
peak1, peak2 = peak_indices
z_pk1, z_pk2 = z_plot[peak1], z_plot[peak2]
axs[1,1].text(z_pk1, ascan_plot[peak1],
              '  R1', verticalalignment='bottom', color='red')
axs[1,1].text(z_pk2, ascan_plot[peak2],
              '  R2', verticalalignment='bottom', color='red')

plt.tight_layout()

# Export the figure as an SVG file
plt.savefig("output.svg", format="svg")
plt.show()
