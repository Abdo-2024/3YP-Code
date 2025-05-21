import numpy as np
import matplotlib.pyplot as plt
"""
This script simulates spectral‐domain OCT interferograms and A‐scans for two reflectors. It:
1. Defines a Gaussian source spectrum over a 30 nm bandwidth centred at 810 nm, converts to wavenumber.
2. Computes interferograms for a single reflector at 50 µm (10% reflectivity) and at 600 µm (5%), and the combined noisy signal of both.
3. Re‐samples the combined interferogram onto a uniform k‐grid, performs an FFT to obtain the depth‐resolved A‐scan, and converts the depth axis to micrometres.
4. Plots four panels: (a) interferogram at 50 µm, (b) interferogram at 600 µm, (c) noisy combined interferogram, and (d) the resulting OCT A‐scan versus depth.
"""

# -----------------------------
# Simulation parameters
# -----------------------------
lambda0 = 810e-9        # Central wavelength (810 nm)
BW = 30e-9              # Full width at half maximum bandwidth (30 nm)
num_pixels = 2048       # Number of detector pixels

# -----------------------------
# Generate wavelength array (linear spacing)
# -----------------------------
lambda_min = lambda0 - BW/2
lambda_max = lambda0 + BW/2
wavelengths = np.linspace(lambda_min, lambda_max, num_pixels)

# -----------------------------
# Convert wavelengths to wavenumbers (k = 2π/λ)
# -----------------------------
k = 2 * np.pi / wavelengths

# -----------------------------
# Define Gaussian source spectrum S(λ)
# (See Eq. (3.1) from page 63 for the interferogram formulation)
# -----------------------------
sigma = BW / 2.355      # Standard deviation from FWHM (Gaussian relation)
S_lambda = np.exp(-((wavelengths - lambda0)**2) / (2 * sigma**2))
S_lambda = S_lambda / np.max(S_lambda)  # Normalize to 1

# -----------------------------
# Reflector parameters
# -----------------------------
# Reflector 1: 50 µm depth, 10% reflectivity
z1 = 50e-6              # Depth = 50 µm in meters
R1 = 0.10               # Reflectivity 10%

# Reflector 2: 600 µm depth, 5% reflectivity
z2 = 600e-6             # Depth = 600 µm in meters
R2 = 0.05               # Reflectivity 5%

phi = 0.0               # Phase offset (can be set as needed)

# -----------------------------
# Field simulation (using the basic equation from page 63)
# E(λ) = sqrt[S(λ)] for the reference arm, and for each reflector:
# E_reflector = sqrt(R * S(λ)) * exp(i·2kz)
# -----------------------------
E_ref = np.sqrt(S_lambda)  # Reference field amplitude

# Fields from reflectors
E1 = np.sqrt(R1 * S_lambda) * np.exp(1j * 2 * k * z1)
E2 = np.sqrt(R2 * S_lambda) * np.exp(1j * 2 * k * z2)

# -----------------------------
# Combined interferometric signal (from page 63-65)
# I(λ) = |E_ref + E1 + E2|^2 = S(λ) · [1 + interference terms]
# Here we simulate the interferogram as in Eq. (3.1)
# -----------------------------
E_total = E_ref + E1 + E2
I_c = np.abs(E_total)**2

# -----------------------------
# OPTIONAL: Add white Gaussian noise to simulate a realistic measurement
# -----------------------------
noise_level = 0.05
noise = np.random.normal(0, noise_level, size=I_c.shape)
I_c_noisy = I_c + noise

# -----------------------------
# IMPORTANT: Re-sample the interferogram onto a uniform k-space grid
# The FFT assumes uniformly spaced data. Note: If your original 'k' is not monotonic,
# flip the arrays so that k increases.
# -----------------------------
if k[0] > k[-1]:
    k = np.flipud(k)
    I_c_noisy = np.flipud(I_c_noisy)

# Create a uniform k grid
k_uniform = np.linspace(k.min(), k.max(), num_pixels)
# Interpolate the noisy interferogram onto the uniform k grid
I_uniform = np.interp(k_uniform, k, I_c_noisy)

# -----------------------------
# Compute the A-scan (depth profile) via FFT
# (Refer to the Fourier relation in Eq. (3.3) from page 64)
# -----------------------------
I_fft = np.fft.fft(I_uniform)
I_fft = np.abs(np.fft.fftshift(I_fft))  # Shift zero frequency to center and take magnitude

# -----------------------------
# Define uniform k spacing and corresponding depth axis
# Depth is proportional to the FFT frequency; here we use the relation:
# depth_axis = |frequency|/2. (see page 64-65)
# -----------------------------
dk = (k_uniform[-1] - k_uniform[0]) / (num_pixels - 1)
freq_axis = np.fft.fftfreq(num_pixels, d=dk)
freq_axis = np.fft.fftshift(freq_axis)
depth_axis = np.abs(freq_axis) / 2  # in meters
depth_axis_um = depth_axis * 1e6   # convert depth to micrometers

# -----------------------------
# Plotting the results
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Interferogram: Single reflector at 50 µm (only using E_ref + E1)
E_total_a = E_ref + E1
I_a = np.abs(E_total_a)**2
axs[0, 0].plot(wavelengths * 1e9, I_a, 'b-', lw=2)
axs[0, 0].set_xlabel('Wavelength (nm)')
axs[0, 0].set_ylabel('Intensity (a.u.)')
axs[0, 0].set_title('Interferogram: Reflector at 50 µm (10% R)')

# Panel (b): Interferogram: Single reflector at 600 µm (only using E_ref + E2)
E_total_b = E_ref + E2
I_b = np.abs(E_total_b)**2
axs[0, 1].plot(wavelengths * 1e9, I_b, 'g-', lw=2)
axs[0, 1].set_xlabel('Wavelength (nm)')
axs[0, 1].set_ylabel('Intensity (a.u.)')
axs[0, 1].set_title('Interferogram: Reflector at 600 µm (5% R)')

# Panel (c): Combined interferogram with both reflectors (noisy version)
axs[1, 0].plot(wavelengths * 1e9, I_c_noisy, 'r-', lw=1)
axs[1, 0].set_xlabel('Wavelength (nm)')
axs[1, 0].set_ylabel('Intensity (a.u.)')
axs[1, 0].set_title('Interferogram: Both Reflectors (Noisy)')

# Panel (d): OCT A-scan (Depth Profile) from the FFT of the uniformly re-sampled interferogram
axs[1, 1].plot(depth_axis_um, I_fft, 'm-', lw=2)
axs[1, 1].set_xlabel('Depth (µm)')
axs[1, 1].set_ylabel('Amplitude (a.u.)')
axs[1, 1].set_title('OCT A-scan (Depth Profile)')
axs[1, 1].set_xlim(0, 1000)  # Adjust depth range as needed

plt.tight_layout()
plt.show()
