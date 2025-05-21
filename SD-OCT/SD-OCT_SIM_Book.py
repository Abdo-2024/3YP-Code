import numpy as np
import matplotlib.pyplot as plt
"""
This script simulates a spectral‐domain OCT measurement for a single reflector. It:
1. Defines a Gaussian light source spectrum over a specified bandwidth centred at 810 nm.
2. Computes the noiseless spectral interferogram for a reflector at 1 mm depth with a given reflectivity.
3. Adds white Gaussian noise to the interferogram.
4. Performs an FFT on the noisy interferogram to generate the depth‐resolved A‐scan.
5. Plots in a 2×2 grid: (a) the source spectrum, (b) the noiseless interferogram, (c) the noisy interferogram, and (d) the resulting A‐scan amplitude versus depth.
"""

# Simulation parameters
lambda0 = 810e-9         # central wavelength in meters
BW = 30e-9               # full width at half maximum bandwidth in meters
num_pixels = 2048        # number of pixels on the detector

# Generate wavelength array (linear spacing)
lambda_min = lambda0 - BW/2
lambda_max = lambda0 + BW/2
wavelengths = np.linspace(lambda_min, lambda_max, num_pixels)

# Convert wavelengths to wavenumbers (k = 2π/λ)
k = 2 * np.pi / wavelengths

# Define Gaussian source spectrum S(λ)
sigma = BW / 2.355      # standard deviation from FWHM
S_lambda = np.exp(-((wavelengths - lambda0)**2) / (2 * sigma**2))
S_lambda = S_lambda / np.max(S_lambda)  # normalize

# Define sample parameters: single reflector at depth z0 with reflectivity R
z0 = 1e-3    # reflector depth (1 mm)
R = 0.05      # reflectivity
phi = 0.0    # phase offset

# Compute the noiseless spectral interferogram I(λ)
I_lambda = S_lambda * (1 + R * np.cos(2 * k * z0 + phi))

# Add white Gaussian noise to simulate a realistic measurement
noise_level = 0.05  # adjust noise level as needed
noise = np.random.normal(0, noise_level, size=I_lambda.shape)
I_lambda_noisy = I_lambda + noise

# Compute the depth profile (A-scan) using FFT
# Note: Ideally, k should be uniformly sampled. Here we assume approximate uniformity.
I_fft = np.fft.fft(I_lambda_noisy)
I_fft = np.abs(np.fft.fftshift(I_fft))  # shift zero frequency to center and take magnitude

# Generate the depth axis
dk = (k[-1] - k[0]) / (num_pixels - 1)
freq_axis = np.fft.fftfreq(num_pixels, d=dk)
freq_axis = np.fft.fftshift(freq_axis)
# Convert frequency axis to depth (using the relation depth ~ frequency/(2))
depth_axis = np.abs(freq_axis) / 2

# Create the 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot (a): Gaussian source spectrum S(λ)
axs[0, 0].plot(wavelengths * 1e9, S_lambda, 'b-', lw=2)
axs[0, 0].set_xlabel('Wavelength (nm)')
axs[0, 0].set_ylabel('Normalized Intensity')
axs[0, 0].set_title('Gaussian Source Spectrum S(λ)')

# Plot (b): Noiseless spectral interferogram I(λ)
axs[0, 1].plot(wavelengths * 1e9, I_lambda, 'g-', lw=2)
axs[0, 1].set_xlabel('Wavelength (nm)')
axs[0, 1].set_ylabel('Intensity (a.u.)')
axs[0, 1].set_title('Noiseless Spectral Interferogram')

# Plot (c): Noisy spectral interferogram I(λ) with noise
axs[1, 0].plot(wavelengths * 1e9, I_lambda_noisy, 'r-', lw=1)
axs[1, 0].set_xlabel('Wavelength (nm)')
axs[1, 0].set_ylabel('Intensity (a.u.)')
axs[1, 0].set_title('Noisy Spectral Interferogram')

# Plot (d): A-scan (FFT of the noisy interferogram)
axs[1, 1].plot(depth_axis, I_fft, 'm-', lw=2)
axs[1, 1].set_xlabel('Depth (m)')
axs[1, 1].set_ylabel('Amplitude (a.u.)')
# axs[1, 1].set_xlim(0, 1000)  # adjust as needed for your depth range
axs[1, 1].set_title('A-scan (Depth Profile)')

plt.tight_layout()
plt.show()
