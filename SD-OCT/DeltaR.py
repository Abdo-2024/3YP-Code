import numpy as np
import matplotlib.pyplot as plt
"""
This script simulates spectral‐domain OCT signals for a single reflector at 1 mm depth with varying reflectivities. It:
1. Defines a Gaussian source spectrum over a specified bandwidth and converts to wavenumber.
2. Generates noisy spectral interferograms I(λ) for reflectivities [0.1,0.2,0.4,0.8,1.0] by adding cosine fringes and white noise.
3. Computes the corresponding A‐scans (depth profiles) via FFT of each noisy interferogram.
4. Plots the noisy spectral interferograms (top) and the resulting OCT depth profiles in micrometres (bottom), with legends and grid.
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

# Define sample parameters
z0 = 1e-3    # reflector depth (1 mm)
phi = 0.0    # phase offset

# Define a list of reflectivities
reflectivities = [0.1, 0.2, 0.4, 0.8, 1.0]

# Set noise level 
noise_level = 0.05

# Create figure with two subplots:
# Top subplot: Noisy spectral interferograms for different reflectivities
# Bottom subplot: Corresponding A-scans (depth profiles)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

colors = ['b', 'g', 'r', 'c', 'm']

for idx, R in enumerate(reflectivities):
    # Compute the noiseless spectral interferogram I(λ)
    I_lambda = S_lambda * (1 + R * np.cos(2 * k * z0 + phi))
    
    # Add white Gaussian noise
    noise = np.random.normal(0, noise_level, size=I_lambda.shape)
    I_lambda_noisy = I_lambda + noise

    # Plot the noisy spectral interferogram
    axs[0].plot(wavelengths * 1e9, I_lambda_noisy, color=colors[idx],
                lw=1.5, label=f'R = {R}')
    
    # Compute the depth profile (A-scan) using FFT
    I_fft = np.fft.fft(I_lambda_noisy)
    I_fft = np.abs(np.fft.fftshift(I_fft))  # shift zero frequency to center and take magnitude
    
    # Generate the depth axis (in meters)
    dk = (k[-1] - k[0]) / (num_pixels - 1)
    freq_axis = np.fft.fftfreq(num_pixels, d=dk)
    freq_axis = np.fft.fftshift(freq_axis)
    depth_axis = np.abs(freq_axis) / 2  # conversion factor for OCT depth
    
    # Convert depth axis to micrometers (µm) for better visualization
    depth_axis_um = depth_axis * 1e6
    
    # Plot the A-scan (depth profile)
    axs[1].plot(depth_axis_um, I_fft, color=colors[idx],
                lw=1.5, label=f'R = {R}')

# Customize top subplot: spectral interferograms
axs[0].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('Intensity (a.u.)')
axs[0].set_title('Noisy Spectral Interferograms for Different Reflectivities')
axs[0].legend()
axs[0].grid(True)

# Customize bottom subplot: A-scans
axs[1].set_xlabel('Depth (µm)')
axs[1].set_ylabel('Amplitude (a.u.)')
axs[1].set_title('OCT A-scan (Depth Profiles) for Different Reflectivities')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(0, 500)  # adjust as needed for your depth range

plt.tight_layout()
plt.show()
