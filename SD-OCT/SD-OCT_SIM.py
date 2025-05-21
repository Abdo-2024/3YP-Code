import numpy as np
import matplotlib.pyplot as plt
"""
This script simulates a spectral‐domain OCT measurement for a single reflector. It defines a Gaussian light source spectrum over a specified bandwidth, computes the spectral interferogram including a cosine fringe term for a reflector at 1 mm depth with reflectivity R, adds white Gaussian noise, and then performs an FFT to obtain the depth‐resolved A‐scan. Finally, it plots the noisy spectral interferogram versus wavelength and the resulting A‐scan amplitude versus depth.
"""

# Simulation parameters
lambda0 = 810e-9         # central wavelength in meters
BW = 30e-9               # full width at half maximum (FWHM) bandwidth in meters
num_pixels = 2048        # number of pixels on the detector (Basler r2l2048-172g5m)

# Generate wavelength array (linear spacing)
lambda_min = lambda0 - BW/2
lambda_max = lambda0 + BW/2
wavelengths = np.linspace(lambda_min, lambda_max, num_pixels)

# Convert wavelengths to wavenumbers (k = 2π/λ)
k = 2 * np.pi / wavelengths

# Define a Gaussian source spectrum S(k) in wavelength space
# The standard deviation is obtained from FWHM (FWHM = 2.355*σ)
sigma = BW / 2.355
S_lambda = np.exp(-((wavelengths - lambda0)**2) / (2 * sigma**2))
# Normalize spectrum for convenience
S_lambda = S_lambda / np.max(S_lambda)

# Define sample parameters: single reflector at depth z0 with reflectivity R
z0 = 1e-3   # reflector depth in meters (1 mm)
R = 0.5     # reflectivity (can be between 0 and 1)
phi = 0.0   # initial phase offset

# Compute the spectral interferogram I(k)
# The interferometric term is modeled as: I(k) = S(k) * [1 + R*cos(2*k*z0 + phi)]
I_k = S_lambda * (1 + R * np.cos(2 * k * z0 + phi))

# Optionally, add some white Gaussian noise (e.g., simulating shot noise)
noise_level = 0.05  # adjust noise level as needed
noise = np.random.normal(0, noise_level, size=I_k.shape)
I_k_noisy = I_k + noise

# To compute the depth profile (A-scan), perform a Fourier transform on the interferogram.
# Note: Ideally, k should be uniformly sampled. If not, you would need to re-sample.
A_scan = np.fft.fft(I_k_noisy)
# Generate the corresponding depth axis.
# The FFT frequency (in terms of k) is computed using the spacing in k.
dk = (k[-1] - k[0]) / (num_pixels - 1)
depth_axis = np.fft.fftfreq(num_pixels, d=dk)
# Since the depth profile is real-valued, take the absolute value and only the positive half.
depth_axis = np.abs(depth_axis)
A_scan_abs = np.abs(A_scan)

# Plot the simulated spectral interferogram and the corresponding A-scan
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(wavelengths * 1e9, I_k_noisy, 'b-')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Simulated Spectral Interferogram')

plt.subplot(1, 2, 2)
plt.plot(depth_axis, A_scan_abs, 'r-')
plt.xlabel('Depth (m)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Simulated A-scan (FFT result)')
plt.tight_layout()
plt.show()
