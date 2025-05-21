import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Define system parameters
# -----------------------
lambda0 = 787.3e-9           # wavelength (m)
NA = 0.0203                  # numerical aperture
dx_fwhm = 17.11e-6           # lateral resolution (FWHM, m)

# According to the given relation: d_x = 2 ln2 * w0
ln2 = np.log(2)
w0 = dx_fwhm / (2 * ln2)     # beam waist (m)
print("Beam waist w0 = {:.2e} m".format(w0))

# (Optional) Compute Rayleigh range in air:
z_R = np.pi * w0**2 / lambda0
print("Rayleigh range z_R = {:.2e} m".format(z_R))
# Note: DOF ~ 2*z_R; here our target DOF is about 450 µm.

# -----------------------
# Set up spatial grid
# -----------------------
# We'll simulate a transverse window that covers several beam widths.
grid_size = 200e-6         # physical size of grid (m)
N = 512                    # number of grid points per axis
dx = grid_size / N         # spatial sampling interval (m)
x = np.linspace(-grid_size/2, grid_size/2 - dx, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# -----------------------
# Define initial Gaussian field at z = 0 (beam waist)
# -----------------------
# For a Gaussian beam (using the convention for intensity I ~ exp(-2r^2/w0^2)),
# here we define the field as:
E0 = np.exp(-(X**2 + Y**2) / (w0**2))
# (Note: Depending on convention you might include an amplitude factor; we set it to 1)

# -----------------------
# Prepare FFT grids
# -----------------------
# Frequency grid (angular frequencies) corresponding to x and y
dk = 2 * np.pi / grid_size
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # convert to angular frequency
ky = kx.copy()
KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))

# Total wave number
k = 2 * np.pi / lambda0

# -----------------------
# Define propagation function using the angular spectrum method
# -----------------------
def propagate(E_in, z):
    """
    Propagates the complex field E_in over a distance z using the angular spectrum method.
    """
    # Compute Fourier transform of input field
    E_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_in)))
    
    # Compute the transfer function H(kx,ky,z)
    # Under the square-root, avoid negative values by using np.maximum(0, ...)
    H = np.exp(1j * np.sqrt(np.maximum(0, k**2 - KX**2 - KY**2)) * z)
    
    # Multiply in frequency domain
    E_fft_prop = E_fft * H
    
    # Inverse Fourier transform to get the propagated field
    E_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_fft_prop)))
    return E_out

# -----------------------
# Simulate propagation over a range of z
# -----------------------
# We will simulate z from -600 µm to +600 µm (relative to focus) in 21 steps.
z_values = np.linspace(-600e-6, 600e-6, 21)
spot_sizes = []  # to store FWHM estimates at each z

# For simplicity, we'll analyze the intensity profile along x at y=0.
y0_index = N // 2

plt.figure(figsize=(10, 6))
for i, z in enumerate(z_values):
    E_z = propagate(E0, z)
    I_z = np.abs(E_z)**2
    
    # Extract 1D intensity profile along x (at y = 0)
    I_profile = I_z[y0_index, :]
    
    # Normalize the profile for analysis
    I_profile = I_profile / np.max(I_profile)
    
    # Estimate FWHM: find indices where intensity >= 0.5
    indices = np.where(I_profile >= 0.5)[0]
    if len(indices) > 0:
        fwhm = (indices[-1] - indices[0]) * dx  # in meters
    else:
        fwhm = 0
    spot_sizes.append(fwhm * 1e6)  # convert to µm
    
    # Plot selected profiles (for z = -600 µm, 0, and 600 µm)
    if i in [0, len(z_values)//2, -1]:
        plt.plot(x*1e6, I_profile, label="z = {:.0f} µm".format(z*1e6))

plt.xlabel("x (µm)")
plt.ylabel("Normalized Intensity")
plt.title("Intensity Profiles along x at Selected z")
plt.legend()
plt.grid(True)
plt.show()

# Plot beam spot size (FWHM) vs propagation distance
plt.figure(figsize=(8, 5))
plt.plot(z_values*1e6, spot_sizes, 'o-')
plt.xlabel("Propagation distance z (µm)")
plt.ylabel("Estimated FWHM (µm)")
plt.title("Beam Spot Size (FWHM) vs. Propagation Distance")
plt.grid(True)
plt.show()

# -----------------------
# (Optional) Simulate passage through a 0.5mm glass slide (n=1.52)
# -----------------------
# When the beam passes through a medium with refractive index n, the optical path length changes.
n_glass = 1.52
thickness_glass = 0.5e-3  # 0.5 mm

# One simple approach is to simulate propagation in air for a distance equal to the optical thickness:
z_optical = thickness_glass * n_glass

# Propagate the beam through the glass (without refraction corrections, just to show extra phase delay)
E_after_glass = propagate(E0, z_optical)
I_after_glass = np.abs(E_after_glass)**2

plt.figure(figsize=(8, 5))
plt.imshow(I_after_glass, extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6], cmap='inferno')
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Intensity Profile after Propagation through {} mm glass (n={})".format(thickness_glass*1e3, n_glass))
plt.colorbar(label="Intensity (a.u.)")
plt.show()
