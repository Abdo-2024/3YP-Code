import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# System Parameters
# -----------------------
lambda0 = 787.3e-9        # wavelength in meters
d_x = 17.11e-6            # lateral resolution (FWHM) in meters
NA = 0.0203               # numerical aperture

# Calculate beam waist (w0) from your given equation:
# d_x = 2 ln2 * w0  =>  w0 = d_x / (2 ln2)
w0 = d_x / (2 * np.log(2))
print(f"Calculated beam waist w0 = {w0:.2e} m")

# Check against the diffraction-limited beam waist:
w0_diff = lambda0 / (np.pi * NA)
print(f"Diffraction-limited beam waist w0_diff = {w0_diff:.2e} m")

# Compute Rayleigh range (z_R) and Depth-of-Focus (DOF ≈ 2z_R)
z_R = np.pi * w0**2 / lambda0
DOF = 2 * z_R
print(f"Rayleigh range z_R = {z_R*1e6:.2f} µm, DOF ≈ {DOF*1e6:.2f} µm")

# -----------------------
# Simulation Grid Setup
# -----------------------
grid_size = 70e-6        # transverse grid size in meters (adjust as needed)
N = 512                   # number of points per dimension
dx_grid = grid_size / N   # spatial sampling interval
x = np.linspace(-grid_size/2, grid_size/2 - dx_grid, N)
y = x.copy()
X, Y = np.meshgrid(x, y)

# -----------------------
# Initial Field: Gaussian Beam at z=0
# -----------------------
# The field amplitude (complex) is given by:
E0 = np.exp( - (X**2 + Y**2) / (w0**2) )
# (Normalization factor omitted; we focus on relative intensity)

# -----------------------
# Prepare Spatial Frequency Grid for FFT
# -----------------------
dk = 2 * np.pi / grid_size
kx = np.fft.fftfreq(N, d=dx_grid) * 2 * np.pi  # angular frequency (rad/m)
ky = kx.copy()
KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
k0 = 2 * np.pi / lambda0  # wave number

# -----------------------
# Angular Spectrum Propagation Function
# -----------------------
def propagate(E_in, z):
    """
    Propagates the complex field E_in by a distance z using the angular spectrum method.
    """
    # Compute FFT of the input field (use fftshift to center zero frequency)
    E_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_in)))
    
    # Calculate the transfer function H(kx, ky, z)
    # sqrt_term ensures non-negative arguments under the square root.
    sqrt_term = np.sqrt(np.maximum(0, k0**2 - KX**2 - KY**2))
    H = np.exp(1j * sqrt_term * z)
    
    # Multiply in Fourier domain and take inverse FFT
    E_fft_prop = E_fft * H
    E_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_fft_prop)))
    return E_out

# -----------------------
# Propagation Simulation Over z
# -----------------------
# We'll simulate propagation from -500 µm to +500 µm relative to the waist.
z_vals = np.linspace(-500e-6, 500e-6, 21)  # 21 steps
FWHM_vs_z = []  # list to store FWHM values (in µm)

# We'll analyze the intensity profile along x at y=0.
center_index = N // 2

plt.figure(figsize=(10,6))
for i, z in enumerate(z_vals):
    E_z = propagate(E0, z)
    I_z = np.abs(E_z)**2
    
    # Extract intensity profile along x (at y = 0)
    I_profile = I_z[center_index, :]
    I_profile = I_profile / np.max(I_profile)  # normalize
    
    # Estimate FWHM: find indices where intensity is >= 0.5
    indices = np.where(I_profile >= 0.5)[0]
    if indices.size > 0:
        fwhm = (indices[-1] - indices[0]) * dx_grid  # in meters
    else:
        fwhm = np.nan
    FWHM_vs_z.append(fwhm * 1e6)  # convert to µm
    
    # Plot profiles for z = -500 µm, 0, and 500 µm
    if i in [0, len(z_vals)//2, len(z_vals)-1]:
        plt.plot(x*1e6, I_profile, label=f'z = {z*1e6:.0f} µm')
        
plt.xlabel('x (µm)')
plt.ylabel('Normalized Intensity')
plt.title('Intensity Profiles along x at Selected z Positions')
plt.legend()
plt.grid(True)
plt.show()

# Plot the estimated FWHM vs. propagation distance
plt.figure(figsize=(8,5))
plt.plot(z_vals*1e6, FWHM_vs_z, 'o-', linewidth=2)
plt.xlabel('Propagation Distance z (µm)')
plt.ylabel('Estimated FWHM (µm)')
plt.title('Beam Spot Size (FWHM) vs. Propagation Distance')
plt.grid(True)
plt.show()

# -----------------------
# Optional: Simulate Passage Through a 0.5 mm Thick Glass Slide
# -----------------------
# When propagating through a medium with refractive index n, the optical path length increases by n.
n_glass = 1.52
thickness_glass = 0.5e-3  # 0.5 mm
z_optical = thickness_glass * n_glass  # effective propagation distance in air

E_after_glass = propagate(E0, z_optical)
I_after_glass = np.abs(E_after_glass)**2

plt.figure(figsize=(8,6))
plt.imshow(I_after_glass, extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6], cmap='inferno')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title(f'Intensity Profile after 0.5 mm Glass (n={n_glass})')
plt.colorbar(label='Intensity (a.u.)')
plt.show()
