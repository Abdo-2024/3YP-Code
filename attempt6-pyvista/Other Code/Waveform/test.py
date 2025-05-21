import numpy as np
import matplotlib.pyplot as plt
from opticspy import zernike
"""
This script generates a normalized polar grid, computes the Zernike polynomial of radial degree 4 and azimuthal frequency 0 using opticspy, converts the grid to Cartesian coordinates, and visualises the resulting wavefront shape with a coloured mesh plot.
"""

# Create a polar grid
r = np.linspace(0, 1, 200)             # radial coordinate from 0 to 1 (normalized)
theta = np.linspace(0, 2*np.pi, 200)     # angular coordinate from 0 to 2π
R, Theta = np.meshgrid(r, theta)

# Compute the Zernike polynomial for mode (n=4, m=0)
# (This function call assumes that opticspy.zernike provides a function named "zernike"
#  which takes the radial degree n, azimuthal frequency m, and arrays R and Theta.)
Z = zernike.zernike(4, 0, R, Theta)

# Convert polar grid to Cartesian coordinates for plotting
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Plot the computed Zernike polynomial
plt.figure(figsize=(6,6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.title("Zernike Polynomial (n=4, m=0)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Zernike Value")
plt.axis('equal')
plt.show()
