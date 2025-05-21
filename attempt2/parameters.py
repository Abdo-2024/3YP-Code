"""
parameters.py

This file holds the central parameters for both noise generation and error detection.
You can adjust these values as needed, and the error detection parameters below are
derived (or suggested) based on the noise parameters.
"""

# ============================
# Noise Generation Parameters
# ============================

# Number of deformed spheres (blobs) added to the main mesh
NUM_SPHERES = 50

# The minimum and maximum radii for the individual spheres that get deformed.
BLOB_RADIUS_MIN = 0.005  # smallest sphere radius
BLOB_RADIUS_MAX = 0.07   # largest sphere radius

# When deforming a sphere, the number of “mini-blob” influences to apply
NUM_BLOBS = 30

# The deformation intensity is a random factor multiplied by the sphere's radius.
# This range defines the factor: the actual intensity = factor * sphere_radius.
BLOB_INTENSITY_MIN = 0.01
BLOB_INTENSITY_MAX = 0.5

# When deforming a sphere, the "blob" has an influence radius defined as a multiple of the sphere's radius.
BLOB_RADIUS_MULTIPLIER = 1.2

# A rough estimate of the maximum possible noise magnitude:
# (e.g., if the sphere had maximum radius and maximum intensity factor)
MAX_NOISE_MAGNITUDE = BLOB_INTENSITY_MAX * BLOB_RADIUS_MAX  # ~0.035 (approx)

# ============================
# Error Detection Parameters
# ============================

# Noise threshold:
# Only vertices with a displacement larger than this value will be considered "deformed".
# Here, we choose a small fixed value (you might adjust this based on your experiments).
NOISE_THRESHOLD = 0.002

# For clustering the "blob" points with DBSCAN:
# We set the eps parameter relative to a typical blob size. Here, we take the average blob radius.
TYPICAL_BLOB_RADIUS = (BLOB_RADIUS_MIN + BLOB_RADIUS_MAX) / 2.0
DBSCAN_EPS = TYPICAL_BLOB_RADIUS * 0.3   # roughly 30% of a typical blob's size
DBSCAN_MIN_SAMPLES = 5                 # minimum number of points required to form a cluster

# ============================
# Visualization Color Mapping
# ============================

# For the error detection visualization (coloring the perfect mesh by noise magnitude,
# and coloring the detected spheres), we define a mapping from small (blue) to large (red).
SMALL_COLOR = [0, 0, 1]  # blue: low noise/deformation
LARGE_COLOR = [1, 0, 0]  # red: high noise/deformation
