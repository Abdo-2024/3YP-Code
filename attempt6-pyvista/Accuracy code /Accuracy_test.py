import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.spatial import cKDTree

def plot_sphere(ax, center, radius, color, alpha=0.3, resolution=20):
    """
    Plots a sphere on the given Axes3D.
    
    Parameters:
      - ax: a matplotlib 3D axis.
      - center: tuple (x, y, z) for the center.
      - radius: sphere radius.
      - color: color of the sphere.
      - alpha: transparency.
      - resolution: number of points for mesh generation.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

# -------------------------
# 1. Load CSV data
# -------------------------
# True headers: x,y,z,radius,label
# Fitted headers: cluster_id,x,y,z,radius,label,points_in,points_out,in_out_ratio
true_df = pd.read_csv("/home/a/Documents/3YP/Code/attempt6-pyvista/sphere_data.csv")
fitted_df = pd.read_csv("/home/a/Documents/3YP/Code/attempt6-pyvista/clusters_with_in_out_ratio.csv")

# -------------------------
# 2. Match true spheres to fitted spheres
# -------------------------
# Build a KDTree for the fitted sphere centers for fast spatial queries.
fitted_coords = fitted_df[['x', 'y', 'z']].to_numpy()
tree = cKDTree(fitted_coords)

# Tolerances (adjust these based on your data scale)
tol_distance = 0.3    # Maximum allowed center-to-center distance for a match
tol_radius_frac = 0.2 # Allowed fractional difference in radii (10%)

matches = []   # To store matching information
unmatched = [] # To record indices of true spheres with no match

for idx, row in true_df.iterrows():
    true_center = np.array([row['x'], row['y'], row['z']])
    true_radius = row['radius']  # Use 'radius' from true CSV
    expected_label = row['label']  # Use the ground truth label from the true CSV
    
    # Find fitted spheres within tol_distance of the true sphere center
    candidate_indices = tree.query_ball_point(true_center, r=tol_distance)
    
    best_candidate = None
    best_distance = np.inf
    for i in candidate_indices:
        fitted_row = fitted_df.iloc[i]
        fitted_radius = fitted_row['radius']  # Use 'radius' from fitted CSV
        # Check if the radii are similar (within tol_radius_frac fraction)
        if abs(fitted_radius - true_radius) <= tol_radius_frac * true_radius:
            # Compute distance between centers
            d = np.linalg.norm(true_center - fitted_row[['x', 'y', 'z']].to_numpy())
            if d < best_distance:
                best_distance = d
                best_candidate = fitted_row
    
    if best_candidate is not None:
        candidate_label = best_candidate['label']
        label_correct = (candidate_label == expected_label)
        matches.append({
            'true_index': idx,
            'fitted_index': best_candidate.name,
            'distance': best_distance,
            'true_radius': true_radius,
            'fitted_radius': best_candidate['radius'],
            'expected_label': expected_label,
            'fitted_label': candidate_label,
            'label_correct': label_correct
        })
    else:
        unmatched.append(idx)

matches_df = pd.DataFrame(matches)
print("Matching results:")
print(matches_df)
if unmatched:
    print("\nThe following true spheres were not captured by any fitted sphere (indices):", unmatched)
else:
    print("\nAll true spheres were captured by a fitted sphere (within tolerances).")

# -------------------------
# 3. Plot the spheres in 3D
# -------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot true spheres in blue
for idx, row in true_df.iterrows():
    center = (row['x'], row['y'], row['z'])
    radius = row['radius']
    plot_sphere(ax, center, radius, color='blue', alpha=0.2)

# Plot fitted spheres; color them based on their label:
# "addition" -> green, "subtraction" -> red.
for idx, row in fitted_df.iterrows():
    center = (row['x'], row['y'], row['z'])
    radius = row['radius']
    label = row['label']
    color = 'green' if label == 'addition' else 'red'
    plot_sphere(ax, center, radius, color=color, alpha=0.3)

# Optionally, draw dashed lines connecting the true sphere centers to their matched fitted sphere centers.
for match in matches:
    true_center = true_df.loc[match['true_index'], ['x', 'y', 'z']].to_numpy()
    fitted_center = fitted_df.loc[match['fitted_index'], ['x', 'y', 'z']].to_numpy()
    ax.plot([true_center[0], fitted_center[0]],
            [true_center[1], fitted_center[1]],
            [true_center[2], fitted_center[2]],
            'k--', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('True Spheres (blue) and Fitted Spheres (green/red)')
plt.show()
