import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
"""
This script reads true and fitted sphere datasets from CSV files, builds a KDTree on fitted sphere centers, and matches each true sphere to the nearest fitted sphere within distance and radius tolerances, recording match quality and unmatched indices. It then uses PyVista to visualise true spheres in light grey, fitted spheres colour-coded by label, and draws lines connecting each true sphere to its matched fitted counterpart in 3D.
"""

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
# Build a KDTree for fitted sphere centers
fitted_coords = fitted_df[['x', 'y', 'z']].to_numpy()
tree = cKDTree(fitted_coords)

# Tolerances (adjust these based on your data scale)
tol_distance = 0.1    # maximum allowed center-to-center distance
tol_radius_frac = 0.1 # allowed fractional difference in radii (10%)

matches = []   # to store matching information
unmatched = [] # to record indices of true spheres with no match

for idx, row in true_df.iterrows():
    true_center = np.array([row['x'], row['y'], row['z']])
    true_radius = row['radius']
    expected_label = row['label']
    
    # Find fitted spheres within tol_distance of the true sphere center
    candidate_indices = tree.query_ball_point(true_center, r=tol_distance)
    
    best_candidate = None
    best_distance = np.inf
    for i in candidate_indices:
        fitted_row = fitted_df.iloc[i]
        fitted_radius = fitted_row['radius']
        # Check if the radii are similar (within tol_radius_frac)
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
# 3. Plot using Pyvista
# -------------------------
plotter = pv.Plotter()

# Plot true spheres (blue)
for idx, row in true_df.iterrows():
    center = (row['x'], row['y'], row['z'])
    radius = row['radius']
    # Create a Pyvista sphere (increasing theta/phi resolution improves quality)
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color='lightgray', opacity=1)

# Plot fitted spheres; color based on label ("addition" -> red, "subtraction" -> blue)
for idx, row in fitted_df.iterrows():
    center = (row['x'], row['y'], row['z'])
    radius = row['radius']
    label = row['label']
    color = 'red' if label == 'addition' else 'blue'
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color=color, opacity=0.5)

# Draw lines connecting true sphere centers to their matched fitted sphere centers
for match in matches:
    true_center = true_df.loc[match['true_index'], ['x', 'y', 'z']].to_numpy()
    fitted_center = fitted_df.loc[match['fitted_index'], ['x', 'y', 'z']].to_numpy()
    # Create a line between the centers
    line = pv.Line(true_center, fitted_center)
    plotter.add_mesh(line, color='black', line_width=9)

# Add axes and show the interactive plot
plotter.add_axes(line_width=2)
plotter.show()
