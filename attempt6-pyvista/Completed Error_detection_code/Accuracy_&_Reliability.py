import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
"""
This script loads true and fitted sphere datasets from CSV files, matches each true sphere to any fitted sphere that touches it (distance between centers ≤ sum of radii), and records match details and unmatched indices. It saves the match results to a CSV file and uses PyVista to visualise true spheres in light gray, fitted spheres colour-coded by addition/subtraction, and draws lines between touching matches, complete with axes and a legend.
"""

# -------------------------
# 1. Load CSV data
# -------------------------
# True CSV headers: x,y,z,radius,label
# Fitted CSV headers: cluster_id,x,y,z,radius,label,points_in,points_out,in_out_ratio
true_csv = "/home/a/Documents/3YP/Code/attempt6-pyvista/sphere_data.csv"
fitted_csv = "/home/a/Documents/3YP/Code/attempt6-pyvista/cluster_summary.csv"

true_df = pd.read_csv(true_csv)
fitted_df = pd.read_csv(fitted_csv)

# -------------------------
# 2. Match true spheres to fitted spheres using the "touching" criterion
# -------------------------
# Build a KDTree for fitted sphere centers
fitted_coords = fitted_df[['x', 'y', 'z']].to_numpy()
tree = cKDTree(fitted_coords)

# Determine the maximum fitted radius to set a generous search window
max_fitted_radius = fitted_df['radius'].max()

matches = []   # To store matching information
unmatched = [] # To record indices of true spheres with no match

for idx, row in true_df.iterrows():
    true_center = np.array([row['x'], row['y'], row['z']])
    true_radius = row['radius']
    expected_label = row['label']
    
    # Search radius: any fitted sphere whose center lies within (true_radius + max_fitted_radius)
    search_radius = true_radius + max_fitted_radius
    candidate_indices = tree.query_ball_point(true_center, r=search_radius)
    
    best_candidate = None
    best_distance = np.inf
    for i in candidate_indices:
        fitted_row = fitted_df.iloc[i]
        fitted_center = fitted_row[['x', 'y', 'z']].to_numpy()
        fitted_radius = fitted_row['radius']
        d = np.linalg.norm(true_center - fitted_center)
        # Check if spheres are touching (or overlapping)
        if d <= (true_radius + fitted_radius):
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
if unmatched:
    print("\nThe following true spheres were not touched by any fitted sphere (indices):", unmatched)
else:
    print("\nAll true spheres were touched by at least one fitted sphere.")

# Save matching results to a CSV file
matches_df.to_csv("Accuracy_and_Reliability.csv", index=True)

# -------------------------
# 3. Plot using Pyvista
# -------------------------
plotter = pv.Plotter()

# Plot true spheres in light gray
for idx, row in true_df.iterrows():
    center = [row['x'], row['y'], row['z']]
    radius = row['radius']
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color='lightgray', opacity=1)

# Plot fitted spheres; color based on label:
# "addition" -> red, "subtraction" -> blue
for idx, row in fitted_df.iterrows():
    center = [row['x'], row['y'], row['z']]
    radius = row['radius']
    label = row['label']
    color = 'red' if label == 'addition' else 'blue'
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color=color, opacity=0.5)

# Draw lines connecting true sphere centers to their matched fitted sphere centers
for match in matches:
    # Convert endpoints to lists to ensure they are 3-element lists
    true_center = true_df.loc[match['true_index'], ['x', 'y', 'z']].to_numpy().tolist()
    fitted_center = fitted_df.loc[match['fitted_index'], ['x', 'y', 'z']].to_numpy().tolist()
    line = pv.Line(true_center, fitted_center)
    plotter.add_mesh(line, color='black', line_width=3)

# Add visible axes
plotter.add_axes(line_width=2)

# Add a legend explaining the colors and lines
legend_entries = [
    ['True Sphere', 'lightgray'],
    ['Fitted Sphere (Addition)', 'red'],
    ['Fitted Sphere (Subtraction)', 'blue'],
    ['Touching Match Line', 'black']
]
plotter.add_legend(legend_entries, bcolor='white')

# Show the interactive plot
plotter.show()
