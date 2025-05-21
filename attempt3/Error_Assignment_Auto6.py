import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score  # For evaluating clustering quality
import csv
from scipy.spatial import cKDTree  # For fast, vectorised nearest‑neighbour queries
"""
3D Model Deformation Analysis and Visualisation using Open3D, DBSCAN, and Minimal Bounding Spheres with Silhouette Score

This script performs deformation analysis on a pair of 3D models using Open3D and DBSCAN clustering, enhanced with silhouette score evaluation:
1. Loads original and deformed STL meshes using Open3D.
2. Aligns the deformed model to the original using Iterative Closest Point (ICP) registration.
3. Extracts deformation points based on an adaptive nearest-neighbour search.
4. Iteratively tunes the deformation threshold using silhouette score to optimise clustering quality.
5. Clusters deformation points into distinct blobs using DBSCAN.
6. Fits minimal bounding spheres to each cluster, calculating deformation intensity.
7. Visualises the original model and overlaid spheres representing deformation using Open3D.

Key Functionalities:
- Load and sample points from STL meshes for comparative analysis.
- Register meshes using ICP to align deformed points with original points.
- Automatically tune deformation threshold using silhouette score for improved clustering quality.
- Cluster deformation points into distinct blobs using DBSCAN.
- Fit minimal bounding spheres to quantify and visualise deformation intensity.
- Export sphere data (centre, radius, deformation intensity) to a CSV file.
- Visualise the original model and spheres representing deformation using Open3D.

User-defined Parameters:
- n_points: Number of points sampled from each mesh.
- threshold_icp: Threshold for ICP alignment distance.
- dbscan_eps: DBSCAN epsilon parameter for clustering deformation points.
- dbscan_min_samples: DBSCAN minimum samples parameter.
- threshold_low, threshold_high: Colour thresholds for sphere radius to indicate deformation magnitude.

Dependencies:
- Open3D (imported as o3d): For 3D data processing, visualization, and registration.
- NumPy (imported as np): For numerical operations and array handling.
- sklearn.cluster.DBSCAN: For density-based clustering of deformation points.
- sklearn.metrics.silhouette_score: For evaluating clustering quality using silhouette score.
- csv: For writing sphere data to a CSV file.
- scipy.spatial.cKDTree: For efficient nearest-neighbour search.

Usage:
1. Replace the file paths ("/home/a/Documents/.../2x2_MN_Array_scaled.stl", ".../2x2_MN_Array_scaled+Noise.stl")
   with the actual paths to your STL files.
2. Adjust user-defined parameters (n_points, thresholds, dbscan_eps, dbscan_min_samples) based on your model and
   deformation characteristics.
3. Run the script to load, align, analyse, and visualise the deformation in the 3D model.
   It will generate overlaid spheres on the deformed model, where each sphere represents a cluster of deformation points,
   coloured according to the magnitude of deformation.

Note:
- Ensure Open3D and necessary dependencies are installed (`pip install open3d numpy scikit-learn scipy`).
- The script assumes valid STL models with surface details for point sampling and deformation analysis.
- Adjust thresholds and parameters to suit your specific model and deformation analysis requirements.
"""

# ---------------------------
# User‐defined parameters – adjust these based on your model scale
# ---------------------------
n_points = 50000
threshold_icp = 0.0002         # ICP alignment distance threshold

# We will tune the deformation threshold automatically.
# DBSCAN clustering parameters are manually refined:
dbscan_eps = 0.01            # DBSCAN eps parameter – smaller values tend to break clusters apart
dbscan_min_samples = 4       # DBSCAN minimum samples

# For colouring the spheres, set lower and upper thresholds for sphere radius.
threshold_low = 0.005   # Lower bound for radius (yellow if below)
threshold_high = 0.015  # Upper bound for radius (red if above)

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
mesh_original = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl"
)
mesh_deformed = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl"
)
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

# ---------------------------
# Step 2: Align the Two Models using ICP
# ---------------------------
trans_init = np.eye(4)  # initial transformation (identity)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_deformed, pcd_original, threshold_icp, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP Transformation:\n", reg_p2p.transformation)
pcd_deformed.transform(reg_p2p.transformation)

# ---------------------------
# Step 3: Extract Noise (Deformation) Points with Adaptive Threshold Tuning
# ---------------------------
original_points = np.asarray(pcd_original.points)
deformed_points = np.asarray(pcd_deformed.points)
tree = cKDTree(original_points)
distances, _ = tree.query(deformed_points, k=1)

# Instead of targeting a fixed cluster count, choose the threshold that gives the best silhouette score.
best_threshold = None
best_silhouette = -1.0

for candidate_threshold in np.arange(0.005, 0.007, 0.0001):
    candidate_mask = distances > candidate_threshold
    candidate_deformation_points = deformed_points[candidate_mask]
    if candidate_deformation_points.shape[0] == 0:
        continue
    
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    candidate_labels = dbscan.fit_predict(candidate_deformation_points)
    
    # Need at least 2 clusters to compute the silhouette score.
    if len(set(candidate_labels)) < 2:
        continue
    
    try:
        sil_score = silhouette_score(candidate_deformation_points, candidate_labels)
    except Exception as e:
        print(f"Silhouette score computation failed for threshold {candidate_threshold:.4f}: {e}")
        continue
    
    if sil_score > best_silhouette:
        best_silhouette = sil_score
        best_threshold = candidate_threshold

if best_threshold is None:
    print("No valid threshold found using silhouette score. Defaulting to 0.005")
    best_threshold = 0.005

print(f"Optimal deformation threshold based on silhouette score: {best_threshold:.4f} (silhouette score: {best_silhouette:.4f})")

mask = distances > best_threshold
deformation_points = deformed_points[mask]
print("Number of deformation points (using tuned threshold):", len(deformation_points))
if len(deformation_points) == 0:
    print("No deformation points detected")
    # Optionally exit:
    # import sys
    # sys.exit()

# ---------------------------
# Step 4: Cluster the Deformation Points into Blobs using DBSCAN
# ---------------------------
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
labels = dbscan.fit_predict(deformation_points)
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)
print("Detected blobs (clusters):", len(unique_labels))
if len(unique_labels) == 0:
    print("No clusters detected. Visualising the deformation points to adjust DBSCAN parameters.")
    deformation_pcd = o3d.geometry.PointCloud()
    deformation_pcd.points = o3d.utility.Vector3dVector(deformation_points)
    o3d.visualization.draw_geometries([deformation_pcd])
    # Optionally exit:
    # import sys
    # sys.exit()

# ---------------------------
# Step 5: Fit a Sphere to Each Blob using a Minimal Bounding Sphere (Ritter's Algorithm)
# and Compute Deformation Intensity
# ---------------------------
def minimal_bounding_sphere(points):
    """
    Compute an approximate minimal sphere that encloses all points.
    Implements Ritter's bounding sphere algorithm.
    Returns: (centre, radius)
    """
    points = np.array(points)
    if len(points) == 0:
        return None, None

    p0 = points[0]
    distances_p0 = np.linalg.norm(points - p0, axis=1)
    i = np.argmax(distances_p0)
    p1 = points[i]
    distances_p1 = np.linalg.norm(points - p1, axis=1)
    j = np.argmax(distances_p1)
    p2 = points[j]
    centre = (p1 + p2) / 2.0
    radius = np.linalg.norm(p2 - centre)
    for p in points:
        d = np.linalg.norm(p - centre)
        if d > radius:
            new_radius = (radius + d) / 2.0
            centre = centre + (p - centre) * ((new_radius - radius) / d)
            radius = new_radius
    return centre, radius

blobs = {}
csv_data = {}

# Save each blob's sphere parameters and also store the contributing error points.
for label in unique_labels:
    blob_points = deformation_points[labels == label]
    centre, radius = minimal_bounding_sphere(blob_points)
    if centre is None:
        continue
        
    distances_blob = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances_blob - radius)
    deformation_intensity = np.mean(residuals)
    
    if radius > 0.01:
        colour = [0.5, 0, 0.5]    # Purple for errors > 0.01
    elif radius > 0.005:
        colour = [1, 0, 0]        # Red for errors between 0.005 and 0.01
    elif radius > 0.001:
        colour = [1, 0.65, 0]     # Orange for errors between 0.001 and 0.005
    else:
        colour = [1, 1, 0]        # Yellow for errors around 0.001 or smaller
    
    blobs[label] = {
        "centre": centre,
        "radius": radius,
        "colour": colour,
        "deformation_intensity": deformation_intensity,
        "points": blob_points  # Store the error points for later merging.
    }
    
    csv_data[label] = {
        'x': centre[0],
        'y': centre[1],
        'z': centre[2],
        'radius': radius,
        'deformation_intensity': deformation_intensity
    }

# Write the sphere data to a CSV file.
csv_filename = "sphere_data_detection.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data.values():
        writer.writerow(row)
print(f"Sphere data written to {csv_filename}")

# ---------------------------
# Merging Overlapping Spheres Based on Error Points
# ---------------------------
def merge_overlapping_spheres(spheres):
    """
    Merge any number of overlapping spheres based on their error points.
    Two spheres are considered overlapping if the distance between their centres is 
    less than or equal to the sum of their radii.
    
    For each connected group of overlapping spheres, we aggregate all of the error points
    and compute the minimal bounding sphere for the group.
    """
    n = len(spheres)
    # Set up a union-find (disjoint set) structure.
    parent = list(range(n))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path compression.
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    # Build the overlap graph: if two spheres overlap, union their indices.
    for i in range(n):
        for j in range(i + 1, n):
            c1, r1 = spheres[i]['centre'], spheres[i]['radius']
            c2, r2 = spheres[j]['centre'], spheres[j]['radius']
            d = np.linalg.norm(c1 - c2)
            if d <= (r1 + r2):
                union(i, j)
    
    # Group spheres by their connected component.
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(spheres[i])
    
    merged_spheres = []
    for group in groups.values():
        # If the group contains only one sphere, keep it as is.
        if len(group) == 1:
            merged_spheres.append(group[0])
            continue

        # Aggregate all error points from the group.
        all_points = np.vstack([s['points'] for s in group])
        new_centre, new_radius = minimal_bounding_sphere(all_points)
        
        # Average the deformation intensities.
        intensities = np.array([s['deformation_intensity'] for s in group])
        new_intensity = np.mean(intensities)
        
        # Determine the colour based on the new radius.
        if new_radius > 0.01:
            new_colour = [0.5, 0, 0.5]
        elif new_radius > 0.005:
            new_colour = [1, 0, 0]
        elif new_radius > 0.001:
            new_colour = [1, 0.65, 0]
        else:
            new_colour = [1, 1, 0]
        
        merged_spheres.append({
            "centre": new_centre,
            "radius": new_radius,
            "colour": new_colour,
            "deformation_intensity": new_intensity
        })
    
    return merged_spheres

# Convert the blobs dictionary into a list of sphere definitions.
spheres_list = [s for s in blobs.values()]
merged_spheres = merge_overlapping_spheres(spheres_list)
print(f"Number of merged spheres: {len(merged_spheres)}")

# ---------------------------
# Step 6: Visualise the Original Model and Overlaid (Merged) Spheres
# ---------------------------
sphere_meshes = []
for sphere_params in merged_spheres:
    centre = sphere_params["centre"]
    radius = sphere_params["radius"]
    colour = sphere_params["colour"]
    
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere_mesh.translate(centre)
    sphere_mesh.paint_uniform_color(colour)
    sphere_meshes.append(sphere_mesh)

geometries = [mesh_original, mesh_deformed] + sphere_meshes
o3d.visualization.draw_geometries(geometries)
