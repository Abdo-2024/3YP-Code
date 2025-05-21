import open3d as o3d
import numpy as np
import random
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score  # For evaluating clustering quality
import csv
from scipy.spatial import cKDTree  # For fast, vectorised nearest‑neighbour queries
"""
This code analyses 3D deformation between two mesh models by aligning them using ICP, detecting deviations through distance comparisons, and identifying deformation regions via DBSCAN clustering. It fits minimal bounding spheres to each cluster to quantify and visualise deformation intensity. Overlapping spheres are merged for more coherent deformation representation, and the final results are visualised along with the original and deformed models. Sphere data is also saved to a CSV file.
"""

# ---------------------------
# Set Random Seeds for Reproducibility
# ---------------------------
np.random.seed(40)
random.seed(40)

# ---------------------------
# User‐defined parameters – adjust these based on your model scale
# ---------------------------
n_points = 50000
threshold_icp = 0.0002         # ICP alignment distance threshold

# We will tune the deformation threshold automatically.
# DBSCAN clustering parameters are manually refined:
dbscan_eps = 0.0087            # DBSCAN eps parameter – smaller values tend to break clusters apart
dbscan_min_samples = 3       # DBSCAN minimum samples

# For colouring the spheres, set lower and upper thresholds for sphere radius.
threshold_low = 0.005   # Lower bound for radius (yellow if below)
threshold_high = 0.015  # Upper bound for radius (red if above)

# ---------------------------
# Importing the models
# ---------------------------
mesh_original = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl"
)
mesh_deformed = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/output_stls/noisy_640.stl"
)
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

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

# Instead of targeting a fixed cluster count, we now choose the threshold based on a smoothed silhouette score.
candidate_thresholds = []
candidate_sil_scores = []
candidate_cluster_counts = []

for candidate_threshold in np.arange(0.005, 0.007, 0.0001):
    candidate_mask = distances > candidate_threshold
    candidate_deformation_points = deformed_points[candidate_mask]
    if candidate_deformation_points.shape[0] == 0:
        continue

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    candidate_labels = dbscan.fit_predict(candidate_deformation_points)
    
    # Count clusters (excluding noise, i.e. label -1).
    unique_labels = set(candidate_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)
    
    if num_clusters < 2:
        continue

    try:
        sil_score = silhouette_score(candidate_deformation_points, candidate_labels)
    except Exception as e:
        print(f"Silhouette score computation failed for threshold {candidate_threshold:.4f}: {e}")
        continue
    
    candidate_thresholds.append(candidate_threshold)
    candidate_sil_scores.append(sil_score)
    candidate_cluster_counts.append(num_clusters)

# If no candidate was valid, default to a threshold of 0.005.
if len(candidate_sil_scores) == 0:
    best_threshold = 0.005
else:
    candidate_sil_scores = np.array(candidate_sil_scores)
    candidate_thresholds = np.array(candidate_thresholds)
    
    # Apply a simple moving average filter to smooth the silhouette scores.
    window = 3  # Adjust the window size as needed.
    smoothed_sil = np.convolve(candidate_sil_scores, np.ones(window)/window, mode='valid')
    
    # Adjust index because the moving average reduces the array length.
    best_index = np.argmax(smoothed_sil) + window//2
    best_threshold = candidate_thresholds[best_index]

print(f"Optimal deformation threshold based on smoothed silhouette score: {best_threshold:.4f}")
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
        colour = [0.5, 0, 0.5]  # Purple for errors > 0.01
    elif radius > 0.007:
        colour = [1, 0, 0]      # Red for errors between 0.007 and 0.01
    elif radius > 0.005:
        colour = [1, 0.65, 0]   # Orange for errors between 0.005 and 0.007
    else:
        colour = [1, 1, 0]      # Yellow for errors <= 0.005

    
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
