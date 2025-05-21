import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import csv
from scipy.spatial import cKDTree  # For fast, vectorised nearest-neighbour queries
"""
3D Model Deformation Analysis and Visualisation using Open3D, DBSCAN, and Minimal Bounding Spheres

This script performs deformation analysis on a pair of 3D models using Open3D and DBSCAN clustering:
1. Loads original and deformed STL meshes using Open3D.
2. Aligns the deformed model to the original using Iterative Closest Point (ICP) registration.
3. Extracts deformation points based on an adaptive nearest-neighbour search.
4. Iteratively tunes the deformation threshold to achieve a target number of clusters using DBSCAN.
5. Clusters deformation points into distinct blobs using DBSCAN.
6. Fits minimal bounding spheres to each cluster, calculating deformation intensity.
7. Visualises the original model and overlaid spheres representing deformation using Open3D.

Key Functionalities:
- Load and sample points from STL meshes for comparative analysis.
- Register meshes using ICP to align deformed points with original points.
- Automatically tune deformation threshold using iterative DBSCAN clustering.
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
dbscan_eps = 0.0095            # DBSCAN eps parameter – smaller value tends to break clusters apart
dbscan_min_samples = 5         # DBSCAN minimum samples

# For colouring the spheres, set lower and upper thresholds for sphere radius.
threshold_low = 0.005   # Lower bound for radius (yellow if below)
threshold_high = 0.015  # Upper bound for radius (red if above)

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
# Load the original and deformed STL meshes.
mesh_original = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl"
)
mesh_deformed = o3d.io.read_triangle_mesh(
    "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled+Noise.stl"
)

# Ensure the meshes have vertex normals (for better ICP registration and visualisation)
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# Sample points from the surface of each mesh.
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
# Step 3: Extract Noise (Deformation) Points with Iterative Tuning
# ---------------------------
# Convert Open3D point clouds to NumPy arrays.
original_points = np.asarray(pcd_original.points)
deformed_points = np.asarray(pcd_deformed.points)

# Build a fast KD-tree on the original points.
tree = cKDTree(original_points)

# Query the nearest neighbour for all deformed points in one vectorised call.
distances, _ = tree.query(deformed_points, k=1)

# Now we perform an iterative (grid search) over candidate deformation thresholds.
# The idea is to choose the threshold that produces a number of clusters closest to a target.
target_clusters = 50  # We expect around 50 errors to be detected.
best_threshold = None
best_diff = float('inf')
best_candidate_cluster_count = 0

# Iterate over candidate thresholds from 0.005 to 0.007 in steps of 0.0001.
for candidate_threshold in np.arange(0.005, 0.007, 0.0001):
    # Create a mask for deformation points using the candidate threshold.
    candidate_mask = distances > candidate_threshold
    candidate_deformation_points = deformed_points[candidate_mask]
    
    # Only proceed if there are some candidate points.
    if candidate_deformation_points.shape[0] == 0:
        continue
    
    # Cluster these candidate deformation points with DBSCAN.
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    candidate_labels = dbscan.fit_predict(candidate_deformation_points)
    candidate_unique_labels = set(candidate_labels)
    # Remove noise points (labelled as -1).
    if -1 in candidate_unique_labels:
        candidate_unique_labels.remove(-1)
    num_clusters = len(candidate_unique_labels)
    
    diff = abs(num_clusters - target_clusters)
    if diff < best_diff:
        best_diff = diff
        best_threshold = candidate_threshold
        best_candidate_cluster_count = num_clusters

print(f"Best candidate deformation_threshold: {best_threshold:.4f}, yielding {best_candidate_cluster_count} clusters")

# Use the best threshold found to extract the final set of deformation points.
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
    print("No clusters detected. You may want to visualise the deformation points to adjust DBSCAN parameters.")
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
    Returns: (center, radius)
    """
    points = np.array(points)
    if len(points) == 0:
        return None, None

    # Pick an arbitrary point (p0)
    p0 = points[0]
    # Find the point (p1) farthest from p0.
    distances = np.linalg.norm(points - p0, axis=1)
    i = np.argmax(distances)
    p1 = points[i]
    # Find the point (p2) farthest from p1.
    distances = np.linalg.norm(points - p1, axis=1)
    j = np.argmax(distances)
    p2 = points[j]
    # Initial sphere: centre is midpoint of p1 and p2; radius is half the distance between them.
    center = (p1 + p2) / 2.0
    radius = np.linalg.norm(p2 - center)
    # Enlarge sphere to enclose all points.
    for p in points:
        d = np.linalg.norm(p - center)
        if d > radius:
            new_radius = (radius + d) / 2.0
            center = center + (p - center) * ((new_radius - radius) / d)
            radius = new_radius
    return center, radius

blobs = {}
csv_data = []

for label in unique_labels:
    blob_points = deformation_points[labels == label]
    centre, radius = minimal_bounding_sphere(blob_points)
    if centre is None:
        continue
        
    # Compute the residual error (the difference between the actual distances and the sphere's radius)
    distances_blob = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances_blob - radius)
    deformation_intensity = np.mean(residuals)
    
    # Assign colours based on the fitted sphere's radius (a proxy for error magnitude)
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
        "deformation_intensity": deformation_intensity
    }
    
    csv_data.append({
        'x': centre[0],
        'y': centre[1],
        'z': centre[2],
        'radius': radius,
        'deformation_intensity': deformation_intensity
    })

# Write the sphere data to a CSV file.
csv_filename = "sphere_data_detection.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)
print(f"Sphere data written to {csv_filename}")

# ---------------------------
# Step 6: Visualise the Original Model and Overlaid Spheres
# ---------------------------
spheres = []
for label, sphere_params in blobs.items():
    centre = sphere_params["centre"]
    radius = sphere_params["radius"]
    colour = sphere_params["colour"]
    
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere_mesh.translate(centre)
    sphere_mesh.paint_uniform_color(colour)
    spheres.append(sphere_mesh)

geometries = [mesh_original, mesh_deformed] + spheres
o3d.visualization.draw_geometries(geometries)
