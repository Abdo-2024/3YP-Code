import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import csv
"""
3D Model Deformation Analysis and Visualisation using Open3D and DBSCAN

This script performs several tasks to analyse and visualise deformation in a 3D model:
1. Loads original and deformed STL meshes using Open3D.
2. Samples points from the surfaces of both meshes for comparison.
3. Aligns the deformed model to the original using Iterative Closest Point (ICP) registration.
4. Identifies deformation points by subtracting the original from the deformed model.
5. Clusters deformation points into blobs using DBSCAN based on proximity.
6. Fits minimal bounding spheres to each blob and computes deformation intensity.
7. Assigns colours to spheres based on their radii, representing deformation magnitude.
8. Visualises the original model and overlaid spheres using Open3D for inspection.

Key Functionalities:
- Load and sample points from STL meshes for comparative analysis.
- Register meshes using ICP to align deformed points with original points.
- Identify and cluster deformation points using DBSCAN.
- Fit spheres around each cluster to quantify and visualise deformation.
- Export sphere data (centre, radius, deformation intensity) to a CSV file.
- Visualise the original model and spheres representing deformation using Open3D.

User-defined Parameters:
- n_points: Number of points sampled from each mesh.
- threshold_icp: Threshold for ICP alignment distance.
- deformation_threshold: Threshold for identifying significant deformation points.
- dbscan_eps: DBSCAN clustering distance parameter.
- dbscan_min_samples: Minimum number of samples required for a DBSCAN cluster.
- threshold_low, threshold_high: Colour thresholds for sphere radius to indicate deformation magnitude.

Dependencies:
- Open3D (imported as o3d): For 3D data processing, visualization, and registration.
- NumPy (imported as np): For numerical operations and array handling.
- sklearn.cluster.DBSCAN: For density-based clustering of deformation points.
- csv: For writing sphere data to a CSV file.

Usage:
1. Replace the file paths ("/home/a/Documents/...") with the actual paths to your STL files.
2. Adjust user-defined parameters (n_points, thresholds, DBSCAN parameters) based on your model and deformation characteristics.
3. Run the script to load, sample, align, analyse, and visualise the deformation in the 3D model.
   It will generate overlaid spheres on the original model, where each sphere represents a cluster of deformation points,
   coloured according to the magnitude of deformation.

Note:
- Ensure Open3D and necessary dependencies are installed (`pip install open3d numpy scikit-learn`).
- The script assumes valid STL models with surface details for point sampling and deformation analysis.
- Adjust thresholds and parameters to suit your specific model and deformation analysis requirements.
"""

# ---------------------------
# User‐defined parameters – adjust these based on your model scale
# ---------------------------
n_points = 50000
threshold_icp = 0.0002         # ICP alignment distance threshold
# Increase the deformation threshold so only clearly different points are selected.
deformation_threshold = 0.006 # (e.g. 0.005 rather than 0.001)
# DBSCAN clustering parameters – try reducing eps so nearby blobs are not merged.
dbscan_eps = 0.0095            # smaller value tends to break clusters apart
dbscan_min_samples = 5

# For colouring the spheres, set lower and upper thresholds for sphere radius.
# (Blobs with very small estimated spheres get yellow; very large get red.)
threshold_low = 0.005   # lower bound for radius (yellow if below)
threshold_high = 0.015   # upper bound for radius (red if above)

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
# Load the original and deformed STL meshes.
mesh_original = o3d.io.read_triangle_mesh("/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl")
mesh_deformed = o3d.io.read_triangle_mesh("/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled+Noise.stl")

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
# Step 3: Subtract the Original from the Deformed to Extract Noise (Deformation) Points
# ---------------------------
original_kd_tree = o3d.geometry.KDTreeFlann(pcd_original)
deformation_points = []

for point in np.asarray(pcd_deformed.points):
    [k, idx, _] = original_kd_tree.search_knn_vector_3d(point, 1)
    nearest_point = np.asarray(pcd_original.points)[idx[0]]
    distance = np.linalg.norm(point - nearest_point)
    if distance > deformation_threshold:
        deformation_points.append(point)

deformation_points = np.array(deformation_points)
print("Number of deformation points:", len(deformation_points))
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
    # Initial sphere: centre is midpoint of p1 and p2, radius is half their distance.
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
        
    distances = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances - radius)
    deformation_intensity = np.mean(residuals)
    
    # Assign colours based on the fitted sphere's radius (which is a proxy for the error magnitude)
    if radius > 0.01:
        colour = [0.5, 0, 0.5]    # Purple for errors > 0.01
    elif radius > 0.005:
        colour = [1, 0, 0]        # Red for errors between 0.005 and 0.01
    elif radius > 0.001:
        colour = [1, 0.65, 0]     # Orange for errors between 0.001 and 0.005
    else:
        colour = [1, 1, 0]        # Yellow for errors ~0.001 or smaller
    
    blobs[label] = {
        "centre": centre,
        "radius": radius,
        "colour": colour,
        "deformation_intensity": deformation_intensity
    }
    
    # print(f"Blob {label}: Centre = {centre}, Radius = {radius:.6f}, Deformation Intensity = {deformation_intensity:.6f}, Colour = {colour}")
    
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
