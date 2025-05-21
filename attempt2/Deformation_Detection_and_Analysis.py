"""
Deformation Detection and Analysis Between 3D Meshes Using Open3D and DBSCAN

This script compares an original and a deformed 3D mesh to detect areas of significant 
deformation. It performs point sampling, ICP alignment, spatial subtraction to identify 
noise (deformation) points, clusters them using DBSCAN, fits minimal bounding spheres 
to each cluster, calculates deformation intensity, and visualises the results with 
colour-coded spheres.

Key Functionalities:
- ICP-based alignment of deformed mesh to original
- Deformation point extraction by nearest-neighbour comparison
- DBSCAN clustering to identify localised deformation regions (blobs)
- Minimal bounding sphere fitting per blob (Ritter's algorithm)
- Colour-coded visualisation of deformation severity
- CSV export of deformation blob metrics (centre, radius, intensity)

Dependencies:
- Open3D
- NumPy
- Scikit-learn (DBSCAN)
- CSV (for output)

Usage:
Ensure paths to the `.stl` files are correctly set and parameters adjusted for your mesh scale.
"""

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import csv

# ---------------------------
# User-defined parameters
# ---------------------------
n_points = 90000
threshold_icp = 0.0002  # Threshold for ICP registration convergence
deformation_threshold = 0.001  # Minimum distance to consider a point deformed
dbscan_eps = 0.002  # DBSCAN: max distance between points to be considered in a cluster
dbscan_min_samples = 5  # DBSCAN: minimum number of points to form a cluster
threshold_low = 0.001  # Radius threshold for colouring small blobs (yellow)
threshold_high = 0.005  # Radius threshold for colouring large blobs (red)

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
mesh_original = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_scaled_dragon.stl")
mesh_deformed = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed_dragon.stl")

# Compute vertex normals for alignment and rendering
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# Sample points using Poisson disk sampling
pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

# ---------------------------
# Step 2: Align the Deformed Model to the Original Using ICP
# ---------------------------
trans_init = np.eye(4)  # Identity transformation
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_deformed, pcd_original, threshold_icp, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP Transformation:\n", reg_p2p.transformation)
pcd_deformed.transform(reg_p2p.transformation)

# ---------------------------
# Step 3: Extract Deformation Points by Comparing Distances
# ---------------------------
original_kd_tree = o3d.geometry.KDTreeFlann(pcd_original)
deformation_points = []

# For each point in the deformed mesh, find the nearest in the original
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
    # import sys; sys.exit()

# ---------------------------
# Step 4: Cluster Deformation Points Using DBSCAN
# ---------------------------
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
labels = dbscan.fit_predict(deformation_points)
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)  # Remove noise points
print("Detected blobs (clusters):", len(unique_labels))

# Visualise if no clusters found
if len(unique_labels) == 0:
    print("No clusters detected. Consider adjusting DBSCAN parameters.")
    deformation_pcd = o3d.geometry.PointCloud()
    deformation_pcd.points = o3d.utility.Vector3dVector(deformation_points)
    o3d.visualization.draw_geometries([deformation_pcd])
    # Optionally exit:
    # import sys; sys.exit()

# ---------------------------
# Step 5: Fit Spheres to Each Blob and Calculate Deformation Intensity
# ---------------------------
def minimal_bounding_sphere(points):
    """
    Estimate the smallest sphere that contains all given points using Ritter’s algorithm.
    """
    points = np.array(points)
    if len(points) == 0:
        return None, None
    p0 = points[0]
    distances = np.linalg.norm(points - p0, axis=1)
    p1 = points[np.argmax(distances)]
    distances = np.linalg.norm(points - p1, axis=1)
    p2 = points[np.argmax(distances)]
    center = (p1 + p2) / 2.0
    radius = np.linalg.norm(p2 - center)
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

    # Assign a colour based on sphere size (proxy for deformation severity)
    if radius > 0.01:
        colour = [0.5, 0, 0.5]    # Purple
    elif radius > 0.005:
        colour = [1, 0, 0]        # Red
    elif radius > 0.001:
        colour = [1, 0.65, 0]     # Orange
    else:
        colour = [1, 1, 0]        # Yellow

    blobs[label] = {
        "centre": centre,
        "radius": radius,
        "colour": colour,
        "deformation_intensity": deformation_intensity
    }

    print(f"Blob {label}: Centre = {centre}, Radius = {radius:.6f}, Deformation Intensity = {deformation_intensity:.6f}, Colour = {colour}")
    
    csv_data.append({
        'x': centre[0],
        'y': centre[1],
        'z': centre[2],
        'radius': radius,
        'deformation_intensity': deformation_intensity
    })

# ---------------------------
# Step 6: Export Blob Data to CSV
# ---------------------------
csv_filename = "sphere_data_detection.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)
print(f"Sphere data written to {csv_filename}")

# ---------------------------
# Step 7: Visualise Deformation Spheres on the Meshes
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

# Combine and display original, deformed mesh, and detected deformation spheres
geometries = [mesh_original, mesh_deformed] + spheres
o3d.visualization.draw_geometries(geometries)

