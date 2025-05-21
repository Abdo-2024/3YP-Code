"""
Mesh Deformation Analysis and Visualisation Using Open3D, DBSCAN, and CSV Export

This script loads two STL meshes (original and deformed), aligns them using ICP, extracts deformation points,
clusters them into blobs using DBSCAN, computes bounding spheres for each blob, estimates deformation intensity,
and visualises the results. It exports deformation data to a CSV file for further analysis.

Dependencies:
- Open3D
- NumPy
- scikit-learn (for DBSCAN)
- CSV (for CSV file operations)

Usage:
Ensure paths for input/output files are correctly set. Adjust parameters such as point sampling, ICP threshold,
DBSCAN parameters, and thresholds for blob size evaluation as needed.
"""

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import csv

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
# Load the original and deformed STL meshes.
mesh_original = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl")
mesh_deformed = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed2.stl")

# Ensure the meshes have vertex normals (for better ICP registration and visualisation)
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# Sample points from the surface of each mesh.
n_points = 50000
pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

# ---------------------------
# Step 2: Align the Two Models using ICP
# ---------------------------
threshold_icp = 0.002  # distance threshold for ICP
trans_init = np.eye(4)  # initial transformation (identity)

reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_deformed, pcd_original, threshold_icp, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP Transformation:\n", reg_p2p.transformation)

# Apply the transformation to the deformed point cloud.
pcd_deformed.transform(reg_p2p.transformation)

# ---------------------------
# Step 3: Subtract the Original from the Deformed to Extract Noise Points
# ---------------------------
original_kd_tree = o3d.geometry.KDTreeFlann(pcd_original)
deformation_points = []

# Define a threshold distance above which the point is considered deformed.
deformation_threshold = 0.001  # Adjust as necessary

for point in np.asarray(pcd_deformed.points):
    [k, idx, _] = original_kd_tree.search_knn_vector_3d(point, 1)
    nearest_point = np.asarray(pcd_original.points)[idx[0]]
    distance = np.linalg.norm(point - nearest_point)
    if distance > deformation_threshold:
        deformation_points.append(point)

deformation_points = np.array(deformation_points)
print("Number of deformation points:", len(deformation_points))

if len(deformation_points) == 0:
    print("No deformation points detected. Please adjust the deformation threshold or check your meshes.")
    # Optionally, exit the programme:
    # import sys; sys.exit()

# ---------------------------
# Step 4: Cluster the Deformation Points into Blobs using DBSCAN
# ---------------------------
# Tuning DBSCAN parameters to avoid merging distinct blobs:
dbscan = DBSCAN(eps=0.005, min_samples=3)
labels = dbscan.fit_predict(deformation_points)

# Identify unique clusters (excluding noise, labelled as -1).
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)

print("Detected blobs (clusters):", len(unique_labels))

if len(unique_labels) == 0:
    print("No clusters detected. Visualising the deformation points for parameter adjustment.")
    deformation_pcd = o3d.geometry.PointCloud()
    deformation_pcd.points = o3d.utility.Vector3dVector(deformation_points)
    o3d.visualization.draw_geometries([deformation_pcd])
    # Optionally, exit if clustering is critical:
    # import sys; sys.exit()

# ---------------------------
# Step 5: Estimate a Bounding Sphere for Each Blob and Compute Deformation Intensity
# ---------------------------
def bounding_sphere(points):
    """
    Compute a sphere that encapsulates all points.
    Centre is the centroid and radius is the maximum distance from the centre.
    """
    centre = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - centre, axis=1))
    return centre, radius

blobs = {}
# Adjust these thresholds based on your desired size (in the same units as your meshes)
threshold_low = 0.005    # below this, sphere is considered too small (yellow)
threshold_high = 0.02    # above this, sphere is considered too large (red)

# Prepare a list for CSV output.
csv_data = []

for label in unique_labels:
    blob_points = deformation_points[labels == label]
    
    # Use the bounding sphere to encapsulate the blob.
    centre, radius = bounding_sphere(blob_points)
    
    # Compute deformation intensity as the mean absolute difference between the distance of each blob point from the centre and the radius.
    distances = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances - radius)
    deformation_intensity = np.mean(residuals)
    
    # Colour the sphere based on its radius.
    if radius > threshold_high:
        colour = [1, 0, 0]   # Red
    elif radius < threshold_low:
        colour = [1, 1, 0]   # Yellow
    else:
        colour = [1, 0.65, 0]  # Orange
    
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
    
    # Create a sphere mesh for visualisation that encapsulates the blob.
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere_mesh.translate(centre)
    sphere_mesh.paint_uniform_color(colour)
    spheres.append(sphere_mesh)

# Prepare visualisation objects.
geometries = [mesh_original, mesh_deformed] + spheres
o3d.visualization.draw_geometries(geometries)

