import open3d as o3d
import numpy as np
import csv
from scipy.spatial import cKDTree
import hdbscan  # pip install hdbscan
"""
3D Model Deformation Analysis and Visualisation with Open3D, HDBSCAN, and Minimal Bounding Spheres

This script performs deformation analysis on a pair of 3D models using Open3D and advanced clustering techniques:
1. Loads original and deformed STL meshes using Open3D.
2. Aligns the deformed model to the original using Iterative Closest Point (ICP) registration.
3. Extracts deformation points based on an adaptive nearest-neighbour search.
4. Clusters deformation points using HDBSCAN to identify distinct deformation regions.
5. Fits minimal bounding spheres to each cluster, calculating deformation intensity.
6. Visualises the original model and overlaid spheres representing deformation using Open3D.

Key Functionalities:
- Load and sample points from STL meshes for comparative analysis.
- Register meshes using ICP to align deformed points with original points.
- Identify deformation points using an adaptive distance threshold.
- Cluster deformation points into distinct blobs using HDBSCAN.
- Fit minimal bounding spheres to quantify and visualise deformation intensity.
- Export sphere data (centre, radius, deformation intensity) to a CSV file.
- Visualise the original model and spheres representing deformation using Open3D.

User-defined Parameters:
- n_points: Number of points sampled from each mesh.
- threshold_icp: Threshold for ICP alignment distance.
- threshold_low, threshold_high: Colour thresholds for sphere radius to indicate deformation magnitude.

Dependencies:
- Open3D (imported as o3d): For 3D data processing, visualization, and registration.
- NumPy (imported as np): For numerical operations and array handling.
- scipy.spatial.cKDTree: For efficient nearest-neighbour search.
- hdbscan: For density-based clustering of deformation points.
- csv: For writing sphere data to a CSV file.

Usage:
1. Replace the file paths ("2x2_MN_Array_scaled.stl", "2x2_MN_Array_scaled+Noise.stl") with the actual paths to your STL files.
2. Adjust user-defined parameters (n_points, thresholds) based on your model and deformation characteristics.
3. Run the script to load, align, analyse, and visualise the deformation in the 3D model.
   It will generate overlaid spheres on the deformed model, where each sphere represents a cluster of deformation points,
   coloured according to the magnitude of deformation.

Note:
- Ensure Open3D and necessary dependencies are installed (`pip install open3d numpy scipy hdbscan`).
- The script assumes valid STL models with surface details for point sampling and deformation analysis.
- Adjust thresholds and parameters to suit your specific model and deformation analysis requirements.
"""

# ---------------------------
# User‐defined parameters – these may now be more data‐driven
# ---------------------------
n_points = 50000
threshold_icp = 0.0002  # ICP alignment distance threshold

# For colouring the spheres (optional)
threshold_low = 0.005   # e.g. yellow if below this radius
threshold_high = 0.015  # e.g. red if above this radius

# ---------------------------
# Step 1: Load and Sample Points
# ---------------------------
mesh_original = o3d.io.read_triangle_mesh("2x2_MN_Array_scaled.stl")
mesh_deformed = o3d.io.read_triangle_mesh("2x2_MN_Array_scaled+Noise.stl")
mesh_original.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

pcd_original = mesh_original.sample_points_poisson_disk(n_points)
pcd_deformed = mesh_deformed.sample_points_poisson_disk(n_points)

# ---------------------------
# Step 2: Align the Two Models using ICP
# ---------------------------
trans_init = np.eye(4)
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_deformed, pcd_original, threshold_icp, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("ICP Transformation:\n", reg_p2p.transformation)
pcd_deformed.transform(reg_p2p.transformation)

# ---------------------------
# Step 3: Extract Deformation Points using a Vectorised Nearest-Neighbour Search
# ---------------------------
original_points = np.asarray(pcd_original.points)
deformed_points = np.asarray(pcd_deformed.points)

tree = cKDTree(original_points)
distances, _ = tree.query(deformed_points, k=1)

# Compute an adaptive deformation threshold using the mean and standard deviation
mean_dist = np.mean(distances)
std_dist = np.std(distances)
adaptive_threshold = mean_dist + 2 * std_dist  # adjust multiplier as needed

mask = distances > adaptive_threshold
deformation_points = deformed_points[mask]

print("Number of deformation points:", len(deformation_points))
if len(deformation_points) == 0:
    print("No deformation points detected – try adjusting the adaptive threshold parameters.")
    # Optionally exit

# Optional: Visualise deformation points for debugging
deformation_pcd = o3d.geometry.PointCloud()
deformation_pcd.points = o3d.utility.Vector3dVector(deformation_points)
o3d.visualization.draw_geometries([deformation_pcd])

# ---------------------------
# Step 4: Cluster the Deformation Points using HDBSCAN (Automatic clustering)
# ---------------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(deformation_points)
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)
print("Detected blobs (clusters):", len(unique_labels))
if len(unique_labels) == 0:
    print("No clusters detected – consider adjusting the min_cluster_size parameter.")

# ---------------------------
# Step 5: Fit a Sphere to Each Blob using an Optimised Minimal Bounding Sphere Algorithm
# and Compute Deformation Intensity
# ---------------------------
def minimal_bounding_sphere(points):
    points = np.array(points)
    if len(points) == 0:
        return None, None
    p0 = points[0]
    distances = np.linalg.norm(points - p0, axis=1)
    p1 = points[np.argmax(distances)]
    distances = np.linalg.norm(points - p1, axis=1)
    p2 = points[np.argmax(distances)]
    centre = (p1 + p2) / 2.0
    radius = np.linalg.norm(p2 - centre)
    while True:
        dists = np.linalg.norm(points - centre, axis=1)
        max_dist = dists.max()
        if max_dist <= radius:
            break
        p = points[np.argmax(dists)]
        new_radius = (radius + max_dist) / 2.0
        centre = centre + (p - centre) * ((new_radius - radius) / max_dist)
        radius = new_radius
    return centre, radius

blobs = {}
csv_data = []
for label in unique_labels:
    blob_points = deformation_points[labels == label]
    centre, radius = minimal_bounding_sphere(blob_points)
    if centre is None:
        continue
    distances_blob = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances_blob - radius)
    deformation_intensity = np.mean(residuals)
    if radius > threshold_high:
        colour = [0.5, 0, 0.5]
    elif radius > (threshold_low + threshold_high) / 2:
        colour = [1, 0, 0]
    elif radius > threshold_low:
        colour = [1, 0.65, 0]
    else:
        colour = [1, 1, 0]
    blobs[label] = {"centre": centre, "radius": radius, "colour": colour,
                    "deformation_intensity": deformation_intensity}
    print(f"Blob {label}: Centre = {centre}, Radius = {radius:.6f}, Deformation Intensity = {deformation_intensity:.6f}, Colour = {colour}")
    csv_data.append({'x': centre[0], 'y': centre[1], 'z': centre[2],
                     'radius': radius, 'deformation_intensity': deformation_intensity})

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

geometries = [mesh_original, pcd_deformed] + spheres
o3d.visualization.draw_geometries(geometries)
