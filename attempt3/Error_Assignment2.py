
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import csv
from scipy.spatial import cKDTree  # Import cKDTree for fast, vectorised nearest-neighbour queries
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
deformation_threshold = 0.002  # Increase the deformation threshold so only clearly different points are selected.
dbscan_eps = 0.01             # DBSCAN eps parameter (smaller values tend to break clusters apart)
dbscan_min_samples = 10         # DBSCAN min_samples

# For colouring the spheres, set lower and upper thresholds for sphere radius.
threshold_low = 0.005   # Lower bound for radius (yellow if below)
threshold_high = 0.015   # Upper bound for radius (red if above)

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
# Step 3: Extract Deformation Points using a Vectorised Nearest Neighbour Search
# ---------------------------
# Convert Open3D point clouds to NumPy arrays
original_points = np.asarray(pcd_original.points)
deformed_points = np.asarray(pcd_deformed.points)

# Build a fast KD‐tree using SciPy's cKDTree on the original points
tree = cKDTree(original_points)

# Query the nearest neighbour for all deformed points in one vectorised call
distances, _ = tree.query(deformed_points, k=1)

# Use a Boolean mask to select points where the distance exceeds the deformation threshold
mask = distances > deformation_threshold
deformation_points = deformed_points[mask]

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
# Step 5: Fit a Sphere to Each Blob using an Optimised Minimal Bounding Sphere Algorithm
# and Compute Deformation Intensity
# ---------------------------
def minimal_bounding_sphere(points):
    """
    Compute an approximate minimal sphere that encloses all points using Ritter's algorithm.
    Returns: (centre, radius)
    """
    points = np.array(points)
    if len(points) == 0:
        return None, None

    # Pick an arbitrary point p0
    p0 = points[0]
    # Find the point p1 farthest from p0
    distances = np.linalg.norm(points - p0, axis=1)
    p1 = points[np.argmax(distances)]
    
    # Find the point p2 farthest from p1
    distances = np.linalg.norm(points - p1, axis=1)
    p2 = points[np.argmax(distances)]
    
    # Initialise the sphere: centre is the midpoint of p1 and p2, radius is half the distance between them
    centre = (p1 + p2) / 2.0
    radius = np.linalg.norm(p2 - centre)
    
    # Enlarge the sphere to enclose all points using vectorised distance calculations
    while True:
        dists = np.linalg.norm(points - centre, axis=1)
        max_dist = dists.max()
        if max_dist <= radius:
            break  # All points are enclosed within the sphere
        # Identify the point (p) furthest from the current centre
        p = points[np.argmax(dists)]
        new_radius = (radius + max_dist) / 2.0
        # Update the centre by moving it towards p proportionally
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
        
    # Compute the residual errors (distance differences from the sphere's surface)
    distances_blob = np.linalg.norm(blob_points - centre, axis=1)
    residuals = np.abs(distances_blob - radius)
    deformation_intensity = np.mean(residuals)
    
    # Assign colours based on the sphere's radius as a proxy for error magnitude
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
    
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere_mesh.translate(centre)
    sphere_mesh.paint_uniform_color(colour)
    spheres.append(sphere_mesh)

# Visualise the original mesh, the aligned deformed point cloud, and the spheres
geometries = [mesh_original, pcd_deformed] + spheres
o3d.visualization.draw_geometries(geometries)
