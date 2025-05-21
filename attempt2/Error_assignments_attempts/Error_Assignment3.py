import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt  # for colormap if needed

# ========================================
# STEP 1: Load meshes and compute noise
# ========================================

# File paths (update as needed)
perfect_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl"
deformed_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed2.stl"

mesh_perfect = o3d.io.read_triangle_mesh(perfect_path)
mesh_deformed = o3d.io.read_triangle_mesh(deformed_path)

mesh_perfect.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# For a good correspondence, you might want to align the meshes (e.g., using ICP) 
# if they aren’t already in the same coordinate system.
# (See previous examples if needed.)

# Extract vertex arrays
pts_perfect = np.asarray(mesh_perfect.vertices)
pts_deformed  = np.asarray(mesh_deformed.vertices)

# Find for each vertex in the perfect mesh its nearest neighbor in the deformed mesh
tree = KDTree(pts_deformed)
distances, indices = tree.query(pts_perfect)
pts_deformed_on_perfect = pts_deformed[indices]

# Compute noise vectors (defect/displacement at each vertex)
noise_vectors = pts_deformed_on_perfect - pts_perfect

# ========================================
# STEP 2: Color the Perfect Mesh by Noise Magnitude
# ========================================

# Compute noise magnitude at each vertex
noise_magnitudes = np.linalg.norm(noise_vectors, axis=1)
min_noise = noise_magnitudes.min()
max_noise = noise_magnitudes.max()

# Map noise magnitude to a color:
#   - small noise: blue (0,0,1)
#   - large noise: red (1,0,0)
# (A simple linear interpolation is used.)
normalized_noise = (noise_magnitudes - min_noise) / (max_noise - min_noise + 1e-8)
# Prepare an array of colors for each vertex.
colors = np.zeros((noise_magnitudes.shape[0], 3))
colors[:, 0] = normalized_noise      # Red channel increases with noise
colors[:, 2] = 1 - normalized_noise  # Blue channel decreases with noise

mesh_perfect.vertex_colors = o3d.utility.Vector3dVector(colors)

# ========================================
# STEP 3: Cluster the "Blob" Points and Approximate with Spheres
# ========================================

# For the purpose of clustering, we use the deformed positions.
# (pts_deformed_on_perfect == pts_perfect + noise_vectors)
blob_points = pts_deformed_on_perfect

# Optionally, consider only points with a significant deformation.
# Adjust the threshold as needed.
noise_threshold = 0.002
significant_idx = np.where(noise_magnitudes > noise_threshold)[0]
if significant_idx.size > 0:
    blob_points = blob_points[significant_idx]
else:
    print("No points exceed the noise threshold; adjust threshold as needed.")

# Use DBSCAN to cluster nearby points that form a "blob".
# The eps parameter defines the maximum distance between points in a cluster.
dbscan_eps = 0.02       # Adjust this value for your data scale
dbscan_min_samples = 5 # Adjust minimum number of points per cluster
clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(blob_points)
labels = clustering.labels_

# Cluster label -1 is considered noise by DBSCAN; ignore it.
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)

spheres = []
cluster_info = []  # We'll store (centroid, radius) for each blob

for label in unique_labels:
    cluster_pts = blob_points[labels == label]
    if cluster_pts.shape[0] == 0:
        continue
    # Compute centroid of the cluster
    centroid = cluster_pts.mean(axis=0)
    # Define the sphere radius as the maximum distance from the centroid.
    distances_cluster = np.linalg.norm(cluster_pts - centroid, axis=1)
    radius = distances_cluster.max()
    cluster_info.append((centroid, radius))

# To map sphere size to color (blue for small, red for large), find min and max radius.
if cluster_info:
    radii = [r for (_, r) in cluster_info]
    min_radius = min(radii)
    max_radius = max(radii)
else:
    min_radius = 0
    max_radius = 1  # dummy values in case there are no clusters

# Create a sphere mesh for each blob and color it based on its radius.
for centroid, radius in cluster_info:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.translate(centroid)  # Position sphere at the cluster's centroid
    # Normalize the radius value
    norm_r = (radius - min_radius) / (max_radius - min_radius + 1e-8)
    # Map: norm_r = 0 -> blue (0,0,1), norm_r = 1 -> red (1,0,0)
    sphere_color = [norm_r, 0, 1 - norm_r]
    sphere.paint_uniform_color(sphere_color)
    sphere.compute_vertex_normals()
    spheres.append(sphere)

# ========================================
# STEP 4: Visualize the Results
# ========================================

# You can visualize the perfect mesh (colored by noise) together with the detected spheres.
o3d.visualization.draw_geometries([mesh_perfect] + spheres)
