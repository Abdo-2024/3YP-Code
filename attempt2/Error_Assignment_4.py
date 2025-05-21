"""
Deformation Detection and Analysis Between 3D Meshes Using Open3D, KDTree, and DBSCAN

This script compares a perfect and a deformed 3D mesh to detect areas of significant 
deformation. It computes noise vectors, colours the perfect mesh by noise magnitude, 
clusters deformation points using DBSCAN, approximates each cluster with spheres, 
and visualises the results.

Key Functionalities:
- Load and preprocess meshes (perfect and deformed)
- Compute noise vectors between corresponding vertices
- Colour the perfect mesh by noise magnitude
- Cluster deformation points into blobs using DBSCAN
- Approximate each blob with a sphere and colour it by its size
- Visualise the perfect mesh with overlaid spheres representing deformation blobs

Dependencies:
- Open3D
- NumPy
- SciPy (KDTree)
- Scikit-learn (DBSCAN)
- Matplotlib (optional, for colormap)

Usage:
Ensure paths to the `.stl` files are correctly set and parameters adjusted in `parameters.py`.
"""

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import parameters  # Import parameters from centralized file

# Step 1: Load meshes and compute noise

# File paths (update as needed)
perfect_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl"
deformed_path = "/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed2.stl"

mesh_perfect = o3d.io.read_triangle_mesh(perfect_path)
mesh_deformed = o3d.io.read_triangle_mesh(deformed_path)

mesh_perfect.compute_vertex_normals()
mesh_deformed.compute_vertex_normals()

# Extract vertex arrays as NumPy arrays
pts_perfect = np.asarray(mesh_perfect.vertices)
pts_deformed = np.asarray(mesh_deformed.vertices)

# Find nearest neighbor in deformed mesh for each vertex in perfect mesh using KDTree
tree = KDTree(pts_deformed)
distances, indices = tree.query(pts_perfect)
pts_deformed_on_perfect = pts_deformed[indices]

# Compute noise vectors (displacement at each vertex)
noise_vectors = pts_deformed_on_perfect - pts_perfect

# Step 2: Color the Perfect Mesh by Noise Magnitude

# Compute noise magnitude at each vertex
noise_magnitudes = np.linalg.norm(noise_vectors, axis=1)
min_noise = noise_magnitudes.min()
max_noise = noise_magnitudes.max()

# Normalize noise values
normalized_noise = (noise_magnitudes - min_noise) / (max_noise - min_noise + 1e-8)

# Map noise magnitude to color between SMALL_COLOR and LARGE_COLOR
colors = np.zeros((noise_magnitudes.shape[0], 3))
for i, norm_val in enumerate(normalized_noise):
    colors[i] = np.array(parameters.SMALL_COLOR) * (1 - norm_val) + np.array(parameters.LARGE_COLOR) * norm_val

mesh_perfect.vertex_colors = o3d.utility.Vector3dVector(colors)

# Step 3: Cluster the "Blob" Points and Approximate with Spheres

# Use deformed positions on perfect mesh for clustering
blob_points = pts_deformed_on_perfect

# Consider only points with significant deformation using threshold from parameters.py
significant_idx = np.where(noise_magnitudes > parameters.NOISE_THRESHOLD)[0]
if significant_idx.size > 0:
    blob_points = blob_points[significant_idx]
else:
    print("No points exceed the noise threshold; adjust the NOISE_THRESHOLD parameter as needed.")

# Cluster points into blobs using DBSCAN
clustering = DBSCAN(eps=parameters.DBSCAN_EPS, min_samples=parameters.DBSCAN_MIN_SAMPLES).fit(blob_points)
labels = clustering.labels_

# Remove noise points (label -1) from consideration
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)

spheres = []
cluster_info = []

# For each cluster, compute centroid and estimate radius
for label in unique_labels:
    cluster_pts = blob_points[labels == label]
    if cluster_pts.shape[0] == 0:
        continue
    centroid = cluster_pts.mean(axis=0)
    distances_cluster = np.linalg.norm(cluster_pts - centroid, axis=1)
    radius = distances_cluster.max()
    cluster_info.append((centroid, radius))

# Determine min and max radii for colour mapping
if cluster_info:
    radii = [r for (_, r) in cluster_info]
    min_radius = min(radii)
    max_radius = max(radii)
else:
    min_radius = 0
    max_radius = 1

# Create a sphere mesh for each blob and colour based on radius
for centroid, radius in cluster_info:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.translate(centroid)  # Place sphere at centroid of cluster
    norm_r = (radius - min_radius) / (max_radius - min_radius + 1e-8)
    sphere_color = np.array(parameters.SMALL_COLOR) * (1 - norm_r) + np.array(parameters.LARGE_COLOR) * norm_r
    sphere.paint_uniform_color(sphere_color.tolist())
    sphere.compute_vertex_normals()
    spheres.append(sphere)

# Step 4: Visualize the Results

# Visualize perfect mesh with overlaid spheres representing deformation blobs
o3d.visualization.draw_geometries([mesh_perfect] + spheres)

