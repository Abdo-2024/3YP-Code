#!/usr/bin/env python3
"""
This script generates noisy versions of a base STL model by adding defect spheres,
runs an error‐detection algorithm 50 times per noise level, and finally produces a
bar plot comparing the actual number of noise spheres to the average number of detected errors.
"""
import os
import shutil
import random
import csv
import numpy as np
import trimesh
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ---------------------------
# PARAMETERS (adjust as needed)
# ---------------------------
N_POINTS = 50000                     # Number of points to sample from each mesh
THRESHOLD_ICP = 0.0002               # ICP alignment distance threshold
DBSCAN_EPS = 0.01                    # DBSCAN eps parameter
DBSCAN_MIN_SAMPLES = 4               # DBSCAN minimum samples

# File paths – adjust these to your system:
ORIGINAL_STL = "/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl"  # Original STL file
OUTPUT_FOLDER = "output_stls"                      # Folder to store generated noisy STLs
RESULT_PLOT = "error_detection_barplot.png"        # Name of the output bar plot image

# Noise levels (i.e. number of defect spheres to add)
NOISE_LEVELS = [0, 10, 20, 40, 80, 160, 320, 640]

# Number of detection runs per STL
N_RUNS = 50

# Make sure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# NOISE GENERATION FUNCTIONS
# ---------------------------
def deform_sphere(sphere, num_blobs=15, blob_radius=0.05, blob_intensity=0.01):
    """
    Deform the sphere to create a 'blob' effect by displacing vertices
    based on random blob centres.
    
    Parameters:
      sphere (trimesh.Trimesh): The sphere mesh to deform.
      num_blobs (int): Number of random blobs used for deformation.
      blob_radius (float): The sphere of influence for each blob.
      blob_intensity (float): Maximum displacement at the centre of a blob.
    
    Returns:
      trimesh.Trimesh: The deformed sphere mesh.
    """
    vertices = sphere.vertices.copy()
    # Determine the bounding box of the sphere
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    # Generate random blob centres within the bounding box
    blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    # For each vertex, compute its displacement from every blob
    for i, vertex in enumerate(vertices):
        total_disp = np.zeros(3)
        for center in blob_centers:
            direction = vertex - center
            distance = np.linalg.norm(direction)
            if distance < blob_radius:
                factor = 1 - (distance / blob_radius)
                unit_dir = direction / distance if distance != 0 else np.zeros(3)
                total_disp += blob_intensity * factor * unit_dir
        vertices[i] = vertex + total_disp
    
    sphere.vertices = vertices
    return sphere

def generate_noisy_stl(num_spheres, output_stl_path):
    """
    Generate a new noisy STL file by adding a given number of defect (noise) spheres.
    The defects are added onto the main mesh (loaded from ORIGINAL_STL) and then the combined
    mesh is exported.
    
    Parameters:
      num_spheres (int): Number of defect spheres to add.
      output_stl_path (str): Path to save the generated noisy STL.
    """
    # Load the main mesh using trimesh
    main_mesh = trimesh.load(ORIGINAL_STL)
    
    # Get the bounding box of the main mesh
    min_coords, max_coords = main_mesh.bounds
    # Define a margin so that we do not attach spheres too close to the bottom
    margin = (max_coords[2] - min_coords[2]) * 0.05

    # Prepare a list to hold meshes (starting with the main mesh)
    meshes = [main_mesh]
    
    # For each defect sphere
    for i in range(num_spheres):
        valid_sample = False
        while not valid_sample:
            # Sample a random point on the surface of the main mesh
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            # Only accept if the point is above the margin
            if point[2] > min_coords[2] + margin:
                valid_sample = True
        
        # Retrieve the normal at the sampled face
        normal = main_mesh.face_normals[face_index]
        # Choose a random radius for the sphere
        radius = random.uniform(0.005, 0.015)
        # Create an icosphere
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        # Decide whether the sphere is attached by its centre or surface
        if random.choice([True, False]):
            sphere_center = point + normal * radius  # Surface-attached
        else:
            sphere_center = point                     # Centre-attached
        sphere.apply_translation(sphere_center)
        
        # Randomly choose a deformation intensity (scaled by the radius)
        deformation_intensity = random.uniform(0.01, 0.5) * radius
        # Deform the sphere to create a blob-like defect
        deformed_sphere = deform_sphere(sphere, num_blobs=30, blob_radius=radius * 1.2, blob_intensity=deformation_intensity)
        # Add the deformed sphere to the list
        meshes.append(deformed_sphere)
    
    # Combine the main mesh and all defect spheres
    combined_mesh = trimesh.util.concatenate(meshes)
    # Export the combined mesh as an STL file
    combined_mesh.export(output_stl_path)
    print(f"Exported noisy STL with {num_spheres} defect spheres to {output_stl_path}")

# ---------------------------
# ERROR DETECTION FUNCTIONS
# ---------------------------
def minimal_bounding_sphere(points):
    """
    Compute an approximate minimal sphere that encloses all points using Ritter's algorithm.
    
    Parameters:
      points (np.ndarray): Array of shape (N,3) containing 3D points.
      
    Returns:
      (centre, radius): Tuple with the sphere centre (np.ndarray) and radius (float).
    """
    points = np.array(points)
    if len(points) == 0:
        return None, None

    p0 = points[0]
    distances = np.linalg.norm(points - p0, axis=1)
    i = np.argmax(distances)
    p1 = points[i]
    distances = np.linalg.norm(points - p1, axis=1)
    j = np.argmax(distances)
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

def merge_overlapping_spheres_points(spheres):
    """
    Merge any number of overlapping spheres based on their associated deformation points.
    For each group of overlapping spheres (determined via union-find), we combine all
    their deformation points and compute the minimal bounding sphere over the union.
    
    Each sphere in the input list is a dictionary with at least the keys:
      'centre': the sphere's centre,
      'radius': the sphere's radius,
      'points': the deformation points (np.ndarray) used to fit the sphere,
      'deformation_intensity': (float) the mean error.
    
    Returns:
      merged_spheres (list): A list of merged sphere dictionaries.
    """
    n = len(spheres)
    parent = list(range(n))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    
    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    # Consider spheres as overlapping if the distance between centres is <= sum of radii.
    for i in range(n):
        for j in range(i+1, n):
            c1, r1 = spheres[i]['centre'], spheres[i]['radius']
            c2, r2 = spheres[j]['centre'], spheres[j]['radius']
            if np.linalg.norm(c1 - c2) <= (r1 + r2):
                union(i, j)
    
    # Group spheres by connected component
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(spheres[i])
    
    merged_spheres = []
    for group in groups.values():
        # Combine all deformation points from the group
        all_points = np.vstack([s['points'] for s in group])
        new_centre, new_radius = minimal_bounding_sphere(all_points)
        # Recalculate deformation intensity from residuals (using the new sphere)
        distances = np.linalg.norm(all_points - new_centre, axis=1)
        residuals = np.abs(distances - new_radius)
        new_intensity = np.mean(residuals)
        # Re-assign a colour (this is arbitrary and may be adjusted)
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
            "deformation_intensity": new_intensity,
            "colour": new_colour,
            "points": all_points  # storing for completeness
        })
    
    return merged_spheres

def detect_errors(deformed_stl_path, original_stl_path=ORIGINAL_STL):
    """
    Run the error detection algorithm on a deformed STL file.
    This function loads the original and deformed models, performs ICP alignment,
    extracts deformation points, clusters them using DBSCAN (with adaptive threshold tuning),
    fits minimal bounding spheres to each cluster, merges overlapping spheres (using the deformation points),
    and returns the number of merged defect spheres.
    
    Parameters:
      deformed_stl_path (str): Path to the noisy (deformed) STL.
      original_stl_path (str): Path to the original STL.
      
    Returns:
      detected_count (int): Number of detected defect spheres after merging.
    """
    # Load the original and deformed meshes via Open3D
    mesh_original = o3d.io.read_triangle_mesh(original_stl_path)
    mesh_deformed = o3d.io.read_triangle_mesh(deformed_stl_path)
    mesh_original.compute_vertex_normals()
    mesh_deformed.compute_vertex_normals()
    
    # Sample points from both meshes
    pcd_original = mesh_original.sample_points_poisson_disk(N_POINTS)
    pcd_deformed = mesh_deformed.sample_points_poisson_disk(N_POINTS)
    
    # ---------------------------
    # ICP Alignment (Step 2)
    # ---------------------------
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_deformed, pcd_original, THRESHOLD_ICP, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    pcd_deformed.transform(reg_p2p.transformation)
    
    # ---------------------------
    # Extract deformation points (Step 3)
    # ---------------------------
    original_points = np.asarray(pcd_original.points)
    deformed_points = np.asarray(pcd_deformed.points)
    tree = cKDTree(original_points)
    distances, _ = tree.query(deformed_points, k=1)
    
    # Adaptive threshold tuning using silhouette score
    best_threshold = None
    best_silhouette = -1.0
    for candidate_threshold in np.arange(0.005, 0.007, 0.0001):
        mask = distances > candidate_threshold
        candidate_pts = deformed_points[mask]
        if candidate_pts.shape[0] == 0:
            continue
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        candidate_labels = dbscan.fit_predict(candidate_pts)
        if len(set(candidate_labels)) < 2:
            continue
        try:
            score = silhouette_score(candidate_pts, candidate_labels)
        except Exception:
            continue
        if score > best_silhouette:
            best_silhouette = score
            best_threshold = candidate_threshold
    if best_threshold is None:
        best_threshold = 0.005
    mask = distances > best_threshold
    deformation_points = deformed_points[mask]
    
    # ---------------------------
    # DBSCAN Clustering (Step 4)
    # ---------------------------
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = dbscan.fit_predict(deformation_points)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) == 0:
        return 0  # No clusters detected
    
    # ---------------------------
    # Fit spheres to each cluster (Step 5)
    # ---------------------------
    blobs = []  # Each blob will be a dictionary that also stores the points
    for label in unique_labels:
        blob_pts = deformation_points[labels == label]
        centre, radius = minimal_bounding_sphere(blob_pts)
        if centre is None:
            continue
        distances_blob = np.linalg.norm(blob_pts - centre, axis=1)
        residuals = np.abs(distances_blob - radius)
        deformation_intensity = np.mean(residuals)
        # Choose a colour based on radius (adjust thresholds as needed)
        if radius > 0.01:
            colour = [0.5, 0, 0.5]
        elif radius > 0.005:
            colour = [1, 0, 0]
        elif radius > 0.001:
            colour = [1, 0.65, 0]
        else:
            colour = [1, 1, 0]
        
        # Save the blob and also store its associated points
        blobs.append({
            "centre": centre,
            "radius": radius,
            "deformation_intensity": deformation_intensity,
            "colour": colour,
            "points": blob_pts
        })
    
    # ---------------------------
    # Merge overlapping spheres (using deformation points)
    # ---------------------------
    merged_spheres = merge_overlapping_spheres_points(blobs)
    detected_count = len(merged_spheres)
    return detected_count

# ---------------------------
# MAIN EXPERIMENTAL LOOP
# ---------------------------
def main():
    # Dictionary to hold detection results per noise level
    results = {}  # key: noise level, value: list of detected error counts (one per run)
    
    # Loop over each noise level
    for noise in NOISE_LEVELS:
        print(f"\nProcessing noise level: {noise} defect spheres")
        stl_filename = os.path.join(OUTPUT_FOLDER, f"noisy_{noise}.stl")
        # Generate the noisy STL (if noise == 0, simply copy the original)
        if noise == 0:
            # Use shutil.copy to copy the original STL to the output folder
            shutil.copy(ORIGINAL_STL, stl_filename)
            print(f"Copied original STL to {stl_filename}")
        else:
            generate_noisy_stl(noise, stl_filename)
        
        detected_counts = []
        for run in range(N_RUNS):
            count = detect_errors(stl_filename)
            detected_counts.append(count)
            print(f"  Run {run+1}/{N_RUNS}: detected {count} defects")
        results[noise] = detected_counts

    # ---------------------------
    # Produce a bar plot with error bars
    # ---------------------------
    noise_levels = sorted(results.keys())
    avg_detected = [np.mean(results[n]) for n in noise_levels]
    std_detected = [np.std(results[n]) for n in noise_levels]
    
    plt.figure(figsize=(10, 6))
    plt.bar([str(n) for n in noise_levels], avg_detected, yerr=std_detected, capsize=5, color='skyblue')
    plt.xlabel("Actual Number of Defect Spheres")
    plt.ylabel("Average Detected Defects (± standard deviation)")
    plt.title("Error Detection Performance Across Noise Levels")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(RESULT_PLOT)
    plt.close()
    print(f"\nBar plot saved as {RESULT_PLOT}")

if __name__ == "__main__":
    main()
