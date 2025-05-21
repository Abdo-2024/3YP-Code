"""
Deformation of a Main Mesh with Attached Deformable Spheres Using Trimesh and CSV Export

This script deforms a main 3D mesh by attaching multiple deformable spheres at random points on its surface. 
Each sphere creates a 'blob' effect by displacing vertices based on random blob centres within its sphere 
of influence. It exports the resulting deformed mesh and records sphere data to a CSV file.

Key Functionalities:
- Load a main STL mesh file (the model not to be deformed)
- Sample random points on the main mesh's surface
- Attach deformable spheres with random deformation intensities
- Deform each sphere to create a blob effect based on nearby blob centres
- Export the combined deformed mesh and sphere data to CSV and STL files

Dependencies:
- Trimesh
- NumPy
- Random
- CSV (for CSV file operations)

Usage:
Ensure paths for input/output files are correctly set, adjust parameters for sphere count and deformation factors.
"""

import trimesh
import numpy as np
import random
import csv  # Import csv module for writing CSV files

def deform_sphere(sphere, num_blobs=15, blob_radius=0.05, blob_intensity=0.01):
    """
    Deform the sphere to create a 'blob' effect by displacing vertices based on random blob centres.
    
    Parameters:
      sphere (trimesh.Trimesh): The sphere mesh to deform.
      num_blobs (int): The number of random blobs to use for deformation.
      blob_radius (float): The sphere of influence for each blob.
      blob_intensity (float): The maximum displacement applied at the centre of a blob.
    
    Returns:
      trimesh.Trimesh: The deformed sphere mesh.
    """
    # Copy the vertices of the sphere (each vertex is a point in 3D).
    vertices = sphere.vertices.copy()

    # Compute the bounding box of the sphere's vertices.
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate random blob centres within the bounding box.
    blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    # For each vertex, compute the total displacement from all blobs.
    for i, vertex in enumerate(vertices):
        total_displacement = np.zeros(3)
        for blob_center in blob_centers:
            # Calculate the vector from the blob centre to the vertex.
            direction = vertex - blob_center
            distance = np.linalg.norm(direction)
            # If the vertex is within the blob's sphere of influence, add a displacement.
            if distance < blob_radius:
                # The influence decreases linearly with distance.
                factor = 1 - (distance / blob_radius)
                # Compute the unit vector (taking care to avoid division by zero).
                if distance != 0:
                    unit_direction = direction / distance
                else:
                    unit_direction = np.zeros(3)
                total_displacement += blob_intensity * factor * unit_direction
        # Update the vertex position with the total displacement.
        vertices[i] = vertex + total_displacement

    # Update the sphere's vertices with the deformed positions.
    sphere.vertices = vertices
    # Note: We are not calling rezero() so that the sphere's translation is preserved.
    return sphere

# -----------------------------
# Main Code
# -----------------------------

# Step 1: Load the main STL file (the model you do not want to deform).
main_mesh = trimesh.load('/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_scaled_dragon.stl')

# Compute the bounding box of the main mesh.
min_coords, max_coords = main_mesh.bounds

# Define a margin above the bottom of the model. Here we use 5% of the model's height.
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare a list to hold all meshes. The list will include the main mesh and the spheres.
meshes = [main_mesh]

# Prepare a list to hold the sphere data for CSV output.
sphere_data = []  # Each element will be a dictionary with keys: x, y, z, radius, deformation_intensity

# Step 2: Define the number of spheres you wish to add.
num_spheres = 50  # Adjust this number as needed.

# Step 3: For each sphere, sample a random point on the surface of the main mesh.
for i in range(num_spheres):
    valid_sample = False
    while not valid_sample:
        # Sample a random point on the main mesh's surface.
        points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
        point = points[0]
        face_index = face_indices[0]
        # Check that the point is not at the bottom of the model.
        if point[2] > min_coords[2] + margin:
            valid_sample = True
        # Otherwise, re-sample.

    # Retrieve the normal at the sampled point.
    normal = main_mesh.face_normals[face_index]
    
    # Step 3.3: Define a random (small) radius for the sphere.
    radius = random.uniform(0.005, 0.01)
    
    # Step 3.4: Create an icosphere with the specified radius.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # Step 3.5: Decide randomly whether the sphere attaches by its centre or its surface.
    if random.choice([True, False]):
        # "Surface-attached": translate so that the sphere's surface touches the model.
        sphere_center = point + normal * radius
    else:
        # "Centre-attached": translate so that the sphere's centre is at the sampled point.
        sphere_center = point
    sphere.apply_translation(sphere_center)
    
    # Step 4: Randomly choose a deformation intensity for this sphere.
    # For example, choose a random factor between 0.01 and 0.5.
    random_intensity_factor = random.uniform(0.01, 0.5)
    deformation_intensity = random_intensity_factor * radius
    
    # Deform the sphere to create a "blob" effect.
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=30, 
                                    blob_radius=radius * 1.2, 
                                    blob_intensity=deformation_intensity)
    
    # Add the deformed (blob-like) sphere to our list of meshes.
    meshes.append(deformed_sphere)
    
    # Record the sphere's centre coordinates, radius, and deformation intensity factor.
    sphere_data.append({
        'x': sphere_center[0],
        'y': sphere_center[1],
        'z': sphere_center[2],
        'radius': radius,
        'deformation_intensity': deformation_intensity
    })

# -----------------------------
# Write the sphere data to a CSV file.
# -----------------------------
with open('sphere_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in sphere_data:
         writer.writerow(data)

# Step 5: Combine the main mesh and all deformed spheres into a single mesh.
combined_mesh = trimesh.util.concatenate(meshes)

# Step 6: Export the combined mesh to a new STL file.
combined_mesh.export('/home/a/Documents/3rd_Year/3YP/Code/attempt2/combined_deformed_dragon.stl')

