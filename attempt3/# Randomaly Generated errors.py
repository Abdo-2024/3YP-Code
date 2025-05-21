# Randomaly Generated errors
import trimesh
import numpy as np
import random

# Function for deforming the Spheres
def deform_sphere(sphere, num_blobs=1, blob_radius=0.05, blob_intensity=0.01):
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

# Step 1: Define the number of spheres you wish to create.
num_spheres = 25  # Adjust this number as needed.

# Prepare a list to hold the deformed spheres.
meshes = []

# Step 2: For each sphere, create and deform it.
for i in range(num_spheres):
    # Step 2.1: Define a random radius for the sphere.
    radius = random.uniform(0.005, 0.015)
    
    # Step 2.2: Create an icosphere with the specified radius.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # Step 2.3: Deform the sphere to create a "blob" effect.
    random_intensity_factor = random.uniform(0.01, 0.5)
    deformation_intensity = random_intensity_factor * radius
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=20, 
                                    blob_radius=radius * 1.2, 
                                    blob_intensity=deformation_intensity)
    
    # Add the deformed (blob-like) sphere to our list of meshes.
    meshes.append(deformed_sphere)

# Step 3: Export all the deformed spheres as a single STL file.
combined_sphere_mesh = trimesh.util.concatenate(meshes)
combined_sphere_mesh.export('deformed_spheres.stl')
