import trimesh
import numpy as np
import random

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
    # Remove or comment out the rezero() call so the translation is preserved:
    # sphere.rezero()  # <-- This line has been removed.

    return sphere

# -----------------------------
# Main Code
# -----------------------------

# Step 1: Load the main STL file (the model you do not want to deform).
main_mesh = trimesh.load('/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl')

# Prepare a list to hold all meshes. The list will include the main mesh and the spheres.
meshes = [main_mesh]

# Step 2: Define the number of spheres you wish to add.
num_spheres = 50  # Adjust this number as needed.

# Step 3: For each sphere, sample a random point on the surface of the main mesh.
for i in range(num_spheres):
    # 3.1: Sample a random point on the main mesh's surface.
    points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
    point = points[0]
    face_index = face_indices[0]
    
    # 3.2: Retrieve the normal at the sampled point.
    normal = main_mesh.face_normals[face_index]
    
    # 3.3: Define a random (small) radius for the sphere.
    radius = random.uniform(0.005, 0.01)
    
    # 3.4: Create an icosphere with the specified radius.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # 3.5: Translate the sphere so that it touches the main mesh.
    # The sphere's centre is positioned at the sampled point plus an offset (its radius in the direction of the normal).
    sphere_center = point + normal * radius
    sphere.apply_translation(sphere_center)
    
    # Step 4: Deform the sphere to create a "blob" effect.
    # The deformation parameters are adjusted relative to the sphere's size.
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=15, 
                                    blob_radius=radius * 1.2, 
                                    blob_intensity=0.1 * radius)
    
    # Add the deformed (blob-like) sphere to our list of meshes.
    meshes.append(deformed_sphere)

# Step 5: Combine the main mesh and all deformed spheres into a single mesh.
combined_mesh = trimesh.util.concatenate(meshes)

# Step 6: Export the combined mesh to a new STL file.
combined_mesh.export('combined_deformed.stl')
