import trimesh
import numpy as np
import random
import csv

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
    return sphere

# -----------------------------
# Main Code
# -----------------------------

# Step 1: Load the main STL file (the model from which you wish to subtract spheres).
main_mesh = trimesh.load('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl')

# Attempt to repair the main mesh by filling holes and processing it.
main_mesh.fill_holes()
# The following line is deprecated but kept here for legacy reasons:
# main_mesh.remove_degenerate_faces()  
main_mesh = main_mesh.process()

# It is good practise to check if the main mesh is watertight.
if not main_mesh.is_watertight:
    print("Warning: main_mesh is not watertight. Boolean operations may fail.")

# Compute the bounding box of the main mesh.
min_coords, max_coords = main_mesh.bounds

# Define a margin above the bottom of the model (here, 5% of the model's height).
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare a list to hold sphere data for CSV output.
sphere_data = []  # Each element will be a dictionary with keys: x, y, z, radius

# Prepare a list to hold all the deformed spheres (cutouts).
cutouts = []

# Step 2: Define the number of spheres you wish to subtract.
num_spheres = 10  # Adjust this number as needed.

# Step 3: For each sphere, sample a random point on the surface of the main mesh.
for i in range(num_spheres):
    valid_sample = False
    while not valid_sample:
        # Sample a random point on the main mesh's surface.
        points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
        point = points[0]
        face_index = face_indices[0]
        # Ensure the point is not at the bottom of the model.
        if point[2] > min_coords[2] + margin:
            valid_sample = True
        # Otherwise, re-sample.

    # Retrieve the normal at the sampled point.
    normal = main_mesh.face_normals[face_index]
    
    # Step 3.3: Define a random (small) radius for the sphere.
    radius = random.uniform(0.015, 0.015)
    
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
    random_intensity_factor = random.uniform(0.0, 0.0)
    deformation_intensity = random_intensity_factor * radius
    
    # Deform the sphere to create a "blob" effect.
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=1, 
                                    blob_radius=radius * 1.2, 
                                    blob_intensity=deformation_intensity)
    
    # Reconstruct the deformed sphere to enforce a watertight (closed volume) mesh.
    deformed_sphere = trimesh.Trimesh(vertices=deformed_sphere.vertices,
                                      faces=deformed_sphere.faces,
                                      process=True)
    
    # Append the repaired sphere to the list of cutouts.
    cutouts.append(deformed_sphere)
    
    # Record the sphere's centre coordinates and radius.
    sphere_data.append({
        'x': sphere_center[0],
        'y': sphere_center[1],
        'z': sphere_center[2],
        'radius': radius,
    })

# -----------------------------
# Write the sphere data to a CSV file.
# -----------------------------
with open('sphere_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in sphere_data:
         writer.writerow(data)

# -----------------------------
# Boolean Operations: Subtracting the Spheres from the Main Mesh
# -----------------------------
# First, create a union of all the cutout spheres.

try:
    cutout_union = trimesh.boolean.union(cutouts, engine='auto')
except Exception as e:
    print("Union of cutout spheres failed with error:", e)
    # If Blender is not available, try using the OpenSCAD engine.
    cutout_union = trimesh.boolean.union(cutouts, engine='blender')

# Process the union to enforce a watertight volume.
cutout_union = trimesh.Trimesh(vertices=cutout_union.vertices,
                               faces=cutout_union.faces,
                               process=True)
if not cutout_union.is_watertight:
    print("Warning: cutout_union is not watertight even after processing.")

# Next, subtract the union of spheres from the main mesh.
try:
    result_mesh = trimesh.boolean.difference([main_mesh, cutout_union], engine='auto')
except Exception as e:
    print("Boolean difference operation failed with error:", e)
    result_mesh = trimesh.boolean.difference([main_mesh, cutout_union], engine='blender')
main_mesh = main_mesh.fill_holes()
# -----------------------------
# Export the resulting mesh to a new STL file.
# -----------------------------
result_mesh.export('2x2_MN_Array_scaled_minusSpheres.stl')
