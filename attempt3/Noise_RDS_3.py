import trimesh
import numpy as np
import random
import csv

# =============================================================================
# Global parameters – adjust these to control sphere deformation and placement.
# =============================================================================
NUM_BLOBS = 15             # Number of blobs for sphere deformation
BLOB_RADIUS = 0.05         # Blob sphere of influence radius for deformation
BLOB_INTENSITY = 0.01      # Blob intensity multiplier (applied as: intensity = BLOB_INTENSITY * sphere_radius)
NUM_SPHERES_ADD = 100      # Number of spheres to add (union) to the model
NUM_SPHERES_SUBTRACT = 100 # Number of spheres to subtract from the model
MIN_SPHERE_RADIUS = 0.005  # Minimum radius for generated spheres
MAX_SPHERE_RADIUS = 0.015  # Maximum radius for generated spheres

# =============================================================================
# Function: deform_sphere
# =============================================================================
def deform_sphere(sphere, num_blobs=15, blob_radius=0.05, blob_intensity=0.01):
    """
    Deform the sphere to create a 'blob' effect by displacing vertices based on random blob centers.
    
    Parameters:
      sphere (trimesh.Trimesh): The sphere mesh to deform.
      num_blobs (int): The number of random blobs to use for deformation.
      blob_radius (float): The sphere of influence for each blob.
      blob_intensity (float): The maximum displacement applied at the centre of a blob.
    
    Returns:
      trimesh.Trimesh: The deformed sphere mesh.
    """
    # Copy the vertices of the sphere
    vertices = sphere.vertices.copy()

    # Compute the bounding box of the sphere's vertices.
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate random blob centers within the bounding box.
    blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    # For each vertex, compute the total displacement from all blobs.
    for i, vertex in enumerate(vertices):
        total_displacement = np.zeros(3)
        for blob_center in blob_centers:
            direction = vertex - blob_center
            distance = np.linalg.norm(direction)
            if distance < blob_radius:
                # Influence decreases linearly with distance.
                factor = 1 - (distance / blob_radius)
                unit_direction = direction / distance if distance != 0 else np.zeros(3)
                total_displacement += blob_intensity * factor * unit_direction
        vertices[i] = vertex + total_displacement

    # Update the sphere's vertices.
    sphere.vertices = vertices
    return sphere

# =============================================================================
# Main Code
# =============================================================================

# Step 1: Load the main STL file (the model that will be modified)
main_mesh = trimesh.load('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/2x2_MN_Array_scaled.stl')

# Compute the bounding box of the main mesh.
min_coords, max_coords = main_mesh.bounds

# Define a margin above the bottom of the model (5% of the model's height).
margin = (max_coords[2] - min_coords[2]) * 0.05

# Lists to hold the meshes for addition and subtraction.
addition_spheres = []
subtraction_spheres = []

# List to record sphere parameters for CSV output.
# Each record includes: x, y, z, radius, and operation ("add" or "subtract")
sphere_data = []

# -----------------------------------------------------------------------------
# Step 2: Create spheres that will be ADDED to the main mesh.
# -----------------------------------------------------------------------------
for i in range(NUM_SPHERES_ADD):
    valid_sample = False
    while not valid_sample:
        # Sample a random point on the main mesh's surface.
        points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
        point = points[0]
        face_index = face_indices[0]
        # Ensure the point is not too near the bottom.
        if point[2] > min_coords[2] + margin:
            valid_sample = True

    normal = main_mesh.face_normals[face_index]
    
    # Randomly choose a sphere radius.
    radius = random.uniform(MIN_SPHERE_RADIUS, MAX_SPHERE_RADIUS)
    
    # Create an icosphere.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # Randomly decide whether to attach by the sphere's surface or center.
    if random.choice([True, False]):
        sphere_center = point + normal * radius  # Surface-attached
    else:
        sphere_center = point                   # Center-attached
    sphere.apply_translation(sphere_center)
    
    # Compute the deformation intensity using the global multiplier.
    deformation_intensity = BLOB_INTENSITY * radius
    
    # Deform the sphere.
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=NUM_BLOBS, 
                                    blob_radius=BLOB_RADIUS, 
                                    blob_intensity=deformation_intensity)
    
    # Append the deformed sphere to the addition list.
    addition_spheres.append(deformed_sphere)
    
    # Record sphere data.
    sphere_data.append({
        'x': sphere_center[0],
        'y': sphere_center[1],
        'z': sphere_center[2],
        'radius': radius,
        'operation': 'add'
    })

# -----------------------------------------------------------------------------
# Step 3: Create spheres that will be SUBTRACTED from the main mesh.
# -----------------------------------------------------------------------------
for i in range(NUM_SPHERES_SUBTRACT):
    valid_sample = False
    while not valid_sample:
        # Sample a random point on the main mesh's surface.
        points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
        point = points[0]
        face_index = face_indices[0]
        # Use the same condition to avoid the very bottom.
        if point[2] > min_coords[2] + margin:
            valid_sample = True

    normal = main_mesh.face_normals[face_index]
    
    # Randomly choose a sphere radius.
    radius = random.uniform(MIN_SPHERE_RADIUS, MAX_SPHERE_RADIUS)
    
    # Create an icosphere.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # Decide on the attachment method.
    if random.choice([True, False]):
        sphere_center = point + normal * radius  # Surface-attached
    else:
        sphere_center = point                   # Center-attached
    sphere.apply_translation(sphere_center)
    
    # Compute the deformation intensity.
    deformation_intensity = BLOB_INTENSITY * radius
    
    # Deform the sphere.
    deformed_sphere = deform_sphere(sphere, 
                                    num_blobs=NUM_BLOBS, 
                                    blob_radius=BLOB_RADIUS, 
                                    blob_intensity=deformation_intensity)
    
    # Append the deformed sphere to the subtraction list.
    subtraction_spheres.append(deformed_sphere)
    
    # Record sphere data.
    sphere_data.append({
        'x': sphere_center[0],
        'y': sphere_center[1],
        'z': sphere_center[2],
        'radius': radius,
        'operation': 'subtract'
    })

# -----------------------------------------------------------------------------
# Step 4: Combine the meshes using boolean operations.
# -----------------------------------------------------------------------------
def ensure_closed_volume(mesh):
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to repair with process()...")
        mesh = mesh.process()
    if not mesh.is_watertight:
        print("Mesh still not watertight after process(). Using convex hull as fallback.")
        mesh = mesh.convex_hull
    return mesh

# Ensure all addition meshes are closed volumes.
all_addition = [ensure_closed_volume(m) for m in ([main_mesh] + addition_spheres)]
print("Performing boolean union for addition spheres...")
union_mesh = trimesh.boolean.union(all_addition, engine='blender')

# Ensure subtraction meshes are closed volumes.
subtraction_spheres = [ensure_closed_volume(m) for m in subtraction_spheres]
print("Performing boolean union for subtraction spheres...")
subtraction_union = trimesh.boolean.union(subtraction_spheres, engine='blender')

print("Performing boolean difference (subtraction)...")
final_mesh = trimesh.boolean.difference([union_mesh, subtraction_union], engine='blender')

# -----------------------------------------------------------------------------
# Step 5: Write sphere data to a CSV file.
# -----------------------------------------------------------------------------
with open('sphere_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius', 'operation']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in sphere_data:
         writer.writerow(data)

# -----------------------------------------------------------------------------
# Step 6: Export the final mesh to an STL file.
# -----------------------------------------------------------------------------
final_mesh.export('2x2_MN_Array_scaled+AddSubNoise.stl')

