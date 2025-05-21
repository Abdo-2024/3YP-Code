import trimesh
import random
import numpy as np
"""
This script generates a "swiss cheese" effect on a 3D model by randomly placing and subtracting spherical voids from a base mesh. It ensures the base model is watertight and a valid volume before performing Boolean operations. Spheres are randomly positioned on the mesh surface (avoiding the bottom region) and subtracted if both objects are valid volumes. The modified mesh is then saved as a new STL file.
"""

# Load the main STL file (the model you do not want to deform)
main_mesh = trimesh.load('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model.stl')

# Check if the main mesh is watertight and a valid volume, and repair it if necessary
if not main_mesh.is_watertight:
    print("Repairing main mesh...")
    main_mesh.fill_holes()

if not main_mesh.is_volume:
    print("Main mesh is not a valid volume. Attempting repair...")
    main_mesh = main_mesh.repair().fill_holes()

# Ensure the mesh is valid after repair
if not main_mesh.is_watertight or not main_mesh.is_volume:
    raise ValueError("Main mesh is still not a valid volume after repair!")

# Compute the bounding box of the main mesh.
min_coords, max_coords = main_mesh.bounds

# Define a margin above the bottom of the model (5% of the model's height)
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare a list to hold the meshes (this will include the main mesh and the spheres)
meshes = [main_mesh]

# Prepare a list to hold the sphere data for CSV output
sphere_data = []  # Each element will be a dictionary with keys: x, y, z, radius

# Define the number of spheres to add
num_spheres = 30  # Adjust this number as needed

# Define a function to create and add spheres
def generate_and_add_sphere():
    valid_sample = False
    while not valid_sample:
        # Sample a random point on the surface of the main mesh
        points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
        point = points[0]
        face_index = face_indices[0]

        # Ensure the point is not too close to the bottom of the model
        if point[2] > min_coords[2] + margin:
            valid_sample = True

    # Retrieve the normal at the sampled point
    normal = main_mesh.face_normals[face_index]
    
    # Randomly define a radius for the sphere
    radius = random.uniform(0.005, 0.02)

    # Create an icosphere with the defined radius
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)

    # Ensure the sphere is watertight and a valid volume
    if not sphere.is_watertight:
        sphere.fill_holes()
    if not sphere.is_volume:
        sphere = sphere.repair()

    # Decide whether the sphere will be "surface-attached" or "centre-attached"
    #if random.choice([True, False]):
        # "Surface-attached": Translate so the sphere's surface touches the model
       # sphere_center = point + normal * radius
   # else:
        # "Centre-attached": Translate so the sphere's centre is at the sampled point
    sphere_center = point

    sphere.apply_translation(sphere_center)

    # Record the sphere's data
    sphere_data.append({
        'x': sphere_center[0],
        'y': sphere_center[1],
        'z': sphere_center[2],
        'radius': radius,
    })
    
    return sphere

# Step 2: Add spheres and subtract them from the model
for i in range(num_spheres):
    sphere = generate_and_add_sphere()

    # Check if both the main mesh and sphere are valid volumes
    if not main_mesh.is_volume:
        print(f"Main mesh is not a valid volume at iteration {i}. Skipping subtraction.")
        continue
    if not sphere.is_volume:
        print(f"Generated sphere is not a valid volume at iteration {i}. Skipping subtraction.")
        continue

    # Subtract the sphere from the main model (ensure both meshes are volumes)
    main_mesh = main_mesh.difference(sphere)

# Save the resulting model with subtracted spheres
main_mesh.export('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model4.stl')

