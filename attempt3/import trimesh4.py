import trimesh
import random
import numpy as np
"""
This script loads a 3D mesh from an STL file and ensures it is watertight and a valid volume, repairing it if necessary. It calculates the bounding box and defines a margin to avoid placing spheres too close to the model’s base (bottom 5%). It then generates a set number of spheres by sampling random surface points (above the margin), creating spheres with random radii at those points, and validating them.

All valid spheres are progressively combined using Boolean union operations into a single mesh. The script checks the integrity of this combined mesh and attempts to repair it if it's not watertight or a valid volume. If the combined spheres mesh is valid, it is subtracted from the main mesh using a Boolean difference. The final result, with all spheres subtracted, is saved as a new STL file.
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

    # Centre the sphere at the sampled point
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

# Step 1: Generate all spheres and add them to a list
spheres = []
for i in range(num_spheres):
    sphere = generate_and_add_sphere()

    # Check if the sphere is valid before adding it
    if not sphere.is_volume:
        print(f"Generated sphere is not a valid volume at iteration {i}. Skipping sphere.")
        continue
    
    spheres.append(sphere)

# Step 2: Union all spheres into a single solid mesh
# Start with the first sphere
combined_spheres = spheres[0]

# Perform the union operation on the remaining spheres
for sphere in spheres[1:]:
    combined_spheres = combined_spheres.union(sphere)

# Check if the combined spheres form a valid volume
if not combined_spheres.is_watertight:
    print("Combined spheres are not watertight. Attempting to repair...")
    combined_spheres.fill_holes()

if not combined_spheres.is_volume:
    print("Combined spheres are still not a valid volume. Skipping subtraction.")
else:
    # Step 3: Subtract the combined spheres mesh from the main model
    if not main_mesh.is_volume:
        print("Main mesh is not a valid volume before subtraction. Skipping.")
    else:
        # Perform the subtraction in one go
        main_mesh = main_mesh.difference(combined_spheres)

        # Save the resulting model with subtracted spheres
        main_mesh.export('/home/a/Documents/Obsidian Vault/Oxford/3rd Year/3YP - 3D Printer Nano Project/Experiments/Code/attempt3/swiss_cheese_model5.stl')
