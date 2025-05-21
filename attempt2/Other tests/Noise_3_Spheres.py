import trimesh
import numpy as np
import random
"""
Sphere Placement on STL Mesh Using Trimesh

This script adds multiple spheres to a given STL mesh by sampling random points on its surface,
placing icospheres at these points, and exporting the combined mesh as a new STL file.

Key Functionalities:
- Loads an STL mesh from a specified file path using Trimesh.
- Samples random points on the surface of the main mesh to determine sphere placement.
- Creates icospheres of random sizes at each sampled point.
- Ensures each sphere touches the main mesh by translating it to the sampled surface point.
- Combines the original mesh and all generated spheres into a single mesh.
- Exports the combined mesh, including the original mesh and all added spheres, to a new STL file.

Dependencies:
- Trimesh: For loading and manipulating 3D meshes, sampling surface points, creating spheres, and exporting meshes.

Usage:
Ensure the `trimesh` library is installed (`pip install trimesh`) and adjust the `num_spheres` variable
to specify the number of spheres to add. Modify the radius range (`0.05, 0.1` in this example) to control
the size variability of the added spheres. The modified mesh will be saved to the file specified by
`combined.stl` in the script.
"""

# Step 1: Load the main STL file.
# Replace 'original.stl' with the path to your STL file.
mesh = trimesh.load('/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl')

# Create a list to hold all meshes (the original mesh and the spheres).
meshes = [mesh]

# Step 2: Define the number of spheres you want to add.
num_spheres = 5  # You can change this to any number you wish.

# Step 3: For each sphere, sample a random point on the surface of the main mesh.
for i in range(num_spheres):
    # 3.1: Sample a random point on the mesh surface.
    # The function 'trimesh.sample.sample_surface' returns:
    #    - points: an array of points sampled on the surface.
    #    - face_indices: an array of indices corresponding to the faces where each point was sampled.
    points, face_indices = trimesh.sample.sample_surface(mesh, count=1)
    
    # Extract the sampled point (a 3D coordinate) and the corresponding face index.
    point = points[0]
    face_index = face_indices[0]
    
    # 3.2: Retrieve the normal of the face at the sampled point.
    # This normal indicates the direction that is "outward" from the mesh at that point.
    normal = mesh.face_normals[face_index]
    
    # 3.3: Define a random radius for the sphere.
    # You can adjust the range (0.5, 3) as needed.
    radius = random.uniform(0.05, 0.1)
    
    # 3.4: Create an icosphere.
    # The icosphere is a common approximation of a sphere using triangular facets.
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    
    # 3.5: Translate the sphere so that it touches the main mesh.
    # To do this, position the sphere's centre at:
    #    sampled_point + (normal vector * sphere_radius)
    # This ensures that one point on the sphere's surface contacts the main mesh.
    sphere_center = point + normal * radius
    sphere.apply_translation(sphere_center)
    
    # Add the sphere to our list of meshes.
    meshes.append(sphere)

# Step 4: Combine all the meshes (main STL and spheres) into a single mesh.
combined_mesh = trimesh.util.concatenate(meshes)

# Step 5: Export the combined mesh to a new STL file.
combined_mesh.export('combined.stl')
