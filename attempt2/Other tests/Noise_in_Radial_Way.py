import numpy as np
from stl import mesh
"""
Mesh Deformation with Centroid-Based Displacement Using STL and NumPy

This script deforms a given STL mesh by displacing each vertex based on its distance from
the mesh's centroid. It calculates the centroid of the mesh, defines a displacement intensity,
and modifies each vertex to create a "blob" effect where vertices move outward from the centroid.

Key Functionalities:
- Loads an STL mesh from a specified file path using the `mesh` module from the `stl` library.
- Computes the overall centroid of the mesh using NumPy.
- Defines a displacement intensity (`blob_intensity`) to control the extent of vertex displacement.
- Iterates over each vertex in the mesh, computing a displacement vector from the centroid,
  normalizing it, and applying a constant displacement proportional to `blob_intensity`.
- Saves the deformed mesh to a new STL file.

Dependencies:
- NumPy: For numerical operations and vector calculations.
- STL (from `stl` library): For loading and manipulating STL meshes.

Usage:
Ensure the `stl` library is installed (`pip install numpy-stl`) and adjust the `blob_intensity`
variable to control the strength of the blob effect. The modified mesh will be saved to the file
specified by `model_blob.stl` in the script.
"""

# Load your mesh
input_file = '/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl'
output_file = 'model_blob.stl'
your_mesh = mesh.Mesh.from_file(input_file)

# Reshape the vertex array (each triangle has 3 vertices)
all_vertices = your_mesh.vectors.reshape(-1, 3)

# Compute the overall centroid of the mesh
centroid = np.mean(all_vertices, axis=0)

# Define how strong the displacement should be
blob_intensity = 0.3  # Adjust this value to get more or less of a blob effect

for i in range(len(your_mesh.vectors)):
    for j in range(3):
        vertex = your_mesh.vectors[i][j]
        direction = vertex - centroid  # Vector from the centroid to the vertex
        norm = np.linalg.norm(direction)
        if norm != 0:
            unit_direction = direction / norm
        else:
            unit_direction = np.zeros(3)
        
        # Displace the vertex along this direction.
        # Here, we simply add a constant displacement; you could also apply a function
        # (for example, a sinusoidal or exponential function) to control the shape further.
        your_mesh.vectors[i][j] += blob_intensity * unit_direction

# Save the deformed mesh
your_mesh.save(output_file)
