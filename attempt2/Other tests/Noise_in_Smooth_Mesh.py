import open3d as o3d
import numpy as np
"""
Mesh Defect Simulation Using Open3D and NumPy

This script simulates defects on a 3D mesh loaded from an STL file using Open3D and NumPy. 
It adds random noise to the vertices of the mesh to simulate imperfections or manufacturing
defects. The noise scale (`noise_scale`) controls the intensity of the defects.

Key Functionalities:
- Loads an STL mesh file using Open3D.
- Converts the mesh vertices to a NumPy array for numerical operations.
- Defines a noise scale (`noise_scale`) to control the intensity of the simulated defects.
- Generates random noise using NumPy's normal distribution.
- Adds the generated noise to the mesh vertices to simulate defects.
- Updates the mesh with the modified vertices and recomputes vertex normals for proper visualization.
- Saves the modified mesh to a new STL file.

Dependencies:
- Open3D: For loading, processing, and saving 3D mesh data.
- NumPy: For numerical operations and generating random noise.

Usage:
Ensure the Open3D library is installed (`pip install open3d`). Adjust the `noise_scale` parameter
to control the intensity of the simulated defects. The modified mesh will be saved as
`model_with_defects.stl` in the current directory.
"""

# Step 1: Load the STL file
mesh = o3d.io.read_triangle_mesh("/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl")

# Step 2: Convert vertices to a NumPy array
vertices = np.asarray(mesh.vertices)

# Step 3: Define the noise scale and generate noise
noise_scale = 0.001  # Adjust this value for more or less defect intensity
noise = np.random.normal(loc=0.0, scale=noise_scale, size=vertices.shape)

# Step 4: Add the noise to the vertices
vertices += noise

# Step 5: Update the mesh with the new vertices
mesh.vertices = o3d.utility.Vector3dVector(vertices)

# Step 6: Recompute normals for better visualisation
mesh.compute_vertex_normals()

# Step 7: Save the modified mesh to a new STL file
o3d.io.write_triangle_mesh("model_with_defects.stl", mesh)
