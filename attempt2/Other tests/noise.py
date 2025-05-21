import numpy as np               # Import the numpy library for numerical operations.
from stl import mesh             # Import the mesh class from the numpy-stl library.
"""
STL Mesh Noise Addition Script

This script adds random noise to the vertices of an STL mesh file. It is useful for simulating imperfections
or deformations in 3D models for various applications such as testing robustness of algorithms or creating
realistic models for simulation.

Key Functionalities:
- Loads an STL mesh from a specified file path.
- Defines the noise scale to control the magnitude of noise added to each vertex.
- Iterates through each triangle in the mesh and adds random noise to each vertex.
- Saves the modified mesh with added noise to a new STL file.

Dependencies:
- NumPy: For numerical operations and generating random noise vectors.
- numpy-stl (stl.mesh): For loading and manipulating STL mesh files.

Usage:
Ensure the `input_file` variable points to the desired STL file path. Adjust the `noise_scale` parameter to
control the amount of noise added to the mesh vertices. The modified mesh will be saved to the `output_file`
specified in the script.
"""

# Step 2.1: Load the STL file.
# Specify the input STL file and the output file.
input_file = '/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl'         # Replace 'model.stl' with the path to your STL file.
output_file = '2x2_MN_Array_scaled_with_noise.stl'

# Load the STL model into a mesh object.
your_mesh = mesh.Mesh.from_file(input_file)

# Step 2.2: Define the noise parameters.
# noise_scale determines the magnitude of the noise added.
noise_scale = 0.005                # Adjust this value to control the amount of noise (defect) applied.

# Step 2.3: Iterate over all the triangles and add random noise to each vertex.
# Each triangle in the STL file is represented by 3 vertices.
for i in range(len(your_mesh.vectors)):      # Loop over each triangle (vector) in the mesh.
    for j in range(3):                       # Loop over each vertex (3 per triangle).
        # Generate a random noise vector.
        # np.random.normal(0, noise_scale, 3) creates a 3-element array where each element
        # is drawn from a normal distribution with a mean of 0 and standard deviation of noise_scale.
        noise = np.random.normal(0, noise_scale, 3)
        
        # Add the noise to the vertex.
        your_mesh.vectors[i][j] += noise

# Step 2.4: Save the modified mesh to a new STL file.
your_mesh.save(output_file)
