import pyvista as pv
import numpy as np

# Step 1: Load the STL model
# Replace 'model.stl' with the path to your STL file.
mesh = pv.read("/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN.stl")

# Step 2: Create a copy of the original mesh.
# This is good practice so that you can compare the original and modified meshes.
deformed_mesh = mesh.copy()

# Step 3: Define the noise amplitude.
# This value determines the strength of the random noise added to each vertex.
noise_amplitude = 0.0001  # You can adjust this value as needed.

# Step 4: Generate random noise for each point.
# 'deformed_mesh.points' is an array with shape (N, 3) where N is the number of vertices.
# We generate random values from a normal distribution with the given standard deviation.
noise = np.random.normal(scale=noise_amplitude, size=deformed_mesh.points.shape)

# Step 5: Add the random noise to the points.
# This perturbs each vertex by a small random vector.
deformed_mesh.points += noise

# Step 6: Define a deformation.
# In this example, we apply a sine-based deformation along the z-axis using the x-coordinate.
# 'warp_magnitude' controls the strength of the deformation.
warp_magnitude = 1  # Adjust this value to increase or decrease deformation.

# Step 7: Apply the deformation to each vertex.
# We loop over every point, compute a sine-based offset using its x-coordinate, and add that to the z-coordinate.
for i, point in enumerate(deformed_mesh.points):
    x, y, z = point  # Unpack the coordinates of the current vertex.
    deformation = warp_magnitude * np.sin(x)  # Calculate the deformation using sine.
    deformed_mesh.points[i, 2] += deformation  # Add the deformation to the z-coordinate.

# Step 8: Visualise the deformed mesh.
# We create a PyVista Plotter object, add the mesh, and then display it.
plotter = pv.Plotter()
plotter.add_mesh(deformed_mesh, color="lightblue", show_edges=True)
plotter.show()
