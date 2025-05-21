import numpy as np
from stl import mesh
"""
Random Blob Deformation Script for STL Mesh

This script deforms an STL mesh by adding random blobs of displacement to its vertices. Each blob
is defined by its center, radius of influence, and intensity of displacement, allowing for
customizable and controlled deformation of the mesh.

Key Functionalities:
- Loads an STL mesh from a specified file path.
- Computes the bounding box of the mesh to confine random blob centers within.
- Generates multiple random blob centers within the mesh's bounding box.
- Defines the radius and intensity of each blob to control its influence and displacement.
- Iterates through each vertex of the mesh and computes the cumulative displacement from all blobs
  within their respective radius of influence.
- Updates each vertex position based on the accumulated displacement from all relevant blobs.
- Saves the deformed mesh with added random blob deformations to a new STL file.

Dependencies:
- NumPy: For numerical operations and vector calculations.
- numpy-stl (stl.mesh): For loading and manipulating STL mesh files.

Usage:
Ensure the `input_file` variable points to the desired STL file path. Adjust `num_blobs`, `blob_radii`,
and `blob_intensities` to customize the number and characteristics of the random blobs added to the mesh.
The modified mesh will be saved to the `output_file` specified in the script.
"""

# Load your mesh
input_file = '/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl'
output_file = 'model_blob_random.stl'
your_mesh = mesh.Mesh.from_file(input_file)

# Reshape the vertex array (each triangle has 3 vertices)
all_vertices = your_mesh.vectors.reshape(-1, 3)

# Compute the bounding box of the mesh (min and max along each axis)
min_coords = np.min(all_vertices, axis=0)
max_coords = np.max(all_vertices, axis=0)

# Define the number of random blobs you want to add
num_blobs = 15  # You can adjust this number

# Generate random blob centres within the bounding box
blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]

# Define a radius and intensity for each blob (you may customise these lists)
# The radius is the sphere of influence for each blob.
# The intensity is the maximum displacement applied at the centre.
blob_radii = [0.5 for _ in range(num_blobs)]      # Adjust as needed (units match your model)
blob_intensities = [0.1 for _ in range(num_blobs)]  # Adjust for more or less deformation

# For each triangle and each vertex, add displacement from each blob if within its radius
for i in range(len(your_mesh.vectors)):
    for j in range(3):
        vertex = your_mesh.vectors[i][j]
        total_displacement = np.zeros(3)
        # Sum contributions from all blobs
        for k in range(num_blobs):
            blob_center = blob_centers[k]
            blob_radius = blob_radii[k]
            blob_intensity = blob_intensities[k]
            
            # Compute vector from the blob centre to the vertex
            direction = vertex - blob_center
            distance = np.linalg.norm(direction)
            
            # If the vertex is within the blob's sphere of influence, compute displacement
            if distance < blob_radius:
                # Factor decreases linearly with distance: 1 at the centre and 0 at the edge
                factor = 1 - (distance / blob_radius)
                # Determine the unit vector in the direction from the blob centre to the vertex
                if distance != 0:
                    unit_direction = direction / distance
                else:
                    unit_direction = np.zeros(3)
                # Add the blob's contribution to the total displacement
                total_displacement += blob_intensity * factor * unit_direction
        
        # Update the vertex position with the total displacement from all blobs
        your_mesh.vectors[i][j] += total_displacement

# Save the deformed mesh to a new STL file
your_mesh.save(output_file)
