import trimesh
import numpy as np
import random
import csv
import pyvista as pv

############################################################################
# This function deforms the voxel (formerly sphere) using multiple blobs 
# of displacement. The deformation simulates the diffusion during polymerization.
############################################################################
def deform_voxel(voxel, num_blobs=5, blob_radius=0.5, blob_intensity=0.01):
    vertices = voxel.vertices.copy()
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    for i, vertex in enumerate(vertices):
        total_displacement = np.zeros(3)
        for blob_center in blob_centers:
            direction = vertex - blob_center
            distance = np.linalg.norm(direction)
            if distance < blob_radius:
                factor = 1 - (distance / blob_radius)
                unit_direction = direction / distance if distance != 0 else np.zeros(3)
                total_displacement += blob_intensity * factor * unit_direction
        vertices[i] = vertex + total_displacement
    
    voxel.vertices = vertices
    return voxel

############################################################################
# Create an ellipsoidal voxel from a base sphere.
# We take a unit sphere and scale it such that:
#   - Lateral dimensions (x and y) are scaled by 'scale'
#   - Axial dimension (z) is scaled by 5*scale
# This mimics a 2PP voxel with a 1:5 aspect ratio.
############################################################################
def create_voxel(scale):
    base = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Scale anisotropically: lateral radius = scale, axial radius = 5 * scale.
    scale_factors = np.array([scale, scale, scale * 5])
    base.vertices *= scale_factors
    return base

############################################################################
# Load the main STL file. Make sure the file path is correct.
############################################################################
main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl')
min_coords, max_coords = main_mesh.bounds
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare lists for added and subtracted voxels and a list to store voxel data.
added_voxels = []
subtracted_voxels = []
voxel_data = []

############################################################################
# Set the number of voxels (errors) for addition and subtraction.
############################################################################
num_add_voxels = 30
num_sub_voxels = 30

############################################################################
# Generate ellipsoidal voxels (instead of spheres).
# The "scale" variable here plays a role similar to the original sphere radius.
############################################################################
def generate_voxels(num_voxels, add=True):
    voxels = []
    for _ in range(num_voxels):
        valid_sample = False
        while not valid_sample:
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            if point[2] > min_coords[2] + margin:
                valid_sample = True
        
        normal = main_mesh.face_normals[face_index]
        # Choose a scale factor similar to the original radius (random between 10 and 15).
        scale = random.uniform(10, 15)
        voxel = create_voxel(scale)
        
        # Offset the voxel center along the normal direction (optionally)
        voxel_center = point + normal * scale * 0.5 if random.choice([True, False]) else point
        voxel.apply_translation(voxel_center)
        
        # Set deformation intensity based on the scale.
        deformation_intensity = random.uniform(0.05, 0.5) * scale
        # Use a blob radius proportional to the scale.
        deformed_voxel = deform_voxel(voxel, num_blobs=30, blob_radius=scale * 1.2, blob_intensity=deformation_intensity)
        
        voxels.append(deformed_voxel)
        voxel_data.append({
            'x': voxel_center[0],
            'y': voxel_center[1],
            'z': voxel_center[2],
            'scale': scale,
            'label': "addition" if add else "subtraction"
        })
    
    return voxels

# Generate voxels for both adding and subtracting material.
added_voxels = generate_voxels(num_add_voxels, add=True)
subtracted_voxels = generate_voxels(num_sub_voxels, add=False)

############################################################################
# Write voxel data (position, scale, and operation) to a CSV file.
############################################################################
with open('voxel_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'scale', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in voxel_data:
        writer.writerow(data)

############################################################################
# Combine the voxels with the main mesh:
# First, add the voxels, then subtract the others.
############################################################################
combined_mesh = trimesh.util.concatenate([main_mesh] + added_voxels)
final_mesh = trimesh.boolean.difference([combined_mesh] + subtracted_voxels, engine='blender')

# If the boolean operation returned multiple meshes, merge them into one.
if isinstance(final_mesh, list):
    final_mesh = trimesh.util.concatenate(final_mesh)
elif hasattr(final_mesh, 'geometry'):
    final_mesh = trimesh.util.concatenate(list(final_mesh.geometry.values()))

############################################################################
# Export the resulting mesh with simulated printing errors.
############################################################################
final_mesh.export('/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.stl')

############################################################################
# Visualize using PyVista.
############################################################################
pv_mesh = pv.wrap(final_mesh)
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color='lightgray', opacity=1.0)
plotter.show()
