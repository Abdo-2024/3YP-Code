import trimesh
import numpy as np
import random
import csv
import pyvista as pv

############################################################################
# Create an ellipsoidal voxel (representing a 2PP voxel) by scaling a sphere.
# In this example, the local ellipsoid has diameters [2, 2, 10],
# which means a radius of 1 in x and y, and 5 in z.
############################################################################
def create_ellipsoid():
    # Create a base unit sphere (radius=1, diameter=2)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Scale anisotropically to get an ellipsoid with local dimensions 2 (x,y) and 10 (z)
    scale_factors = np.array([0.5, 0.5, 2.5])
    sphere.vertices = sphere.vertices * scale_factors
    return sphere

############################################################################
# Deformation function using Gaussian weighting to simulate diffusion.
############################################################################
def deform_voxel(voxel, num_blobs=5, blob_radius=0.5, blob_intensity=0.01, sigma_factor=1.0):
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
                sigma = blob_radius * sigma_factor
                weight = np.exp(- (distance**2) / (2 * sigma**2))
                unit_direction = direction / distance if distance != 0 else np.zeros(3)
                total_displacement += blob_intensity * weight * unit_direction
        vertices[i] = vertex + total_displacement
    
    voxel.vertices = vertices
    return voxel

############################################################################
# Rotate the voxel so that its local -z axis (the “bottom”) aligns with the given normal.
############################################################################
def align_voxel_to_normal(voxel, normal):
    # We want to rotate the voxel so that its local -z (i.e. [0, 0, -1]) aligns with the surface normal.
    R = trimesh.geometry.align_vectors(np.array([0, -1, 0]), normal)
    voxel.apply_transform(R)
    return R

############################################################################
# Load the main model mesh.
############################################################################
main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl')
min_coords, max_coords = main_mesh.bounds
margin = (max_coords[2] - min_coords[2]) * 0.05

# Lists for voxels and voxel data.
added_voxels = []
subtracted_voxels = []
voxel_data = []

############################################################################
# Set the number of voxels (errors) for addition and subtraction.
############################################################################
num_add_voxels = 10
num_sub_voxels = 10

############################################################################
# Generate ellipsoidal voxels such that each voxel’s bottom touches the model.
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
        voxel = create_ellipsoid()
        
        # Align the voxel so its local -z axis (its bottom) aligns with the surface normal.
        R = align_voxel_to_normal(voxel, normal)
        local_bottom = np.array([0, 0, -5])
        # Extract the rotation component (3x3) from R to compute the offset.
        bottom_offset = R[:3, :3].dot(local_bottom)
        translation = point - bottom_offset
        voxel.apply_translation(translation)
        
        # (Rest of the deformation and data collection code follows...)
        characteristic_length = np.linalg.norm([1, 1, 5])
        deformation_intensity = random.uniform(0.05, 0.5) * characteristic_length
        blob_radius = np.mean([10, 10, 50]) * 0.8
        deformed_voxel = deform_voxel(voxel, num_blobs=10, blob_radius=blob_radius,
                                      blob_intensity=deformation_intensity, sigma_factor=1.0)
        
        voxels.append(deformed_voxel)
        voxel_data.append({
            'x': translation[0],
            'y': translation[1],
            'z': translation[2],
            'dimensions': [2, 2, 10],
            'label': "addition" if add else "subtraction"
        })
    return voxels

# Generate voxels for both adding and subtracting material.
added_voxels = generate_voxels(num_add_voxels, add=True)
subtracted_voxels = generate_voxels(num_sub_voxels, add=False)

############################################################################
# Write voxel data (position, dimensions, and operation) to a CSV file.
############################################################################
with open('voxel_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'dimensions', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in voxel_data:
        writer.writerow(data)

############################################################################
# Combine the voxels with the main mesh:
# First add the voxels, then subtract the others.
############################################################################
combined_mesh = trimesh.util.concatenate([main_mesh] + added_voxels)
final_mesh = trimesh.boolean.difference([combined_mesh] + subtracted_voxels, engine='blender')

if isinstance(final_mesh, list):
    final_mesh = trimesh.util.concatenate(final_mesh)
elif hasattr(final_mesh, 'geometry'):
    final_mesh = trimesh.util.concatenate(list(final_mesh.geometry.values()))

############################################################################
# Export the final mesh with simulated printing errors.
############################################################################
final_mesh.export('/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN_Noisy.obj')

############################################################################
# Visualize the final result using PyVista.
############################################################################
pv_mesh = pv.wrap(final_mesh)
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color='lightgray', opacity=1.0)
plotter.show()
