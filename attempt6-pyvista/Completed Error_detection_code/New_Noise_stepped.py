import trimesh
import numpy as np
import random
import csv
import pyvista as pv

############################################################################
# Create an ellipsoidal voxel (representing a 2PP voxel) by scaling a sphere.
############################################################################
def create_ellipsoid():
    # Create a base unit sphere (radius=1, diameter=2)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Scale anisotropically to get an ellipsoid with dimensions 2 (x,y) and 10 (z)
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
# Rotate the voxel so that its local -z axis aligns with the given normal.
############################################################################
def align_voxel_to_normal(voxel, normal):
    # We want to rotate the voxel so that its local -z (i.e. [0, -1, 0]) aligns with the surface normal.
    R = trimesh.geometry.align_vectors(np.array([0, -1, 0]), normal)
    voxel.apply_transform(R)
    return R

############################################################################
# Load the main model mesh.
############################################################################
main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/Samples/STL/2x2_NEW_MN.stl')
min_coords, max_coords = main_mesh.bounds
margin = (max_coords[2] - min_coords[2]) * 0.05

# Lists for voxel data and for later Boolean operations.
voxel_data = []

# Reduce number of voxels to reduce computational expense.
num_add_voxels = 2
num_sub_voxels = 2

############################################################################
# Generate voxels in three stages:
#   1. Base (generated sphere before deformation)
#   2. Deformed sphere
#   3. Aligned (moved to surface) sphere
############################################################################
def generate_voxels(num_voxels, add=True):
    base_voxels = []
    deformed_voxels = []
    aligned_voxels = []
    
    for _ in range(num_voxels):
        valid_sample = False
        while not valid_sample:
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            if point[2] > min_coords[2] + margin:
                valid_sample = True

        normal = main_mesh.face_normals[face_index]
        
        # Stage 1: Create the base sphere.
        voxel_base = create_ellipsoid()
        base_voxels.append(voxel_base.copy())
        
        # Stage 2: Deform the base sphere.
        # Use a characteristic length to set the deformation intensity.
        characteristic_length = np.linalg.norm([1, 1, 5])
        deformation_intensity = random.uniform(0.05, 0.5) * characteristic_length
        # For speed, we use 10 blobs (you can adjust if needed)
        blob_radius = np.mean([10, 10, 50]) * 0.8
        voxel_deformed = deform_voxel(voxel_base.copy(), num_blobs=10, blob_radius=blob_radius,
                                      blob_intensity=deformation_intensity, sigma_factor=1.0)
        deformed_voxels.append(voxel_deformed.copy())
        
        # Stage 3: Align and translate the deformed sphere so its local bottom touches the surface.
        voxel_aligned = voxel_deformed.copy()
        R = align_voxel_to_normal(voxel_aligned, normal)
        local_bottom = np.array([0, 0, -5])
        # Compute bottom offset using the rotation component
        bottom_offset = R[:3, :3].dot(local_bottom)
        translation = point - bottom_offset
        voxel_aligned.apply_translation(translation)
        aligned_voxels.append(voxel_aligned.copy())
        
        # Save voxel data for potential further use.
        voxel_data.append({
            'x': translation[0],
            'y': translation[1],
            'z': translation[2],
            'dimensions': [2, 2, 10],
            'label': "addition" if add else "subtraction"
        })
        
    return base_voxels, deformed_voxels, aligned_voxels

# Generate voxels for additions and subtractions.
base_added, deformed_added, aligned_added = generate_voxels(num_add_voxels, add=True)
base_subtracted, deformed_subtracted, aligned_subtracted = generate_voxels(num_sub_voxels, add=False)

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
# Export the three intermediate voxel sets as combined meshes.
############################################################################
all_base_voxels = base_added + base_subtracted
all_deformed_voxels = deformed_added + deformed_subtracted
all_aligned_voxels = aligned_added + aligned_subtracted

# Concatenate each list into a single mesh.
base_mesh = trimesh.util.concatenate(all_base_voxels)
deformed_mesh = trimesh.util.concatenate(all_deformed_voxels)
aligned_mesh = trimesh.util.concatenate(all_aligned_voxels)

# Export each stage as an .obj file.
base_mesh.export('base_spheres.obj')         # Generated sphere before deformation.
deformed_mesh.export('deformed_spheres.obj')   # Generated spheres after deformation.
aligned_mesh.export('aligned_spheres.obj')     # Sample when spheres have been moved to the surface.

############################################################################
# Combine the aligned spheres with the main mesh:
#   Add the aligned-added voxels, then subtract the aligned-subtracted voxels.
############################################################################
combined_mesh = trimesh.util.concatenate([main_mesh] + aligned_added)
final_mesh = trimesh.boolean.difference([combined_mesh] + aligned_subtracted, engine='blender')

if isinstance(final_mesh, list):
    final_mesh = trimesh.util.concatenate(final_mesh)
elif hasattr(final_mesh, 'geometry'):
    final_mesh = trimesh.util.concatenate(list(final_mesh.geometry.values()))

############################################################################
# Export the final mesh (after Boolean operations) as an .obj.
############################################################################
final_mesh.export('final_mesh.obj')

############################################################################
# Visualize the final result using PyVista.
############################################################################
pv_mesh = pv.wrap(final_mesh)
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color='lightgray', opacity=1.0)
plotter.show()
