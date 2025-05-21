import trimesh
import numpy as np
import random
import csv
import pyvista as pv

############################################################################
# This function deforms any mesh by displacing its vertices based on nearby blob centers.
############################################################################
def deform_mesh(mesh, num_blobs=5, blob_radius=0.1, blob_intensity=0.01):
    vertices = mesh.vertices.copy()
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
    
    mesh.vertices = vertices
    return mesh

# Load the main STL file – ensure the path is correct!
main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/STL/2x2_MN_Smooth.stl')
min_coords, max_coords = main_mesh.bounds
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare lists for added and subtracted ellipsoidal voxels, and a list to store voxel data
added_voxels = []
subtracted_voxels = []
voxel_data = []

############################################################################
# Define voxel (ellipsoid) dimensions:
# Lateral diameters (x, y): 0.2 micrometers (200 nm)
# Axial diameter (z): 1.0 micrometer
############################################################################
voxel_extents = np.array([2, 2, 10])
# For ellipsoid scaling, we use half of the extents (i.e. the radii)
scale_factors = voxel_extents / 2

############################################################################
# Generate ellipsoidal voxels for either addition or subtraction.
# This function samples a point on the main mesh, creates an ellipsoidal voxel,
# optionally offsets it along the face normal, and applies a deformation.
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
        
        # Create an ellipsoid by generating an icosphere and scaling it.
        ellipsoid_voxel = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        ellipsoid_voxel.apply_scale(scale_factors)
        
        # Randomly decide whether to offset the ellipsoid center along the normal.
        voxel_center = point + normal * scale_factors[2] if random.choice([True, False]) else point
        ellipsoid_voxel.apply_translation(voxel_center)
        
        # Apply deformation to the ellipsoid.
        deformation_intensity = random.uniform(0.03, 0.01) * scale_factors[2]
        deformed_voxel = deform_mesh(ellipsoid_voxel, num_blobs=30, blob_radius=scale_factors[2] * 1, blob_intensity=deformation_intensity)
        
        voxels.append(deformed_voxel)
        voxel_data.append({
            'x': voxel_center[0],
            'y': voxel_center[1],
            'z': voxel_center[2],
            'extent': voxel_extents.tolist(),
            'label': "addition" if add else "subtraction"
        })
    
    return voxels

# Set the number of ellipsoidal voxels for each operation.
num_add_voxels = 50
num_sub_voxels = 50

# Generate the ellipsoidal voxels.
added_voxels = generate_voxels(num_add_voxels, add=True)
subtracted_voxels = generate_voxels(num_sub_voxels, add=False)

# Write voxel data to CSV (updated file name and field 'extent' instead of 'radius')
with open('voxel_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'extent', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in voxel_data:
        writer.writerow(data)

# Add voxels to the main mesh.
combined_mesh = trimesh.util.concatenate([main_mesh] + added_voxels)

# Subtract voxels from the combined mesh.
final_mesh = trimesh.boolean.difference([combined_mesh] + subtracted_voxels, engine='blender')

# In case multiple meshes are returned, merge them into one.
if isinstance(final_mesh, list):
    final_mesh = trimesh.util.concatenate(final_mesh)
elif hasattr(final_mesh, 'geometry'):
    final_mesh = trimesh.util.concatenate(list(final_mesh.geometry.values()))

# Export the resulting mesh.
final_mesh.export('/home/a/Documents/3YP/Code/attempt6-pyvista/STL/2x2_MN+Noise_voxels.stl')

# Visualize using PyVista.
pv_mesh = pv.wrap(final_mesh)
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color='lightgray', opacity=1.0)
plotter.show()
