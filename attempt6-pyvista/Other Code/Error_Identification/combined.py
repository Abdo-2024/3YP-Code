import trimesh
import numpy as np
import random
import csv

def deform_sphere(sphere, num_blobs=1, blob_radius=5, blob_intensity=0.01):
    vertices = sphere.vertices.copy()
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
    
    sphere.vertices = vertices
    return sphere

# Load the main STL file
main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_Smooth.stl')
min_coords, max_coords = main_mesh.bounds
margin = (max_coords[2] - min_coords[2]) * 0.05

# Prepare lists for both added and subtracted spheres
added_spheres = []
subtracted_spheres = []
sphere_data = []

num_add_spheres = 50
num_sub_spheres = 30

# Function to generate spheres (either for addition or subtraction)
def generate_spheres(num_spheres, add=True):
    spheres = []
    for _ in range(num_spheres):
        valid_sample = False
        while not valid_sample:
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            if point[2] > min_coords[2] + margin:
                valid_sample = True
        
        normal = main_mesh.face_normals[face_index]
        radius = random.uniform(5, 20) if add else random.uniform(10, 20)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        
        sphere_center = point + normal * radius * 0.5 if random.choice([True, False]) else point
        sphere.apply_translation(sphere_center)
        
        deformation_intensity = random.uniform(0.01, 0.5) * radius
        deformed_sphere = deform_sphere(sphere, num_blobs=30, blob_radius=radius * 1.2, blob_intensity=deformation_intensity)
        
        spheres.append(deformed_sphere)
        sphere_data.append({'x': sphere_center[0], 'y': sphere_center[1], 'z': sphere_center[2], 'radius': radius})
    
    return spheres

# Generate spheres for both operations
added_spheres = generate_spheres(num_add_spheres, add=True)
subtracted_spheres = generate_spheres(num_sub_spheres, add=False)

# Write sphere data to CSV
with open('sphere_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'y', 'z', 'radius']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in sphere_data:
        writer.writerow(data)

# Add spheres to the main mesh
combined_mesh = trimesh.util.concatenate([main_mesh] + added_spheres)

# Subtract spheres from the combined mesh
final_mesh = trimesh.boolean.difference([combined_mesh] + subtracted_spheres, engine='blender')

# Export the resulting mesh
final_mesh.export('2x2_C.stl')
