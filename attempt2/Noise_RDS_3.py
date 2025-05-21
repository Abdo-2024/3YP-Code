#!/usr/bin/env python
"""
Noise_RDS_2.py

This script loads a main STL file, adds deformed icospheres (“blobs”) to it,
and exports a combined mesh with noise. All key parameters are imported from
parameters.py so that you only need to adjust values there.
"""

import trimesh
import numpy as np
import random
import csv
import parameters  # Import central parameters

def deform_sphere(sphere, num_blobs=15, blob_radius=0.05, blob_intensity=0.01):
    """
    Deform the sphere to create a 'blob' effect by displacing vertices based on random blob centres.
    
    Parameters:
      sphere (trimesh.Trimesh): The sphere mesh to deform.
      num_blobs (int): The number of random blob influences.
      blob_radius (float): The sphere of influence for each blob.
      blob_intensity (float): The maximum displacement at the blob's centre.
    
    Returns:
      trimesh.Trimesh: The deformed sphere mesh.
    """
    # Copy the vertices of the sphere.
    vertices = sphere.vertices.copy()

    # Compute the bounding box of the sphere's vertices.
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate random blob centres within the bounding box.
    blob_centers = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    # For each vertex, compute the total displacement from all blobs.
    for i, vertex in enumerate(vertices):
        total_displacement = np.zeros(3)
        for blob_center in blob_centers:
            direction = vertex - blob_center
            distance = np.linalg.norm(direction)
            if distance < blob_radius:
                # The influence decreases linearly with distance.
                factor = 1 - (distance / blob_radius)
                # Avoid division by zero.
                unit_direction = direction / distance if distance != 0 else np.zeros(3)
                total_displacement += blob_intensity * factor * unit_direction
        vertices[i] = vertex + total_displacement

    # Update the sphere's vertices with the deformed positions.
    sphere.vertices = vertices
    return sphere

def main():
    # -----------------------------
    # Step 1: Load the Main Mesh
    # -----------------------------
    main_mesh = trimesh.load('/home/a/Documents/3rd_Year/3YP/Code/attempt2/2x2_MN_Array_scaled.stl')
    min_coords, max_coords = main_mesh.bounds
    margin = (max_coords[2] - min_coords[2]) * 0.05  # 5% of the model's height

    # This list will hold the main mesh plus all the deformed spheres.
    meshes = [main_mesh]
    
    # List to record sphere data for CSV output.
    sphere_data = []  # Each entry will include x, y, z, radius, deformation_intensity

    # -----------------------------
    # Step 2: Define the Number of Spheres
    # -----------------------------
    num_spheres = parameters.NUM_SPHERES  # e.g., 50 as defined in parameters.py

    # -----------------------------
    # Step 3: Add Deformed Spheres (Blobs)
    # -----------------------------
    for i in range(num_spheres):
        valid_sample = False
        while not valid_sample:
            # Sample a random point on the main mesh's surface.
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            # Ensure the sample is above the bottom margin.
            if point[2] > min_coords[2] + margin:
                valid_sample = True

        normal = main_mesh.face_normals[face_index]
        
        # Use parameters for the sphere (blob) radius.
        radius = random.uniform(parameters.BLOB_RADIUS_MIN, parameters.BLOB_RADIUS_MAX)
        
        # Create an icosphere with the chosen radius.
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        
        # Randomly choose attachment type: by surface or by centre.
        if random.choice([True, False]):
            # "Surface-attached": the sphere's surface touches the model.
            sphere_center = point + normal * radius
        else:
            # "Centre-attached": the sphere's centre is at the sampled point.
            sphere_center = point
        sphere.apply_translation(sphere_center)
        
        # Choose a random deformation intensity factor using parameters.
        random_intensity_factor = random.uniform(parameters.BLOB_INTENSITY_MIN, parameters.BLOB_INTENSITY_MAX)
        deformation_intensity = random_intensity_factor * radius
        
        # Deform the sphere using parameters from parameters.py:
        # - Use NUM_BLOBS for the number of blob influences.
        # - Set blob_radius as a multiple of the sphere radius.
        # - Use the computed deformation_intensity.
        deformed_sphere = deform_sphere(
            sphere,
            num_blobs=parameters.NUM_BLOBS,
            blob_radius=radius * parameters.BLOB_RADIUS_MULTIPLIER,
            blob_intensity=deformation_intensity
        )
        
        # Add the deformed sphere to the list.
        meshes.append(deformed_sphere)
        
        # Record the sphere's data.
        sphere_data.append({
            'x': sphere_center[0],
            'y': sphere_center[1],
            'z': sphere_center[2],
            'radius': radius,
            'deformation_intensity': deformation_intensity
        })

    # -----------------------------
    # Step 4: Write Sphere Data to CSV
    # -----------------------------
    with open('sphere_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z', 'radius', 'deformation_intensity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in sphere_data:
            writer.writerow(data)

    # -----------------------------
    # Step 5: Combine and Export the Mesh
    # -----------------------------
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export('combined_deformed2.stl')
    print("Combined deformed mesh exported to 'combined_deformed2.stl'.")

if __name__ == '__main__':
    main()
