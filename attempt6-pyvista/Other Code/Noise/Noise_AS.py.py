import trimesh
import numpy as np
import random
import csv

# -----------------------------
# Deforming the spheres
# -----------------------------
def deform_sphere(sphere, num_blobs=1, blob_radius=5, blob_intensity=0.01):
    """
    Deform the sphere to create a 'blob' effect by displacing vertices based on random blob centres.
    
    Parameters:
      sphere (trimesh.Trimesh): The sphere mesh to deform.
      num_blobs (int): The number of random blobs to use for deformation.
      blob_radius (float): The sphere of influence for each blob.
      blob_intensity (float): The maximum displacement applied at the centre of a blob.
    
    Returns:
      trimesh.Trimesh: The deformed sphere mesh.
    """
    # Copy the vertices from the sphere mesh.
    vertices = sphere.vertices.copy()
    # Determine the bounding box of the sphere.
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Generate random blob centres within the bounding box.
    blob_centres = [np.random.uniform(min_coords, max_coords) for _ in range(num_blobs)]
    
    # For each vertex, accumulate the displacement from all blobs.
    for i, vertex in enumerate(vertices):
        total_displacement = np.zeros(3)
        for blob_centre in blob_centres:
            # Vector from the blob centre to the current vertex.
            direction = vertex - blob_centre
            distance = np.linalg.norm(direction)
            
            # Apply displacement if within the blob radius.
            if distance < blob_radius:
                # The influence diminishes linearly with distance.
                factor = 1 - (distance / blob_radius)
                
                # Compute the unit direction (avoiding division by zero).
                if distance != 0:
                    unit_direction = direction / distance
                else:
                    unit_direction = np.zeros(3)
                
                # Add to the total displacement.
                total_displacement += blob_intensity * factor * unit_direction
        
        # Update the vertex position.
        vertices[i] = vertex + total_displacement

    # Update the sphere's vertices with the new positions.
    sphere.vertices = vertices
    return sphere

# -----------------------------
# Main code
# -----------------------------
def main():
    # Step 1: Load the main STL file.
    # Replace this path with the STL you want to modify.
    main_mesh = trimesh.load('/home/a/Documents/3YP/Code/attempt6-pyvista/2x2_MN_large.stl')
    
    # Determine the bounding box of the main mesh.
    min_coords, max_coords = main_mesh.bounds
    
    # Define a margin above the bottom of the model. For instance, 5% of the model’s total height.
    margin = (max_coords[2] - min_coords[2]) * 0.05
    
    # Prepare a list to store information about each sphere (for CSV output).
    sphere_data = []
    
    # Step 2: Choose how many spheres to add.
    num_spheres = 50  # Increase or decrease as you wish.
    
    # Step 3: For each sphere, do the following:
    for i in range(num_spheres):
        # Continuously sample until we find a point that is above the margin.
        valid_sample = False
        while not valid_sample:
            # Sample a random point on the mesh's surface.
            points, face_indices = trimesh.sample.sample_surface(main_mesh, count=1)
            point = points[0]
            face_index = face_indices[0]
            if point[2] > min_coords[2] + margin:
                valid_sample = True
        
        # Retrieve the normal at the sampled face.
        normal = main_mesh.face_normals[face_index]
        
        # Step 3.3: Randomise the sphere's radius.
        radius = random.uniform(5, 15)
        
        # Step 3.4: Create an icosphere with the chosen radius.
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        
        # Decide how the sphere is placed relative to the mesh:
        # - Half the time, place the sphere so that its surface touches the mesh.
        # - Other times, place its centre at the mesh surface.
        if random.choice([True, False]):
            # Surface-attached: translate so that the sphere just touches the surface point.
            sphere_centre = point + normal * radius
        else:
            # Centre-attached: place the sphere's centre exactly at the sampled point.
            sphere_centre = point
        
        # Apply the translation.
        sphere.apply_translation(sphere_centre)
        
        # Step 4: Deform each sphere slightly (optional, for "blob" effect).
        random_intensity_factor = random.uniform(0.01, 0.5)
        deformation_intensity = random_intensity_factor * radius
        deformed_sphere = deform_sphere(
            sphere,
            num_blobs=20,
            blob_radius=radius * 1.2,
            blob_intensity=deformation_intensity
        )
        
        # Step 5: Decide randomly whether to union or subtract this sphere from the main mesh.
        # We will perform the boolean operation immediately on main_mesh.
        # Setting engine='scad' can help if you have OpenSCAD installed, but 'auto' often works too.
        operation_type = random.choice(['union', 'difference'])
        
        if operation_type == 'union':
            # Merge the new sphere with the main mesh to add new volume.
            main_mesh = trimesh.boolean.union([main_mesh, deformed_sphere], engine='auto')
        else:
            # Subtract the sphere from the main mesh to create "holes".
            main_mesh = trimesh.boolean.difference(main_mesh, deformed_sphere, engine='auto')
        
        # Store sphere data in a list for the CSV file.
        sphere_data.append({
            'x': sphere_centre[0],
            'y': sphere_centre[1],
            'z': sphere_centre[2],
            'radius': radius
        })
    
    # Step 6: Write the sphere data to a CSV file.
    with open('sphere_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z', 'radius']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in sphere_data:
            writer.writerow(data)
    
    # Step 7: Export the modified main mesh to a new STL file.
    main_mesh.export('2x2_MN_Array_scaled+Noise_with_subtractions.stl')

if __name__ == "__main__":
    main()
